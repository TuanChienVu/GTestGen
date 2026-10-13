"""Lane databases and the preflight checks run before (and during) a batch.

The lanes live in container gtestgen-er-db-1 (compose project gtestgen-er, MySQL on tmpfs, network gtestgen-er_er with
no Internet). What init-db.sh sets up, and what preflight/check_accounts verify before a batch:
  - root has no network account (MYSQL_ROOT_HOST=localhost): only `docker exec` into the db container (the socket)
    reaches it, so a run cannot use root; the migrations run as a temporary er_migrate that is dropped afterwards;
  - a run connects as er_laneN@'%', with exactly SELECT/INSERT/UPDATE/DELETE on sylius_laneN and a random password
    rotated at every init-db, so it cannot create triggers, views, routines, events or tables, change the schema, or
    reach another lane's database;
  - check_lane pins 83 tables and a fingerprint of the lane schema (tables, columns, indexes, constraints, foreign
    keys) plus the counts of triggers/routines/events/views in every non-system schema, and requires MIN_FREE_BYTES
    free on the db tmpfs; the batch re-checks it after every run that did not PASS. DML state is not checked here:
    Sylius' Behat hooks purge the lane before a scenario.
Where the lane password is: SECRETS_DIR/laneN.pw on the host (mode 600); the environment of the docker CLI and of the
lane container (DATABASE_URL), hence Docker's container config (`docker inspect`: visible to whoever controls Docker on
this host, who controls the db anyway); the run's PHP process, model code included (it only grants that lane's DML).
Never on a command line (argv), never in the results or in an error string (er.runner masks it before any truncation),
and the harness never writes it to a log (model code, in the same process, could print its own lane's URL).
init-db.sh rotates the passwords: it refuses to run while a lane lock is held or a gtestgen-er-lane* container exists.
Never run `docker compose ... up -d db` during a batch: a changed config recreates the container and wipes the tmpfs.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

DB_CONTAINER = "gtestgen-er-db-1"
NETWORK = "gtestgen-er_er"
N_TABLES = 83
# schema_fingerprint() of a freshly migrated sylius_laneN (c096ca3, 26 migrations; 83 tables, 285 index columns,
# 107 foreign keys, no trigger/routine/event/view), identical for lanes 1-3 (pinned 24/09 after init-db.sh)
SCHEMA_FINGERPRINT = "22ec61e5a5e8131ee2dd95bf539003236ac65a3b7be1c960d634d45c1e39b365"
SECRETS_DIR: Path = Path(os.environ.get("ER_SECRETS_DIR", Path.home() / ".config" / "gtestgen-er"))
MIN_FREE_BYTES = 200 * 2**20  # free space on the db tmpfs (/var/lib/mysql)
DOCKER_TIMEOUT = 30
DOCKER_ATTEMPTS = 2
SYSTEM_SCHEMAS = "('mysql', 'sys', 'information_schema', 'performance_schema')"
SYSTEM_USERS = {"mysql.infoschema", "mysql.session", "mysql.sys"}
RESET = "run ER_LANES={lane} ER_RESET=1 er/docker/init-db.sh"  # outside any batch


class InfraError(Exception):
    """The environment, not the output under test, failed: the batch must stop (nothing is recorded as done)."""


def _docker(cmd: list[str], what: str, timeout: int = DOCKER_TIMEOUT, env: dict[str, str] | None = None) -> str:
    """A failed docker call (non-zero exit, timeout, OSError) is retried once, then InfraError. Every use of it by the
    harness is a read-only check or query, so a retry is safe; sql() is a plain root executor, so a caller that sends
    DDL (only the real-run test does) must tolerate a statement that ran although its first attempt timed out."""
    for attempt in range(1, DOCKER_ATTEMPTS + 1):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            why = f"{what}: no answer from docker within {timeout} s"
        except OSError as exc:
            why = f"{what}: {exc}"
        else:
            if out.returncode == 0:
                return out.stdout.strip()
            why = f"{what}: {out.stderr.strip()[-200:]}"
        if attempt < DOCKER_ATTEMPTS:
            time.sleep(1)
    raise InfraError(why)


def lane_password(lane: int) -> str:
    p = SECRETS_DIR / f"lane{lane}.pw"
    try:
        pw = p.read_text(encoding="utf-8").strip()
    except OSError:
        pw = ""
    if not pw:
        raise InfraError(f"lane {lane}: no password in {p} (run er/docker/init-db.sh)")
    return pw


def db_url(lane: int) -> str:
    return (f"mysql://er_lane{lane}:{lane_password(lane)}@db:3306/sylius_lane{lane}"
            "?serverVersion=8.4.0&charset=utf8mb4")


def sql(query: str) -> str:
    """Run `query` as root over the socket inside the db container (root has no network account)."""
    return _docker(["docker", "exec", "-e", "MYSQL_PWD=root", DB_CONTAINER, "mysql", "-uroot", "-N", "-e", query],
                   "mysql")


def table_count(lane: int) -> int:
    return int(sql(f"select count(*) from information_schema.tables where table_schema='sylius_lane{lane}'"))


def schema_fingerprint(lane: int) -> str:
    """sha256 of the lane schema without its name (all lanes agree) + global trigger/routine/event/view counts."""
    s = f"sylius_lane{lane}"
    parts = [  # (row, order, source): one GROUP_CONCAT each
        ("table_name, table_type", "table_name", f"tables WHERE table_schema='{s}'"),
        ("table_name, column_name, column_type, is_nullable, IFNULL(column_default,'~'), extra",
         "table_name, ordinal_position", f"columns WHERE table_schema='{s}'"),
        ("table_name, index_name, non_unique, seq_in_index, IFNULL(column_name,'~'), IFNULL(sub_part,'~'), "
         "IFNULL(expression,'~')", "table_name, index_name, seq_in_index", f"statistics WHERE table_schema='{s}'"),
        ("table_name, constraint_name, constraint_type, enforced", "table_name, constraint_name, constraint_type",
         f"table_constraints WHERE table_schema='{s}'"),
        ("table_name, constraint_name, referenced_table_name, unique_constraint_name, match_option, update_rule, "
         "delete_rule", "table_name, constraint_name", f"referential_constraints WHERE constraint_schema='{s}'"),
        ("table_name, constraint_name, ordinal_position, column_name, referenced_table_name, referenced_column_name",
         "table_name, constraint_name, ordinal_position",
         f"key_column_usage WHERE table_schema='{s}' AND referenced_table_name IS NOT NULL"),
    ]
    counts = [("trg", "triggers", "trigger_schema"), ("rtn", "routines", "routine_schema"),
              ("evt", "events", "event_schema"), ("viw", "views", "table_schema")]
    selects = [f"(SELECT GROUP_CONCAT(CONCAT_WS(':', {row}) ORDER BY {order} SEPARATOR ';') "
               f"FROM information_schema.{src})" for row, order, src in parts]
    selects += [f"(SELECT CONCAT('{tag}:', COUNT(*)) FROM information_schema.{t} WHERE {col} NOT IN {SYSTEM_SCHEMAS})"
                for tag, t, col in counts]
    return sql(f"SET SESSION group_concat_max_len = 16777216; SELECT SHA2(CONCAT_WS('|', {', '.join(selects)}), 256)")


def network_internal(name: str = NETWORK) -> bool:
    try:
        return _docker(["docker", "network", "inspect", "--format", "{{.Internal}}", name], "network") == "true"
    except InfraError:
        return False


def free_bytes() -> int:
    """Free bytes of /var/lib/mysql (tmpfs) in the db container."""
    out = _docker(["docker", "exec", DB_CONTAINER, "df", "-P", "-B1", "/var/lib/mysql"], "df")
    try:
        return int(out.splitlines()[-1].split()[3])
    except (IndexError, ValueError):
        raise InfraError(f"df: unexpected output {out[-200:]!r}") from None


def lane_connect(lane: int) -> None:
    """The lane user logs in from the er network, as a run does, and runs `select 1`; InfraError otherwise."""
    image = _docker(["docker", "inspect", "--format", "{{.Image}}", DB_CONTAINER], "db image")
    env = {**os.environ, "MYSQL_PWD": lane_password(lane)}
    out = _docker(["docker", "run", "--rm", "--network", NETWORK, "-e", "MYSQL_PWD", image, "mysql", "-hdb",
                   f"-uer_lane{lane}", "-N", "-e", "select 1", f"sylius_lane{lane}"], f"lane {lane} login", 60, env)
    if out != "1":
        raise InfraError(f"lane {lane} login: unexpected answer {out[-80:]!r}")


def check_lane(lane: int) -> None:
    """Raise InfraError unless the lane has the pinned schema (83 tables, fingerprint) and the db tmpfs has at least
    MIN_FREE_BYTES free (a full disk fails every later run of every lane)."""
    n = table_count(lane)
    if n != N_TABLES:
        raise InfraError(f"lane {lane}: {n} tables, expected {N_TABLES} ({RESET.format(lane=lane)})")
    fp = schema_fingerprint(lane)
    if fp != SCHEMA_FINGERPRINT:
        raise InfraError(f"lane {lane}: schema fingerprint {fp[:12]}… differs from the pinned one "
                         f"(poisoned or drifted: {RESET.format(lane=lane)})")
    free = free_bytes()
    if free < MIN_FREE_BYTES:
        raise InfraError(f"lane {lane}: db tmpfs has {free} bytes free, need {MIN_FREE_BYTES}")


def check_accounts(lanes: list[int]) -> None:
    """Raise InfraError unless the MySQL accounts are exactly root@localhost, the mysql.* system accounts and
    er_laneN@'%' (every lane in `lanes` present), each er_laneN with only DML on sylius_laneN."""
    rows = [tuple(line.split("\t")) for line in sql("select user, host from mysql.user").splitlines() if line]
    lane_users, bad = set(), []
    for user, host in rows:
        m = re.fullmatch(r"er_lane([1-9][0-9]*)", user)
        if (user, host) == ("root", "localhost") or (user in SYSTEM_USERS and host == "localhost"):
            continue
        if m and host == "%":
            lane_users.add(int(m.group(1)))
        else:
            bad.append(f"{user}@{host}")
    if bad:
        raise InfraError(f"unexpected MySQL accounts {', '.join(sorted(bad))} (run er/docker/init-db.sh)")
    missing = sorted(set(lanes) - lane_users)
    if missing:
        raise InfraError(f"no MySQL user for lanes {missing} (run er/docker/init-db.sh)")
    for n in sorted(lane_users):
        got = set(sql(f"show grants for 'er_lane{n}'@'%'").splitlines())
        want = {f"GRANT USAGE ON *.* TO `er_lane{n}`@`%`",
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON `sylius_lane{n}`.* TO `er_lane{n}`@`%`"}
        if got != want:
            raise InfraError(f"er_lane{n}: grants {sorted(got - want)} beyond DML on sylius_lane{n}, or missing "
                             f"{sorted(want - got)} (run er/docker/init-db.sh)")


def lane_healthy(lane: int) -> bool:
    """Never raises: the lane user can log in, the lane has the pinned schema and the db has free space."""
    try:
        lane_connect(lane)
        check_lane(lane)
        return free_bytes() >= MIN_FREE_BYTES
    except Exception:  # noqa: BLE001 - any failure means "not healthy"
        return False


def preflight(lanes: list[int]) -> dict:
    if not network_internal():
        raise InfraError(f"network {NETWORK} missing or not internal")
    check_accounts(lanes)
    for lane in lanes:
        lane_connect(lane)
        check_lane(lane)
    free = free_bytes()
    if free < MIN_FREE_BYTES:
        raise InfraError(f"db tmpfs: {free} bytes free, need {MIN_FREE_BYTES}")
    return {"network": NETWORK, "schema_fingerprint": SCHEMA_FINGERPRINT, "lanes": lanes}

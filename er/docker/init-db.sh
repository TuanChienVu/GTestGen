#!/usr/bin/env bash
# Start MySQL (never recreated here: tmpfs) and, for each lane, create sylius_laneN with the Sylius migrations and a
# DML-only user er_laneN@'%' with a new random password, written to $ER_SECRETS_DIR/laneN.pw (mode 600).
# root has no network account: SQL runs as root over the socket inside the db container; the migrations run as a
# temporary er_migrate@'%' (ALL on that lane's database only, random password), dropped again on exit.
# Idempotent; rerun after every restart of db (tmpfs). Never while a run or a batch uses the lanes (the passwords
# change): it refuses, before touching the db, while any lane lock is held or any gtestgen-er-lane* container exists.
#   ER_LANES     lanes to set up (default "1 2 3"; lane 3 is for tests)
#   ER_RESET=1   drop sylius_laneN first (a poisoned or drifted lane: er.db.check_lane says
#                `ER_LANES=<n> ER_RESET=1 er/docker/init-db.sh`)
#   ER_FORCE=1   skip the lane lock / container refusal (emergencies only)
#   ER_SECRETS_DIR  default .config/gtestgen-er under home;  ER_LOCK_DIR  lane locks of er.runner, default /tmp
# Passwords go to mysql on stdin and to docker through its environment: never on a command line or in the output.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=gtestgen-er/sylius:c096ca3
NETWORK=gtestgen-er_er
DB_CONTAINER=gtestgen-er-db-1
SECRETS_DIR="${ER_SECRETS_DIR:-$HOME/.config/gtestgen-er}"
LOCK_DIR="${ER_LOCK_DIR:-/tmp}"
if [ "${ER_FORCE:-0}" = 1 ]; then
  echo "warning: ER_FORCE=1: lane locks and lane containers not checked" >&2
else
  # a lane lock (er.runner.lane_lock, any lane) held by a run, the er tests or a batch; unopenable counts as held
  command -v python3 >/dev/null || { echo "init-db.sh: refused: python3 is needed to test the lane locks" >&2; exit 1; }
  held=""
  for f in "$LOCK_DIR"/gtestgen-er-lane*.lock; do
    [ -e "$f" ] || continue
    python3 -c "import fcntl,sys; f=open(sys.argv[1],'w'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)" "$f" \
      2>/dev/null || held="$held $f"
  done
  if [ -n "$held" ]; then
    echo "init-db.sh: refused: lane lock held:$held (a run or a batch is using the lanes and init-db rotates their" \
      "passwords); wait for it to finish, or ER_FORCE=1 in an emergency" >&2
    exit 1
  fi
  # a lane container: a run in progress, or the orphan of a crashed harness
  containers=$(docker ps -aq -f name=^gtestgen-er-lane) || { echo "init-db.sh: refused: docker ps failed" >&2; exit 1; }
  if [ -n "$containers" ]; then
    echo "init-db.sh: refused: lane containers exist ($(echo $containers)): a run is in progress, or remove the" \
      "orphans (docker ps -a -f name=^gtestgen-er-lane); ER_FORCE=1 in an emergency" >&2
    exit 1
  fi
fi
umask 077
mkdir -p "$SECRETS_DIR" && chmod 700 "$SECRETS_DIR"
docker compose -f compose.yaml up -d --wait --no-recreate db
# --no-recreate keeps a db container created from an older compose.yaml: say so (a warning, the lanes still work)
want=$(docker compose -f compose.yaml config --hash db 2>/dev/null | awk '$1 == "db" {print $2}') || want=""
have=$(docker inspect --format '{{index .Config.Labels "com.docker.compose.config-hash"}}' "$DB_CONTAINER" \
  2>/dev/null) || have=""
if [ -z "$want" ]; then
  echo "warning: could not compute the config-hash of compose.yaml's db service: not compared with $DB_CONTAINER" >&2
elif [ "$want" != "$have" ]; then
  echo "warning: $DB_CONTAINER config-hash ${have:-none} differs from compose.yaml's $want: a change of the db" \
    "service was not applied (--no-recreate); outside any batch run: docker compose -f er/docker/compose.yaml up -d" \
    "--wait --force-recreate db && er/docker/init-db.sh (wipes the lanes)" >&2
fi
sql() { docker compose -f compose.yaml exec -T -e MYSQL_PWD=root db mysql -uroot -N --batch; }  # SQL on stdin
cleanup() { echo "DROP USER IF EXISTS 'er_migrate'@'%';" | sql || echo "warning: could not drop er_migrate" >&2; }
trap cleanup EXIT
# a db container created before MYSQL_ROOT_HOST=localhost still has root@'%'
echo "DROP USER IF EXISTS 'root'@'%'; DROP USER IF EXISTS 'er_migrate'@'%';" | sql
for lane in ${ER_LANES:-1 2 3}; do
  db="sylius_lane${lane}" reset=""
  if [ "${ER_RESET:-0}" = 1 ]; then
    echo "DROP DATABASE IF EXISTS ${db};" | sql
    reset="dropped and re-migrated, "
  fi
  migrate_pw=$(openssl rand -hex 24)
  sql <<SQL
CREATE USER 'er_migrate'@'%' IDENTIFIED BY '${migrate_pw}';
GRANT ALL PRIVILEGES ON ${db}.* TO 'er_migrate'@'%';
SQL
  DATABASE_URL="mysql://er_migrate:${migrate_pw}@db:3306/${db}?serverVersion=8.4.0&charset=utf8mb4" \
    docker run --rm --network "$NETWORK" -e DATABASE_URL "$IMAGE" sh -c \
    'bin/console doctrine:database:create --if-not-exists -q && bin/console doctrine:migrations:migrate -n -q --allow-no-migration'
  echo "DROP USER 'er_migrate'@'%';" | sql
  pw=$(openssl rand -hex 24)
  sql <<SQL
CREATE USER IF NOT EXISTS 'er_lane${lane}'@'%' IDENTIFIED BY '${pw}';
ALTER USER 'er_lane${lane}'@'%' IDENTIFIED BY '${pw}' WITH MAX_USER_CONNECTIONS 10;
REVOKE ALL PRIVILEGES, GRANT OPTION FROM 'er_lane${lane}'@'%';
GRANT SELECT, INSERT, UPDATE, DELETE ON ${db}.* TO 'er_lane${lane}'@'%';
SQL
  printf '%s\n' "$pw" > "$SECRETS_DIR/lane${lane}.pw.tmp"
  mv -f "$SECRETS_DIR/lane${lane}.pw.tmp" "$SECRETS_DIR/lane${lane}.pw"
  n=$(echo "select count(*) from information_schema.tables where table_schema='${db}';" | sql)
  echo "${db}: ${reset}${n} tables, user er_lane${lane} (DML only, password in $SECRETS_DIR/lane${lane}.pw)"
  [ "$n" -eq 83 ] || {
    echo "lane ${lane}: ${n} tables, expected 83 (run ER_LANES=${lane} ER_RESET=1 er/docker/init-db.sh)" >&2; exit 1; }
done

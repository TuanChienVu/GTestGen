#!/usr/bin/env bash
# Runs inside a fresh container from the ER image. The image layer is the clean tree, so every run starts clean.
# Usage: er-inner <suite> <feature_path>:<line>
#   /overlay_base (read-only, optional): the ER tree of the record; /overlay (read-only, optional): condition files,
#   copied after the base. Results go to /out: inner.json, behat.log, junit/, trace.json (when ER_TRACE=1).
set -uo pipefail
suite="$1"; target="$2"; timeout_s="${ER_TIMEOUT:-60}"
cd /app
layers=()
for d in /overlay_base /overlay; do
  if [ -d "$d" ]; then layers+=("$d"); fi
done
# A new file or a non-PHP file (service/suite config) needs the Symfony container rebuilt (APP_DEBUG=0 never checks).
cold=0
for d in "${layers[@]}"; do
  while IFS= read -r -d '' f; do
    rel="${f#"$d"/}"
    if [ ! -f "/app/$rel" ] || [[ "$rel" != *.php ]]; then cold=1; fi
  done < <(find "$d" -type f -print0)
done
for d in "${layers[@]}"; do
  while IFS= read -r -d '' f; do
    rel="${f#"$d"/}"
    mkdir -p "$(dirname "/app/$rel")"; cp "$f" "/app/$rel"
  done < <(find "$d" -type f -print0)
done
[ "${ER_FORCE_COLD:-0}" = 1 ] && cold=1
[ "$cold" = 1 ] && rm -rf var/cache/test_cached
echo "{\"cold\": $cold}" > /out/inner.json
if [ "${ER_TRACE:-0}" = 1 ]; then
  export ER_TRACE_OUT=/out/trace.json
  export BEHAT_PARAMS='{"extensions": {"Sylius\\Behat\\ErTrace\\ErTraceExtension": {}}}'
fi
start=$(date +%s.%N)
timeout --kill-after=10 "$timeout_s" vendor/bin/behat --strict --no-interaction --no-colors \
  --suite="$suite" -f pretty -o std -f junit -o /out/junit "$target" > /out/behat.log 2>&1
code=$?
end=$(date +%s.%N)
peak=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo null)
oom=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || true)
echo "{\"cold\": $cold, \"behat_exit\": $code, \"behat_seconds\": $(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}'), \"mem_peak_bytes\": $peak, \"oom_kill\": ${oom:-0}}" > /out/inner.json
exit 0

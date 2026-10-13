#!/bin/sh
# MooseFS emulation (smoke pod 28/09, EU-RO-1): a RunPod /workspace can be
# a MooseFS FUSE mount (a RunPod network volume) that allows chmod but refuses every ownership change, even for root.
# Installed (copied) as `chown` and as `install` ahead of the real ones on PATH, this refuses any ownership change of
# a path under MOOSEFS_ROOTS (space-separated directories) the way that volume does, and hands every other call to
# the real binary, MOOSEFS_REAL_DIR/<its own name> (default /usr/bin: the gate's own env -i test runs carry no
# MOOSEFS_* variable, and a test there calling install or chown must reach the real one). Used by
# ops/pod_parity_gate.sh's container (every gate run covers a MooseFS workspace) and by tests_gpu/test_ops_scripts.py;
# never installed on a pod. An ownership change is -o/-g in any form install accepts (separate or attached value, a short
# option cluster such as -dov2test, --owner/--group) or any chown. Limits, by design: only ABSOLUTE
# paths are matched against the roots (the bootstrap only ever passes absolute ones; resolving relative arguments
# would mistake an ownership spec or an option value for a path under the working directory), and symlinks are not
# followed.
name=${0##*/}
real=${MOOSEFS_REAL_DIR:-/usr/bin}/$name
own=0
[ "$name" = chown ] && own=1
for a in "$@"; do
  case "$a" in --owner|--owner=*|--group|--group=*) own=1 ;; --*) ;; -*[og]*) own=1 ;; esac
done
if [ "$own" = 1 ]; then
  for a in "$@"; do
    for r in ${MOOSEFS_ROOTS:-}; do
      case "$a" in
        "$r"|"$r"/*)
          echo "$name: cannot change owner and permissions of '$a': Operation not permitted" \
               "(MooseFS emulation, ops/moosefs_shim.sh)" >&2
          exit 1 ;;
      esac
    done
  done
fi
exec "$real" "$@"

#!/bin/zsh
# Phase 1 clones. Requires zsh (${0:A:h:h}, ${=var}). Full history without blobs (blob:none) and a sparse checkout of
# the .feature files, the step files and, when the source has "scoping", the Behat suite/service files
# (config/sources.json). Each repository is checked out DETACHED at the commit pinned in the dataset records,
# out/phase2/records/*.jsonl (scripts/fetch_full.py:pins, the same pins as the full clones; PHASE2=<dir> reads another
# copy of the records). Safe to re-run: any failed step stops the script, and the checkout must end at the pin.
set -eu -o pipefail
ROOT=${0:A:h:h}; mkdir -p $ROOT/repos-hist
python3 -c "
import json, sys; from pathlib import Path; sys.path.insert(0, '$ROOT/scripts'); from fetch_full import pins; pin = pins(Path('${PHASE2:-$ROOT/out/phase2}'))
for r in json.load(open('$ROOT/config/sources.json')):
    sc = r.get('scoping', {})
    scoping_globs = sc.get('suite_globs', []) + sc.get('service_globs', [])
    print(r['dir'], r['repo'], pin[r['repo']], *r['step_globs'])
    print('scoping', *scoping_globs)
" |
while read -r dir repo commit globs; do
  read -r tag scopingglobs
  d=$ROOT/repos-hist/$dir
  if [ ! -d $d ]; then
    git clone -q --filter=blob:none --no-checkout https://github.com/$repo.git $d
  fi
  (cd $d && git sparse-checkout set --no-cone '*.feature' ${=globs} ${=scopingglobs} && { git cat-file -e "$commit^{commit}" 2>/dev/null || git fetch -q origin $commit; } && git checkout -q --detach $commit)
  # Fetch every historical blob of the .feature and step files once (NOT the scoping suite/service files, which are
  # only needed at HEAD), so that git blame does not fetch them lazily one by one (38 s per file -> about 0.1 s).
  (cd $d && git log --format= --raw --no-abbrev --no-renames -m -- '*.feature' ${=globs} | awk '{print $4}' | grep -v '^0\{40\}$' | sort -u |
    git -c fetch.negotiationAlgorithm=noop fetch -q origin --no-tags --no-write-fetch-head --recurse-submodules=no --filter=blob:none --stdin)
  # sparse-checkout reads patterns with .gitignore rules (`*` does NOT match `/`) while `git ls-files` reads pathspecs
  # (`*` matches `/`): reset the sparse set to exactly the step and scoping files ls-files sees, so every one is on disk.
  (cd $d && { printf '%s\n' '*.feature'; git ls-files ${=globs} ${=scopingglobs} | sed 's|^|/|'; } | git sparse-checkout set --no-cone --stdin)
  (cd $d && test "$(git rev-parse HEAD)" = "$commit" && echo "$repo $(git rev-parse HEAD) commits=$(git rev-list --count HEAD) features=$(git ls-files '*.feature' | wc -l | tr -d ' ') stepfiles=$(git ls-files ${=globs} | wc -l | tr -d ' ') not_checked_out=$(git ls-files -t '*.feature' ${=globs} | grep '^S' | wc -l | tr -d ' ')")
done

#!/bin/sh
# QA R5-CCCC-N46-S2-D068-INDEPENDENT: the seven cases, s = 2, QA scratch only
export PYTHONIOENCODING=utf-8
Q=<QA_WORKTREE>
cd $Q
for c in "C4 4 1 9 9" "C5 5 1 11 11" "C6 6 1 13 13" "C7 5 0.6 20 13" "C1 4 1 32 32" "C2 5 1 32 32" "C3 6 1 32 32"; do
  set -- $c
  out=scratch/d068_s2/runs/$1
  rm -rf "$out"
  start=$(date +%s)
  .venv/Scripts/python.exe -m kkvim run-cccc-n46 --N $2 --a 1 --b $3 --t 0.01 --E 2.1e11 --nu 0.3 --q 1000 --nx $4 --ny $5 --half-steps 2 --out "$out" > "$out.log" 2>&1
  code=$?
  echo "$1 N=$2 b=$3 ${4}x$5 exit=$code seconds=$(( $(date +%s) - start ))"
done

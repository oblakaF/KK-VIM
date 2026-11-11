#!/bin/sh
# R5-CCCC-N46-S2-D068-FINAL: production runs on the non-Git snapshot (754dcfe + certificate of d78962a). QA scratch only.
# Physical input t = 0.01, E = 2.1e11, nu = 0.3; half_steps = 2. Run from the QA worktree root:  sh scratch/d068_final/run_cases.sh
export PYTHONIOENCODING=utf-8
F=scratch/d068_final
PY=.venv/Scripts/python.exe
# tag N a b nx ny q
for c in \
  "R_C4 4 1 1 9 9 1000" "R_C5 5 1 1 11 11 1000" "R_C6 6 1 1 13 13 1000" "R_C7 5 1 0.6 20 13 1000" \
  "R_C1 4 1 1 32 32 1000" "R_C2 5 1 1 32 32 1000" "R_C3 6 1 1 32 32 1000" \
  "B_pos 6 1 5 24 32 1000" "B_neg 6 1 5 24 32 -1000" \
  "C_pos 6 5 1 32 13 1000" "C_neg 6 5 1 32 13 -1000" \
  "D_N4_neg 4 1 1 32 32 -1000" "D_N5_neg 5 1 0.6 20 13 -1000"; do
  set -- $c
  out=$F/runs/$1
  rm -rf "$out"
  mkdir -p $F/runs
  start=$(date +%s)
  QA_MODULES=$F/runs/$1.modules.json $PY -I $F/kkcli.py run-cccc-n46 --N $2 --a $3 --b $4 --t 0.01 --E 2.1e11 --nu 0.3 --q $7 \
    --nx $5 --ny $6 --half-steps 2 --out "$out" > "$out.log" 2>&1
  code=$?
  echo "$1 N=$2 a=$3 b=$4 ${5}x$6 q=$7 exit=$code seconds=$(( $(date +%s) - start ))"
done

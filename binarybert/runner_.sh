#!/bin/bash
# Evaluation for robustness
for a in `seq 0 9`;
do
    sh scripts/eval_SST2.sh
done
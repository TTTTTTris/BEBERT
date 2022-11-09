#!/bin/bash
# Evaluation for robustness
for a in `seq 0 9`;
do
    sh scripts/train_sst2.sh
done
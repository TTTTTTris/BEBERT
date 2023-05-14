#!/bin/bash
# for i in {0..2};do ;done
for a in `seq 0 9`;
do
    sh scripts/train_sst2.sh
done
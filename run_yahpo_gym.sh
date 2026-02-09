#!/bin/bash

set -xeuo pipefail

scenario="rbv2_super"
instances=(1053 1457 1063 1479 15 1468)
evaluator="cond_ped_anova"
n_trials=1000

for instance in "${instances[@]}"; do
    python -m experiments.run_yahpo_gym \
        --scenario "$scenario" \
        --instance "$instance" \
        --evaluator-name "$evaluator" \
        --n-trials "$n_trials"

    python -m experiments.plot_yahpo_gym \
        --input-path "results/yahpo_${scenario}_${instance}_${evaluator}_${n_trials}trials.pkl"
done

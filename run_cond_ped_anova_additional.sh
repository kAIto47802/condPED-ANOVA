#!/bin/bash

set -xeuo pipefail

n_trials=1000
objectives=("nested-conditions" "three-way-branching")

for objective in "${objectives[@]}"; do
    python -m experiments.run_cond_ped_anova \
        --objective-name "$objective" \
        --n-trials "$n_trials" \
        --region-quantiles 1.0

    python -m experiments.plot_mixed \
        --input-paths "results/${objective}_cond_ped_anova_${n_trials}trials.pkl" \
        --titles \
        --save-name "cond_ped_anova_${objective}_${n_trials}trials" \
        --legend-loc "center left" \
        --legend-bbox-to-anchor -0.01 0.5 \
        --font-scale 0.8
done

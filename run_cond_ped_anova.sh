#!/bin/bash

set -xeuo pipefail

n_trials=1000
objectives=("activation-disjoint" "activation-overlap" "regime-dependent-domain")

results=()
for objective in "${objectives[@]}"; do
    python -m experiments.run_cond_ped_anova \
        --objective-name "$objective" \
        --n-trials "$n_trials"
    results+=("results/${objective}_cond_ped_anova_${n_trials}trials.pkl")
done

python -m experiments.plot_mixed \
    --input-paths "${results[@]}" \
    --titles \
    --save-name "cond_ped_anova_${n_trials}trials" \
    --legend-loc "upper right" \
    --legend-column 2

for result in "${results[@]}"; do
    python -m experiments.plot_cond_ped_anova \
        --input-path "$result" \
        --data-indices 1 2
done

#!/bin/bash

set -xeuo pipefail

n_trials_all=(100 500)
objectives=("activation-disjoint" "activation-overlap" "regime-dependent-domain")

for n_trials in "${n_trials_all[@]}"; do
    results=()
    titles=()
    for objective in "${objectives[@]}"; do
        python -m experiments.run_cond_ped_anova \
            --objective-name "$objective" \
            --n-trials "$n_trials"
        results+=("results/${objective}_cond_ped_anova_${n_trials}trials.pkl")
        titles+=("\$N=${n_trials}\$")
    done

    python -m experiments.plot_mixed \
        --input-paths "${results[@]}" \
        --titles "${titles[@]}" \
        --save-name "cond_ped_anova_${n_trials}trials" \
        --legend-loc "upper right" \
        --legend-column 2
done

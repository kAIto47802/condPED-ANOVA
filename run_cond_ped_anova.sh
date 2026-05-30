#!/bin/bash

set -xeuo pipefail

n_trials=1000
objectives=("activation-disjoint" "activation-overlap" "regime-dependent-domain")
samplers=("random" "tpe")

for sampler in "${samplers[@]}"; do
    # For backward compatibility, add the sampler name only for non-random samplers.
    if [ "$sampler" != "random" ]; then
        sampler_suffix="_${sampler}"
    else
        sampler_suffix=""
    fi

    results=()
    for objective in "${objectives[@]}"; do
        python -m experiments.run_cond_ped_anova \
            --objective-name "$objective" \
            --n-trials "$n_trials" \
            --sampler "$sampler"
        results+=("results/${objective}_cond_ped_anova${sampler_suffix}_${n_trials}trials.pkl")
    done

    python -m experiments.plot_mixed \
        --input-paths "${results[@]}" \
        --titles \
        --save-name "cond_ped_anova${sampler_suffix}_${n_trials}trials" \
        --legend-loc "upper right" \
        --legend-column 2

    for result in "${results[@]}"; do
        python -m experiments.plot_cond_ped_anova \
            --input-path "$result" \
            --data-indices 1 2
    done
done

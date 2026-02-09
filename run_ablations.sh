#!/bin/bash

set -xeuo pipefail

n_trials=1000
objectives=("activation-disjoint" "activation-overlap" "regime-dependent-domain")
evaluators=("local_marginal_variance" "wo_regime_probabilities")

for evaluator in "${evaluators[@]}"; do
    results=()
    for objective in "${objectives[@]}"; do
        python -m experiments.run_cond_ped_anova \
            --evaluator "$evaluator" \
            --objective-name "$objective" \
            --n-trials "$n_trials"
        results+=("results/${objective}_${evaluator}_${n_trials}trials.pkl")
    done

    python -m experiments.plot_mixed \
        --input-paths "${results[@]}" \
        --titles \
        --save-name "ablation_${evaluator}_${n_trials}trials" \
        --legend-loc "upper right" \
        --legend-column 2 \
        --height 2.6

    if [ "$evaluator" = "wo_regime_probabilities" ]; then
        ylabels=("HPI" "$\\tilde v_{\\gamma,\\mathrm{within}}^{(d)}$")

        python -m experiments.plot_both \
            --input-paths "${results[@]}" \
            --titles \
            --save-name "ablation_${evaluator}_${n_trials}trials" \
            --legend-loc "upper right" \
            --legend-column 2 \
            --ylabels "${ylabels[@]}" \
            --height 2.6
    fi
done

#!/bin/bash

set -xeuo pipefail

run_and_plot_baselines() {
    local -n objective_names="$1"
    local -n handling_strategies="$2"
    local n_trials="${3:-1000}"
    local evaluators=("ped_anova" "fanova" "mdi" "shap")
    local evaluator_names=("PED-ANOVA ($\\gamma=1.0$)" "f-ANOVA" "MDI" "SHAP")

    for objective_name in "${objective_names[@]}"; do
        local results=()
        local titles=()
        for i in "${!evaluators[@]}"; do
            local evaluator="${evaluators[$i]}"
            for strategy in "${handling_strategies[@]}"; do
                echo "Running evaluator: ${evaluator} with strategy: ${strategy} for objective: ${objective_name}"
                if [ "$evaluator" = "ped_anova" ]; then
                    python -m experiments.run_cond_ped_anova \
                        --objective-name "$objective_name" \
                        --evaluator-name "${evaluator}_${strategy}" \
                        --region-quantiles 1.0 \
                        --n-trials "$n_trials"
                else
                    python -m experiments.run_baselines \
                        --objective-name "$objective_name" \
                        --evaluator-name "${evaluator}_${strategy}" \
                        --n-trials "$n_trials"
                fi
                results+=("results/${objective_name}_${evaluator}_${strategy}_${n_trials}trials.pkl")
                titles+=("${evaluator_names[$i]} w/ ${strategy^}")
            done
        done

        echo "Plotting results for objective: ${objective_name}"
        python -m experiments.plot_mixed \
            --input-paths "${results[@]}" \
            --titles "${titles[@]}" \
            --save-name "baselines_${objective_name}_${n_trials}trials" \
            --height 3.5
    done
}

objective_names=("activation-disjoint" "activation-overlap")
handling_strategies=("filtering" "imputation")

run_and_plot_baselines objective_names handling_strategies

objective_names=("regime-dependent-domain")
handling_strategies=("expansion")

run_and_plot_baselines objective_names handling_strategies

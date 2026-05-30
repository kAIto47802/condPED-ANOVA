#!/bin/bash

set -xeuo pipefail

evaluators=(
    "cond_ped_anova"
    "ped_anova_filtering"
    "ped_anova_imputation"
    "fanova_filtering"
    "fanova_imputation"
    "mdi_filtering"
    "mdi_imputation"
    "shap_filtering"
    "shap_imputation"
)
n_trials=1000

run_yahpo_suite() {
    local scenario="$1"
    local instances_var="$2"
    local metric="$3"
    local stat_key="$4"
    local -n instances="$instances_var"

    for instance in "${instances[@]}"; do
        for evaluator in "${evaluators[@]}"; do
            python -m experiments.run_yahpo_gym \
                --scenario "$scenario" \
                --instance "$instance" \
                --evaluator-name "$evaluator" \
                --n-trials "$n_trials" \
                --metric "$metric" \
                --stat-key "$stat_key"

            python -m experiments.plot_yahpo_gym \
                --input-path "results/yahpo_${scenario}_${instance}_${evaluator}_${n_trials}trials.pkl"
        done
    done

    python -m experiments.run_yahpo_comparison \
        --scenario "$scenario" \
        --instances "${instances[@]}" \
        --evaluator-names "${evaluators[@]}" \
        --n-trials "$n_trials"
}

rbv2_instances=(1053 1457 1063 1479 15 1468)
iaml_instances=(40981 41146 1489 1067)

run_yahpo_suite "rbv2_super" rbv2_instances "acc" "learner_id"
run_yahpo_suite "iaml_super" iaml_instances "f1" "learner"

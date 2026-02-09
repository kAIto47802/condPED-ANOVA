#!/bin/bash

set -xeuo pipefail

python -m experiments.run_runtime_comparison \
    --objective-name "activation-disjoint"

python -m experiments.plot_times \
    --input-path "results/activation-disjoint_runtimes.npz"

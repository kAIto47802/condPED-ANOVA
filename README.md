<h1 align="center">
  <a href="https://github.com/kAIto47802/condPED-ANOVA">
    <img width="94%" height="13px" src="assets/titleLine1t.svg" />
  </a>
  Conditional PED-ANOVA: Hyperparameter Importance in Hierarchical & Dynamic Search Spaces
  <a href="https://github.com/kAIto47802/condPED-ANOVA">
    <img width="94%" height="9px" src="assets/titleLine1b.svg" />
  </a>
</h1>

<p align="center">
  Kaito Baba &emsp; Yoshihiko Ozaki &emsp; Shuhei Watanabe
</p>

<p align="center">
  We propose <em>conditional PED-ANOVA (condPED-ANOVA)</em>, a principled framework for estimating hyperparameter importance (HPI) in conditional search spaces, where the presence or domain of a hyperparameter can depend on other hyperparameters.
  Although the original PED-ANOVA provides a fast and efficient way to estimate HPI within the top-performing regions of the search space, it assumes a fixed, unconditional search space and therefore cannot properly handle conditional hyperparameters.
  To address this, we introduce a conditional HPI for top-performing regions and derive a closed-form estimator that accurately reflects conditional activation and domain changes.
  Experiments show that naive adaptations of existing HPI estimators yield misleading or uninterpretable importances in conditional settings, whereas condPED-ANOVA consistently provides meaningful importances that reflect the underlying conditional structure.
</p>

<div  align="center">
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue" alt="Python" />
  </a>
  <a href="http://arxiv.org/abs/2601.20800">
    <img src="https://img.shields.io/badge/arXiv-2601.20800-b31b1b.svg" alt="arXiv" />
  </a>
</div>

<br />

<h2 align="center">
  <div>🚀 Quick Start 🚀</div>
  <a href="https://github.com/kAIto47802/condPED-ANOVA">
    <img width="80%" height="8px" src="assets/line.svg" />
  </a>
</h2>

✨️ `cond_ped_anova` is fully compatible with [Optuna](https://github.com/optuna/optuna)’s built-in hyperparameter importance API.

## 1. Installation

```bash
uv add git+https://github.com/kAIto47802/condPED-ANOVA.git
# If you're using pip:
# pip install git+https://github.com/kAIto47802/condPED-ANOVA.git
```

## 2. Usage

```python
import optuna
from cond_ped_anova import CondPedAnovaImportanceEvaluator


def objective(trial: optuna.trial.Trial) -> float:
    c = trial.suggest_float("c", 0.0, 1.0)
    if c < 0.5:
        x = trial.suggest_float("x", -5.0, -2.0)
        return x
    else:
        y = trial.suggest_float("y", 2.0, 5.0)
        return y


sampler = optuna.samplers.RandomSampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=1000)

evaluator = CondPedAnovaImportanceEvaluator()
importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
```

<h2 align="center">
  <div>📊 Reproducing the Results 📊</div>
  <a href="https://github.com/kAIto47802/condPED-ANOVA">
    <img width="80%" height="8px" src="assets/line.svg" />
  </a>
</h2>

🔁 All experiments in [our paper](https://arxiv.org/abs/2601.20800) are fully reproducible.

## 1. Clone this Repository & Install Dependencies

```bash
git clone https://github.com/kAIto47802/condPED-ANOVA.git
```

```bash
uv sync --python 3.13 --extra experiments
# If you're using pip:
# pip install --upgrade pip  # enable PEP 660 support
# pip install -e '.[experiments]'
```

## 2. Run Experiments

Running the commands below reproduces the corresponding experiments and regenerates the exact figures:

- condPED-ANOVA on synthetic problems (Figures 1, 7, and 11):
   ```bash
   ./run_cond_ped_anova.sh
   ```

- Baseline methods on synthetic problems (Figures 2, 9, and 10):
   ```bash
   ./run_baselines.sh
   ```

- Ablation study (Figures 3 and 4):
    ```bash
    ./run_ablations.sh
    ```

- condPED-ANOVA on real-world benchmarks (Figures 5 and 14 and Tables 1 and 2):
    ```bash
    # prepare YAHPO Gym data
    git clone https://github.com/slds-lmu/yahpo_data.git

    ./run_yahpo_gym.sh
    ```

- Runtime comparison (Figure 6):
    ```bash
    ./run_runtime_comparison.sh
    ```

- condPED-ANOVA Results with Different $N$ (Figure 8):
    ```bash
    ./run_cond_ped_anova_different_n.sh
    ```

- Additional Experiments on Synthetic
Problems (Figures 12 and 13):
    ```bash
    ./run_cond_ped_anova_additional.sh
    ```

## 3. Check the Results

The generated figures will be saved in the `figures/` directory.
The raw experiment outputs (pickled results) and the values used in the tables are saved under `results/`.

> [!NOTE]
> If a LaTeX environment is not available, the figures will be rendered without LaTeX, which may slightly change font rendering and layout.


<h2 align="center">
  <div>🔖 Citation 🔖</div>
  <a href="https://github.com/kAIto47802/condPED-ANOVA">
    <img width="80%" height="8px" src="assets/line.svg" />
  </a>
</h2>

If you find condPED-ANOVA useful in your research, please consider citing the following paper:

```bibtex
@article{baba2026condpedanova,
  title={Conditional {PED-ANOVA}: Hyperparameter Importance in Hierarchical \& Dynamic Search Spaces},
  author={Baba, Kaito and Ozaki, Yoshihiko and Watanabe, Shuhei},
  journal={arXiv preprint arXiv:2601.20800},
  year={2026},
}
```

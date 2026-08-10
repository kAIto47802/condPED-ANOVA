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
  <strong>✨️ Accepted to the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining ✨</strong>
  <br />
  <strong> (KDD 2026) </strong>
</p>

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
  <!-- Use shields.io for the DOI badge to keep badge padding consistent. -->
  <a href="https://doi.org/10.5281/zenodo.20467920">
    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20467920-blue.svg" alt="DOI" />
  </a>
</div>
<div  align="center">
　<a href="http://arxiv.org/abs/2601.20800">
    <img src="https://img.shields.io/badge/arXiv-2601.20800-b31b1b.svg" alt="arXiv" />
  </a>
  <a href="https://github.com/kAIto47802/condPED-ANOVA/blob/main/poster.pdf">
    <img src="https://img.shields.io/badge/KDD%202026-Poster-blue.svg" alt="poster"/>
  </a>
</div>

<br />

<h2 align="center">
  <div>🚀 Quick Start 🚀</div>
  <a href="https://github.com/kAIto47802/condPED-ANOVA">
    <img width="80%" height="8px" src="assets/line.svg" />
  </a>
</h2>

`cond_ped_anova` is fully compatible with [Optuna](https://github.com/optuna/optuna)’s built-in hyperparameter importance API.

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

All experiments in [our paper](https://arxiv.org/abs/2601.20800) are fully reproducible.

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

- condPED-ANOVA on synthetic problems (Figures 1, 7, 11, and 12):
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

- condPED-ANOVA and baseline methods on real-world benchmarks (Figures 5 and 15 and Tables 1, 2, 3, and 4):
    ```bash
    # prepare YAHPO Gym data
    git clone https://github.com/slds-lmu/yahpo_data.git

    ./run_yahpo_gym.sh
    ```

- Runtime comparison (Figure 6):
    ```bash
    ./run_runtime_comparison.sh
    ```

- condPED-ANOVA results with different $N$ (Figure 8):
    ```bash
    ./run_cond_ped_anova_different_n.sh
    ```

- Additional experiments on synthetic problems (Figures 13 and 14):
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
@inproceedings{baba2026condpedanova,
  title     = {Conditional {PED-ANOVA}: Hyperparameter Importance in Hierarchical \& Dynamic Search Spaces},
  author    = {Baba, Kaito and Ozaki, Yoshihiko and Watanabe, Shuhei},
  year      = {2026},
  month     = {August},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  publisher = {Association for Computing Machinery},
  address   = {Jeju Island, Republic of Korea},
  doi       = {10.1145/3770855.3817758},
  isbn      = {979-8-4007-2259-2/2026/08},
}
```

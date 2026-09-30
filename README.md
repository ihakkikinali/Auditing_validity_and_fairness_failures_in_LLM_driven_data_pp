# Robustness and agent-level analyses – replication package

Reproduces every table and figure of Section 6 ("Robustness and External-Validity
Analyses") and the related Supplementary sections of *Auditing Validity and Fairness
Failures in LLM-Driven Data Preparation Pipelines*. **No LLM/API calls are made**:
each analysis holds a logged agent decision fixed (trap retained/removed; flagged
rows removed/kept) and re-draws splits, trap noise and flagged subsets.

## Environment
Python 3.12 with the pinned versions in `requirements.txt` (pandas < 3 is required to
reproduce the original German Credit encoding exactly): `pip install -r requirements.txt`.
Re-running the agent calls additionally needs `pip install -r requirements-agent.txt`
and the user's own API key; the robustness analyses need no API access.

## Data (`data/`)
| File | Source | Notes |
|---|---|---|
| `adult-all.csv` | UCI Adult (adult.data + adult.test, headerless) | rows 1–32,561 = adult.data; rows 32,562–48,842 = official test release |
| `german_credit_data_with_risk.csv` | Kaggle "German Credit Data with Risk" (UCI Statlog) | identical file used in the agent runs |
| `bank_additional_full.csv` | UCI Bank Marketing, bank-additional-full.csv (Moro et al., 2014) | 41,188 rows, chronological order |

## Run
```
cd code
python robustness_analyses.py german 50        # ~4 min
python robustness_analyses.py adult 20         # ~10 min (resumable: adult <end> <start>)
python robustness_analyses.py deletion 15      # ~20 min (resumable)
python robustness_analyses.py bank 0 -1        # chronological split
python robustness_analyses.py bank 10 0        # random splits + statistical screen
python aggregate.py                          # tables/, figures/, results/robustness_numbers.json
```
When running in chunks, concatenate the per-chunk CSVs into `adult_replay_all.csv`
and `deletion_replay_all.csv` before `aggregate.py`.

## Files
* `audit_core.py` – Adult pipeline transcribed from the Stage 2 notebook (trap generator, one-hot encoding, classifiers).
* `german_core.py` – German Credit pipeline transcribed from the Stage 3A notebook, including the original loader behaviour (`faithful=True`: headers replaced by `feature_0..feature_9`, target `class`).
* `robustness_analyses.py` – split replay, noise floor, imputation robustness, shift and external-test evaluation, deletion replay, Bank Marketing case study.
* `aggregate.py` – builds all LaTeX tables and the split-robustness figure; writes every quoted number to `results/robustness_numbers.json`.

## Agent-level runs (`agent_runs/`)
`run_new_agent_experiments.py` produced `decision_logs.jsonl` (E1 Bank `duration`, E2 German original names,
E4 repeated information ablation) and `decision_logs_E3v2.jsonl` (iterative code-executing agent, corrected tool).
`decision_logs_E3v1_discarded.jsonl` is the first iterative run, discarded because the tool returned no output
for trailing expressions. `code/e1_replay.py temporal` and `code/e1_replay.py 0,1,2` replay the Bank decisions;
`code/agg_agent.py` builds Tables 19–22 and `results/agent_numbers.json`.

## Additional analyses
* `code/extra_analyses.py` – clean utility regression (CUR) for the Bank Marketing decisions, specificity of the statistical screens on native features, parse/invalid-action counts over `agent_runs/stage_logs` and the agent-level logs, demographic-parity deltas, and the shift table with DVG and DVG^ex at all intensities.
* `code/legit_check.py` – PIG vs DVG^ex for legitimate Adult Income predictors (`capital.gain`, `relationship`).

## Input checksums (SHA-256)
* `data/adult-all.csv` 21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2
* `data/german_credit_data_with_risk.csv` 42be3b82a2e5073bd5ca23bce1d1c31426b78f72d20cd892f3aacaa2ba30a075
* `data/bank_additional_full.csv` 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8

## Agent-run Adult input
The original Adult agent experiments used `adult.csv` from the Kaggle distribution
`uciml/adult-census-income` (32,561 records, 15 columns). This file is not required by the
robustness scripts in this package; its original Kaggle path is preserved in the archived
notebooks (`notebooks/`).

## Notebooks (`notebooks/`)
Kaggle notebooks of the main experiments, as run: `stage1_adult_leakage_pilot`,
`stage2_adult_target_leakage_grid`, `stage2_1_adult_repeatability`,
`stage3a_german_proxy_leakage_grid`, `stage3a_1_german_repeatability`,
`stage3b_adult_minority_deletion_grid`, `stage3b_1_adult_deletion_repeatability`; and of the
agent-level analyses: `agent_level_experiments_kaggle` (Bank Marketing, German Credit with
original names, repeated information ablation) and `agent_level_iterative_agent_kaggle`
(iterative code-executing agent with the corrected tool). The per-call logs they produced
are in `agent_runs/` (`stage_logs/` for the main experiments).

## Licences
Code and notebooks: MIT License (`LICENSE`). The bundled datasets are redistributed under
their original terms: Adult Income, Bank Marketing (Moro et al., 2014) and Statlog German
Credit are UCI Machine Learning Repository datasets released under CC BY 4.0 (the German
Credit CSV is a public derivative of the UCI Statlog data); please cite the original
sources when using them.

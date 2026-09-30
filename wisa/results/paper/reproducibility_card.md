# Reproducibility Card

This card summarizes the public artifact boundary for this public artifact package.

## Included

- `scripts/`: preprocessing, splitting, window generation, synthetic generation, training, evaluation, and diagnostics.
- `config.yaml`: canonical paths, split policy, and 128/32 windowing configuration.
- `requirements.txt` and `requirements.lock.txt`: dependency specifications.
- `datasets/DOWNLOAD.md`: original dataset sources, checksums, licenses, and local placement paths.
- `results/tables/`: source CSVs for the manuscript numbers and diagnostic checks.
- `results/paper/figures/figure_37_evaluation_ladder.pdf`: external figure used by the manuscript.
- `results/paper/generator_rule_spec.md`: full rule card for generated attacks and stress tests.
- `results/paper/paper_artifacts_manifest.md`: current manuscript-to-artifact mapping.

## Intentionally Excluded

- Raw datasets and downloaded archives.
- Generated window arrays and synthetic windows.
- Trained model weights and standardizers.
- Per-run experiment logs and training histories.
- Historical copied `table_XX` CSVs and unused paper-ready figures.

## Core Rebuild Commands

Run these from the artifact repository root after downloading the public datasets into the paths described in `datasets/DOWNLOAD.md`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt

python scripts/preprocess_datasets.py
python scripts/validate_preprocessing.py
python scripts/split_datasets.py
python scripts/generate_windows.py
python scripts/generate_rule_based_synthetic.py
python scripts/generate_variant_test.py
python scripts/generate_variant_sensitivity.py
python scripts/generate_target_id_shift_stress.py
python scripts/generate_out_of_generator_stress.py

python scripts/train_real_only_baselines.py
python scripts/train_rule_synthetic_ratio_sweep.py
python scripts/train_channel_ablation.py
python scripts/train_rf_augmented.py
python scripts/run_seed_extension_stress.py --train-missing --evaluate

python scripts/evaluate_variant_baselines.py
python scripts/evaluate_variant_sensitivity.py
python scripts/evaluate_target_id_shift_stress.py
python scripts/evaluate_out_of_generator_stress.py
python scripts/run_cantt_tier1.py
python scripts/preprocess_road.py
python scripts/evaluate_road_external.py
python scripts/road_calibration_probe_logit.py
python scripts/run_cantt_rf_native_transfer.py
python scripts/make_support_shift_summary.py
python scripts/make_evaluation_ladder_figure.py
```

## Interpretation Boundaries

- External datasets are evaluation-only except for explicitly labeled diagnostic probes.
- External attack recall near 1.0 under FPR near 1.0 is a degenerate everything-is-attack failure, not successful transfer.
- The retained CSVs support the manuscript's bounded claim: rule-based augmentation improves controlled generated variants but does not solve cross-dataset shift.

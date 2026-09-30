> **졸업과제 저장소 안내**: 이 폴더는 WISA 2026 논문의 공개 재현 패키지를 그대로 옮긴 것이다. `requirements*.txt`는 저장소 최상위로, `datasets/DOWNLOAD.md`는 최상위 `datasets/`로 옮겼으며 `wisa/datasets`는 그 폴더를 가리키는 링크다. 설치는 최상위 [README](../README.md)의 5장을 따른다.

# Artifact: The Shift Ladder for Synthetic CAN Attack Augmentation

Reproduction package for the paper *"The Shift Ladder: Locating the Gains and Failures of Rule-Based Synthetic CAN Attack Augmentation"* (WISA submission).

The paper evaluates rule-based synthetic CAN attack augmentation with a layered, leakage-controlled Shift Ladder: an episode-split in-dataset test, generated variant/stress tests (fixed variant, sensitivity sweep, target-ID shift, out-of-generator payload position), three evaluation-only external datasets (OTIDS, can-train-and-test, ROAD), a can-train native/transfer RF sanity check, and an attribution chain (ID-whitelist baseline, ID-channel masking, channel-ablation retraining, label-free calibration/OOD probes, target-normal reinstantiation, and support-factor decomposition).

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Full experimental pipeline (preprocessing, splitting, windowing, generation, training, evaluation, diagnostics, paper artifacts) |
| `config.yaml` | Main experiment configuration (paths, split policy, windowing 128/32) |
| `requirements.txt` / `requirements.lock.txt` | Python dependencies (top-level / exact installed versions) |
| `results/tables/` | Source CSVs for every number in the paper |
| `results/paper/` | Current submission manifest, Fig. 1 PDF, `generator_rule_spec.md`, and `reproducibility_card.md`; copied historical table/figure artifacts are intentionally omitted |
| `datasets/DOWNLOAD.md` | Where to obtain the four public datasets and where to place them |

Trained weights, generated window arrays, raw datasets, per-run experiment logs, obsolete paper-ready figures, and copied `table_XX` CSVs are intentionally not redistributed. The package keeps the source CSV/figure artifacts needed to audit the reported numbers, and the scripts regenerate `models/`, `experiments/`, and derived dataset folders locally after the public datasets are downloaded.

## Environment

- Python 3.12, PyTorch (CUDA), scikit-learn, pandas, numpy, matplotlib (exact versions in `requirements.lock.txt`).
- Experiments were run on Linux (aarch64) with a single NVIDIA GPU; CPU-only runs work but are slower.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

## Datasets

None of the datasets are redistributed here; see `datasets/DOWNLOAD.md` for sources, checksums, licenses, and placement paths. Dataset roles are fixed by design and must not be changed when reproducing:

- **Car-Hacking** (HCRL): training, validation, synthetic generation, and in-dataset testing only.
- **OTIDS / CAN-Intrusion** (HCRL): external binary evaluation only (scenario-level labels).
- **can-train-and-test** (Lampe & Meng): external evaluation/profiling only.
- **ROAD** (ORNL, Zenodo, CC-BY-4.0): external evaluation only; never used for training, validation, or model selection. Accelerator captures are excluded (no injected frames).

## Reproduction pipeline

All scripts read `config.yaml` and write into `results/` and generated local output folders such as `models/` and `experiments/`. Seeds: main source/external CNN/RF results use 7, 42, 123; sensitivity/stress and channel-ablation diagnostics use 7, 42, 123, 2026, 3407; the training synthetic pool uses generator seed 314159 (see the rule card for per-generator seeds).

1. **Preprocess and split**
   ```bash
   python scripts/preprocess_datasets.py
   python scripts/validate_preprocessing.py
   python scripts/split_datasets.py        # episode split (1.0 s gap, 60/20/20)
   python scripts/generate_windows.py      # 128-frame windows, stride 32
   python scripts/profile_datasets.py
   ```
2. **Generate synthetic training pool and evaluation-only variants**
   ```bash
   python scripts/generate_rule_based_synthetic.py     # 260k training attack windows
   python scripts/generate_variant_test.py             # fixed variant (positive control)
   python scripts/generate_variant_sensitivity.py      # 18k variants + 12k normals
   python scripts/generate_target_id_shift_stress.py
   python scripts/generate_out_of_generator_stress.py
   python scripts/generate_filtered_synthetic.py
   ```
3. **Train**
   ```bash
   python scripts/train_real_only_baselines.py         # <=20 epochs, patience 4
   python scripts/train_rule_synthetic_ratio_sweep.py  # +10/30/50/100%; <=12 epochs, patience 3
   python scripts/train_oversampling_baseline.py       # +50% real oversampling
   python scripts/train_filtered_synthetic.py
   python scripts/train_channel_ablation.py            # ID-only / non-ID-only, 5 seeds
   python scripts/train_rf_augmented.py                # rule +30% RF (second model family)
   python scripts/build_target_normal_reinstantiation_cache.py
   python scripts/generate_target_normal_synthetic.py
   python scripts/train_target_normal_reinstantiation.py
   python scripts/train_support_factor_decomposition.py
   python scripts/train_rf_target_normal_reinstantiation.py
   ```
   This step regenerates `models/` locally; trained weights are not shipped in the anonymous package.
4. **Evaluate the ladder**
   ```bash
   python scripts/evaluate_variant_baselines.py        # rung 1-2 (+ OTIDS)
   python scripts/evaluate_variant_sensitivity.py      # rung 3
   python scripts/evaluate_target_id_shift_stress.py   # rung 4
   python scripts/evaluate_out_of_generator_stress.py  # rung 5
   python scripts/run_cantt_tier1.py                   # rung 7 (can-train external)
   python scripts/preprocess_road.py                   # frame labels from published metadata
   python scripts/evaluate_road_external.py            # rung 8
   python scripts/evaluate_rf_external.py              # RF model-family axis
   python scripts/run_seed_extension_stress.py --train-missing --evaluate
   ```
5. **Diagnostics and attribution (RQ4-RQ6, RQ9)**
   ```bash
   python scripts/profile_synthetic_quality.py         # real-vs-synthetic AUC, JS divergences
   python scripts/diagnose_synthetic_effects.py
   python scripts/diagnose_otids_normal_shift.py
   python scripts/evaluate_threshold_sensitivity.py    # threshold + oracle diagnostics
   python scripts/run_cantt_lightweight_baseline.py    # ID-whitelist baseline
   python scripts/run_cantt_id_ablation.py             # ID-channel input masking
   python scripts/evaluate_channel_ablation_external.py
   python scripts/evaluate_published_baselines.py      # entropy / per-ID timing baselines
   python scripts/road_calibration_probe.py            # label-free calibration probe
   python scripts/road_calibration_probe_logit.py      # max-logit / energy OOD-score probe
   python scripts/road_whitelist_baseline.py           # ID-whitelist baseline on ROAD
   python scripts/train_adaptation_probe.py            # diagnostic can-train adaptation probe
   python scripts/road_boundary_diagnostic.py
   python scripts/run_cantt_rf_native_transfer.py      # can-train native/transfer RF sanity check
   ```
6. **Aggregate summaries and regenerate the current paper figure**
   ```bash
   python scripts/aggregate_baseline_results.py
   python scripts/analyze_attack_type_recall.py
   python scripts/prepare_reinforcement_tables.py
   python scripts/make_evaluation_ladder_figure.py     # paper Fig. 1
   python scripts/make_support_shift_summary.py
   ```

## Mapping results to the paper

`results/paper/paper_artifacts_manifest.md` maps current manuscript tables and figures to source CSVs and generating scripts. Headline checks:

- Table 1 (synthetic rules) — inline manuscript table; full card in `results/paper/generator_rule_spec.md`
- Table 2 (Shift Ladder claim semantics) — inline manuscript table; dataset roles in `datasets/DOWNLOAD.md` and `config.yaml`
- Fig. 1 (evaluation ladder) — `results/paper/figures/figure_37_evaluation_ladder.pdf`
- Table 3 (five-seed sensitivity sweep) — `results/tables/main_seed_extension_variant_sensitivity_summary.csv`
- Table 4 (evidence matrix) — `results/tables/main_seed_extension_headline.csv`, external summaries, and repair/attribution summaries
- Table 5 (target-ID / out-of-generator stress) — `results/tables/main_seed_extension_target_id_shift_key_comparisons.csv`, `results/tables/main_seed_extension_out_of_generator_key_comparisons.csv`
- Table 6 (RF native/transfer diagnostic) — `results/tables/cantt_rf_native_transfer_matrix.csv`
- Table 7 (support-shift summary) — `results/tables/support_shift_summary.csv`; generated by `scripts/make_support_shift_summary.py`
- Table 8 (failure attribution and repair probes) — `results/tables/cantt_id_ablation_summary.csv`, `results/tables/channel_ablation_external_summary.csv`, `results/tables/cantt_lightweight_baseline_summary.csv`, `results/tables/road_whitelist_baseline.csv`, `results/tables/road_calibration_probe_logit_summary.csv`, `results/tables/target_normal_reinstantiation_compact_key.csv`, `results/tables/support_factor_decomposition_key_comparisons.csv`
- External FPR results — `results/tables/cantt_external_eval_summary_mean_std.csv`, `road_external_summary.csv`
- Calibration probes — `results/tables/road_calibration_probe_summary.csv`, `road_calibration_probe_logit_summary.csv` (max-logit/energy)
- ROAD whitelist baseline — `results/tables/road_whitelist_baseline.csv`
- Diagnostic adaptation probe — `results/tables/adaptation_probe_summary.csv` (per-seed in `adaptation_probe_by_seed.csv`)
- Augmented RF arm — `results/tables/rf_augmented_summary.csv`
- Target-normal reinstantiation — `results/tables/target_normal_reinstantiation_summary.csv`, `target_normal_reinstantiation_compact_key.csv`
- Support-factor decomposition — `results/tables/support_factor_decomposition_summary.csv`, `support_factor_decomposition_key_comparisons.csv`
- RF target-normal replication — `results/tables/rf_target_normal_reinstantiation_summary.csv`, `rf_target_normal_reinstantiation_key_comparisons.csv`
- Headline paired t-tests — `results/tables/statistical_tests.csv` (sensitivity split, +100% vs real-only: t=13.27, p=0.0056)

## Notes

- External datasets are evaluation-only by construction; do not use OTIDS/can-train-and-test/ROAD for training, tuning, or model selection. The single exception is the explicitly diagnostic adaptation probe (`scripts/train_adaptation_probe.py`), which adds can-train calibration-normal windows under a capture-level split, stores regenerated models separately in `models/adaptation/`, and informs no main claim.
- External attack recall near 1.0 under FPR near 1.0 is a degenerate everything-is-attack failure mode and must not be read as detection or transfer.
- Statistical reporting: mean±std over seeds; per-seed paired deltas are preferred over hypothesis tests at these seed counts (paper-facing per-seed values are in `results/tables/*_by_seed.csv` and the corresponding summary CSVs).

## License

Code is released under the MIT License (see `LICENSE`). Datasets remain under their original licenses (ROAD: CC-BY-4.0; HCRL datasets: see their terms).

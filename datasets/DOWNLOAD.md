# Dataset download and placement

None of the four datasets are redistributed in this repository. Download them from their original sources and place the archives at the paths below (paths are read from `config.yaml`).

## 1. Car-Hacking (HCRL) — train/val/test + synthetic generation

- Source: https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
- Place as: `datasets/9) Car-Hacking Dataset.zip`
- Role: the ONLY dataset used for training, validation, synthetic generation, and in-dataset testing.

## 2. CAN-Intrusion / OTIDS (HCRL) — external evaluation only

- Source: https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset
- Place as: `datasets/10) CAN-Intrusion Dataset.zip`
- Labels are scenario-level (weak); used for external binary evaluation only.

## 3. can-train-and-test (Lampe & Meng) — external evaluation only

- Source: see Lampe, B., Meng, W.: "can-train-and-test: A Curated CAN Dataset for Automotive Intrusion Detection", Computers & Security 140 (2024), doi:10.1016/j.cose.2024.103777 (distribution link in the paper).
- Place as: `datasets/can-train-and-test.zip`
- Expected archive: 1,507,455,719 bytes, SHA256
  `a9c607b38bd28f1768021ad01c29ffbfe4e82bb0ae5815ac3ce7ad74751ae061`,
  containing 236 CSV files under `set_01`–`set_04` with schema
  `timestamp,arbitration_id,data_field,attack` (DLC is derived from the hex payload length).

## 4. ROAD (ORNL) — external evaluation only

- Source: https://zenodo.org/records/10462796 (`road.zip`)
- Place as: `datasets/road.zip`
- MD5: `cab184cfc2fe12c0834bc46188c0f330`
- License: CC-BY-4.0. Citation: Verma et al., "A Comprehensive Guide to CAN IDS Data and Introduction of the ROAD Dataset", PLOS ONE 19(1), 2024.
- Per-frame labels are reconstructed by `scripts/preprocess_road.py` from the published injection intervals, target IDs, and payload patterns; the four accelerator captures are excluded (they contain no injected frames).

After downloading, run the pipeline from step 1 of the main README. Never use OTIDS, can-train-and-test, or ROAD for training, tuning, or model selection: they are evaluation-only by design.

# 3단계: 요인 교차·맞춤 대조군·관계 활용도 확장 검증

최종보고서 5–6장의 확장 실험 코드다. 공통 함수는 `scripts/lib_common.py`에 있고,
경로는 이 폴더(`journal/`)를 기준으로 한다. 데이터는 `datasets/`(저장소 최상위 `datasets/`로의 링크)에서 읽는다.

| 실험 | 내용 | 주요 스크립트 | 기록 |
|---|---|---|---|
| 기본 | L1–L4 기본 실험 (CNN·RF, 5개 학습 반복) | `generate_rule_based_synthetic_v2.py`, `train_generator_extension_cnn.py`, `train_generator_extension_rf.py`, `evaluate_generator_extension.py` | `results/tables/generator_extension_by_seed.csv` |
| E13 | +100% 증강군의 합성 표본을 중복 없이 다시 뽑은 보정 재실행, 외부 운영점 그림 | `build_e13_sampling_v2_analysis.py`, `external_operating_point_neural.py`, `make_e13_sampling_v2_figures.py` | `experiments/e13_strict_v2_sampling/` |
| E14 | L4 요인 교차: 값의 위치·배치·값 변화 규칙 | `run_e14_matched_real_training.py`, `evaluate_l4_counterfactual_factorial.py`, `analyze_l4_counterfactual_factorial.py` | `experiments/e14_l4_counterfactual_factorial/` |
| E15 | 새 생성 반복 4개에서 요인 교차 반복 | `generate_rule_construction_sensitivity.py`, `train_rule_construction_seed_crossing.py`, `evaluate_rule_construction_seed_crossing.py`, `analyze_rule_construction_seed_crossing.py` | `experiments/e15_rule_construction_crossing/` |
| E16 | 맞춤 대조군(placebo)으로 위치 이동 반응 분해, 20개 생성 반복 | `generate_e16_rule_placebo_pairs.py`, `run_e16_paired_training.py`, `evaluate_e16_relational_response.py`, `analyze_e16_relational_response.py` | `experiments/e16_relational_response_decomposition/` |
| E17–E19 | 같은 생성 반복에서 RF, 관계 특징 RF, BiLSTM 비교 | `run_e17_rf_paired_training.py`, `run_e18_relation_visible_rf_training.py`, `run_e19_bilstm_paired_training.py` 및 각 실험의 `evaluate_*`·`analyze_*` 스크립트 | `experiments/e17_*`, `e18_*`, `e19_*` |
| 양성 대조 | 관계가 분류에 유용하도록 설계한 합성 문제에서 효용 회복 확인 | `run_attribution_positive_control_v1.py {freeze,pilot,confirmation}`, `verify_attribution_positive_control_v1.py` | `experiments/attribution_strengthening_20260906/a_positive_control/` |
| 100회 재현 | 새 생성 반복 100개·CNN 1,000회 학습으로 관계 활용도 L(k)와 효용 U(k) 추정 | `run_attribution_local_replication_v1.py`, `run_attribution_local_grid_v1.py`, `evaluate_attribution_local_replication_v2.py` | `experiments/attribution_strengthening_20260906/b_local_replication/` |

## 실행 순서

각 실험은 `experiments/<실험>/PREREG.md`(또는 `PROTOCOL_v1.md`)에 적힌 순서대로 실행한다.
대체로 동결 → 합성 자료 생성 → 학습 → 채점 → 분석 순서이며, 각 단계는 이전 단계의 검사(gate)를
통과한 기록이 있어야 진행된다.

스크립트별 옵션은 `python scripts/<이름>.py --help`로 확인한다. 단, 인자 처리(argparse)가 없는 스크립트는
`--help`를 붙여도 바로 실행되어 결과 파일을 덮어쓴다. 해당 스크립트는 다음 11개이며, 실행 전에 파일 첫 부분의
설명을 먼저 읽는다.

- 그림: `make_external_operating_figure.py`, `make_focal_forest_figure.py`, `make_generator_protocol_valid_figure.py`,
  `make_grammar_exit_figure.py`, `make_headline_results_figure.py`, `make_ladder_overview_figure.py`, `make_protocol_figure.py`
- 그 밖: `make_e16_response_share_table.py`, `generate_rule_placebo_twin.py`, `stats_confirmatory.py`,
  `verify_attribution_local_pilot_v1.py`

## 포함 범위

- `scripts/`: 위 실험의 진입점과 이들이 import하는 모듈만 포함했다.
- `experiments/`: 사전 계획, 수정 기록, 실행·검사 기록(JSON)과 요약 결과. 개별 학습 단위의 출력은 제외했다.
- `results/tables/`: 보고서 수치의 근거가 되는 요약표. 대용량 시나리오별 표와 합성 윈도 배열, 학습 모델은 제외했다.
- 실행 기록에는 원래 연구 환경의 절대 경로와 커밋 해시가 남아 있다. 출처 추적용 기록이며, 다시 실행하면 새 기록이 생성된다.

## 공개용 정리

공개하면서 실험 기록의 운영 메모(작업 서비스 이름, 작업 환경 안내 등)를 지우거나 고쳤다. 실험 설계, 수치와
판정 내용은 바꾸지 않았다.

- 삭제: `b_local_replication/EVALUATION_LAUNCH_20260907_v2.md`, `b_local_replication/INTERRUPTION_AUDIT_20260906.md`
- 문장 수정: `b_local_replication/`의 `PROTOCOL_v1.md`, `RECOVERY_v1.md`, `EVALUATION_v2.md`, `RESULTS_20260913_v1.md`,
  `a_positive_control/RESULTS_v1.md`

이 중 `PROTOCOL_v1.md`, `RECOVERY_v1.md`, `EVALUATION_v2.md`는 동결 기록(`freeze*.json`, `recovery_preflight_v1.json`)에
SHA-256으로 묶여 있어, 수정 후의 해시가 기록된 값과 다르다. 기존 동결 기록을 그대로 검증하는 실행 단계
(예: `evaluate_attribution_local_replication_v2.py`의 freeze 검증)는 입력 변경 오류로 멈춘다. 테스트 결과에는 영향이 없다.

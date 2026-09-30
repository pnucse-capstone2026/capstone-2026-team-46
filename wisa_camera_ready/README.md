# 2단계: 동일 학습량·비중첩·외부 점수 순위 보강 실험

1단계(`../wisa/`)의 Shift Ladder 결과를 보강하기 위한 세 실험이다. 1단계의 데이터 분할,
윈도 배열과 학습 모델을 입력으로 사용하며, 결과는 이 폴더 안에만 기록한다.

| 실험 | 스크립트 | 기록 | 보고서 내용 |
|---|---|---|---|
| 동일 학습량 대조 | `scripts/run_matched_update_v1.py {preflight,train,evaluate,all}` | `experiments/02_matched_update/` | 실제 자료·실제 공격 복제·Rule +30%의 세 학습군을 같은 7,992회 갱신, 다섯 시드로 비교 |
| 원시 메시지 비중첩 평가 | `scripts/evaluate_overlap_unit_v1.py {preflight,evaluate}` | `experiments/03_overlap_unit_audit/` | 입력 구간이 겹치지 않는 평가 부분집합에서 증강 이득 유지 확인 |
| 외부 점수 순위 | `scripts/evaluate_external_ranking_v1.py` | `experiments/01_external_ranking/` | 세 외부 데이터의 AUROC·AP와 FPR 0.1%·1% 이내 TPR |

각 실험의 사전 계획은 `PLAN.md`, 결과 해석은 `RESULTS_v1.md`, 집계표는 `results/tables/`에 있다.
학습 모델 가중치(`models/`)는 용량 문제로 포함하지 않았으며 `train` 단계로 다시 만들 수 있다.

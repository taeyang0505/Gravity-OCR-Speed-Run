# Gravity OCR Speed Run — 1.974 s (team taktak)

Trillion Labs 해커톤 Track 2 "Gravity OCR Speed Run" 제출본과 그 과정입니다.
GravityOCR-KO로 크롭 402장(텍스트 210, 표 62, 수식 130)을 L40S 한 장에서 읽는 시간을 품질 합격선 안에서 줄이는 대회입니다.

| 제출 | 리더보드 시간 | 품질 | 내용 |
|---|---:|---:|---|
| 1등 팀 기록 (v14, submission 260) | 2.351 s | 97.19 | 공개 저장소 [sungu-che/minion](https://github.com/sungu-che/minion) |
| 318 `ty_m3d` | 2.282 s | 96.97 | 1등 v17 + 초안 학습 가중치 + 글자 크기 목표 11 px |
| **319 `ty_m4_lanes_c4`** | **1.974 s** | **96.85** | + 서버 2대(MPS) + 계산량 절감 4가지 (이 저장소) |

품질 합격선: 텍스트 ≥ 96.24, 수식 ≥ 94.88, 표 ≥ 97.50 (기준 모델 점수 − 1점).

## 변경 사항

출발점은 1등 팀의 제출본 v17입니다(Apache 2.0). 그 위에 아래 항목을 얹었고, 모두 기본값은 꺼짐이며 `submission/TY_SETTINGS`에서 켭니다.

| 변경 | 스위치 | 단독 효과 (RunPod L40S, dev 3회 중앙값) |
|---|---|---|
| 초안 학습 LM 가중치 (우리 278번 제출의 가중치, 바뀐 텐서 118개만) | `weights_delta/` | 2.35 → 2.29 s |
| 글자 크기 목표 12 → 11 px (1등 팀 스위치) | `TRACK2_TEXT_GLYPH`, `TRACK2_TABLE_GLYPH` | 2.29 → 2.21 s |
| 수식 크롭에도 글자 크기 축소 (14 px) | `TRACK2_FORMULA_GLYPH=14` | 2.18 → 2.15 s |
| 빈 2×2 묶음 ViT 건너뛰기, 호스트 동기화 없는 버전 (표 제외) | `SS_VIT_DROP=1`, `SS_VIT_DROP_TABLE=0` | 2.18 → 2.15 s |
| ViT MLP의 gate_up 행렬곱 + SiLU를 한 Triton 커널로 | `SS_FUSED_SILU_VIT=1` | 2.18 → 2.16 s |
| 글자 축소 대상을 패치 256개 이상으로 넓힘 (1등 팀 스위치) | `TRACK2_GLYPH_MIN_PATCHES=256` | 2.18 → 2.14 s |
| 위 네 가지 함께 (서버 1대) | | 2.18 → 2.06 s |
| **서버 2대 (CUDA MPS)**: 표 서버(초안 33, MPS 우선순위 낮춤) + 텍스트·수식 서버(초안 13) | `TRACK2_LANES=table;text,formula` 외 | **2.06 → 1.91 s** |

최종 구성을 제출 zip 그대로 돌린 결과(RunPod L40S, dev 402장, 새 프로세스 3회):

| | 3회 시간 | dev 텍스트 / 수식 / 표 | 홀드아웃 100장 텍스트 / 수식 / 표 |
|---|---|---|---|
| `ty_m4_lanes_c4` | 1.94 / 1.89 / 1.92 s (중앙값 1.92 s) | 96.91 / 95.19 / 98.63 | 96.79 / 96.16 / 99.85 |

왜 이 순서로 효과가 났는지, 효과가 없던 시도까지는 [docs/ANALYSIS.md](docs/ANALYSIS.md)에 정리했습니다.

## 디렉터리 구조

```
submission/   제출 zip의 내용 (가중치 제외). team.yaml -> ty_launch.py -> TY_SETTINGS를 읽고 serve.py 실행
patches/      1등 팀 v17 폴더에 우리 변경을 넣는 스크립트 (모두 기본 꺼짐), 제출 zip 만들기, 합친 커널과 벤치
tools/        측정 도구: 대회 하네스와 같은 방식의 dev 측정(dev_harness.py), 도커/네이티브 실행, 공식 채점, 홀드아웃 채점
tools/profiling/  nsys 프로파일(시간 측정 구간만), 커널 시간 분해, 요청별 타임라인과 라운드 수 분석
results/      각 실험의 설정과 시간(timing.json), 품질(scores.json) 요약
docs/         분석 문서
```

## 재현 방법

1. 대회 이미지 `track2-base:v1`(스타터 키트의 `track2-base/Dockerfile`)이 있는 L40S(드라이버 570)에서,
   1등 팀 저장소의 빌드 순서(`tools/build_kernels.py`, `tools/build_draft_attn.py`, `tools/build_overlay.py`)로 v17 팀 폴더를 만듭니다.
   이 저장소의 `submission/`에는 그렇게 빌드한 결과(`gravity_ocr/_native/*.so`, SGLang overlay)가 이미 들어 있습니다.
2. `patches/pack_minion_submission.sh <이름> patches/TY_SETTINGS` 가 위 폴더에 패치를 넣고 zip을 만듭니다.
3. 측정: `TEAM=<팀 폴더> REPS=3 bash tools/devrun.sh <이름>` (시간), `bash tools/score.sh <이름>` (공식 채점).

가중치는 이 저장소에 없습니다. `submission/weights_delta/README.md`를 보세요. 가중치 파일이 없으면 서버는 이미지의
기본 GravityOCR-KO 가중치로 그대로 돌아갑니다(측정상 약 3% 느림).

## 라이선스

서빙 스택, 커널, 전처리는 1등 팀의 v17 제출본 [sungu-che/minion @ c0f921e](https://github.com/sungu-che/minion/tree/c0f921ef6829f6a957047c30332c48298cb4aa75)
(Apache License 2.0)에서 왔고, 바꾼 내용은 [NOTICE.md](submission/NOTICE.md)에 적었습니다. SGLang overlay 파일의 저작권 표시는
원본 그대로입니다. 이 저장소도 Apache License 2.0을 따릅니다([LICENSE](LICENSE)).

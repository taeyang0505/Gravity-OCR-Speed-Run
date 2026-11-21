# Measured runs (RunPod L40S, dev 402 crops, fresh process per run)

Each run folder keeps the settings (`settings.txt`, extra environment on top of the team folder's own), the wall time of
every fresh-process run (`r*/timing.json`) and, where scored, the kit's scoring output (`scores.json`).
`P2_zip_m3d` is submission 318 (leaderboard 2.282 s), `P3_zip_m4` is submission 319 (leaderboard 1.974 s), both run from the zip as submitted.

| run | walls (s) | median (s) | dev quality | settings |
|---|---|---:|---|---|
| P2_zip_m3d | 2.18 2.19 2.14 | 2.18 | text 97.12 / formula 95.09 / table 99.03 | |
| L2_m3 | 2.16 | 2.16 |  | `TRACK2_LOG=2 TRACK2_LOG_DIR=/root/prof/m3log` |
| S1_tb2 | 2.17 2.18 2.19 | 2.18 |  | `TRACK2_TEXT_BOOST=2` |
| S2_tb3 | 2.15 2.20 2.24 | 2.20 |  | `TRACK2_TEXT_BOOST=3` |
| S3_mr200 | 2.18 2.25 2.22 | 2.22 |  | `MAXREQ=200 GRAPHBS=200` |
| O1_oracle | 2.21 2.15 2.20 | 2.20 |  | `TRACK2_ORDER_ORACLE=/root/prof/m3_rounds.json` |
| O2_oracle_b0 | 2.21 2.18 2.22 | 2.21 |  | `TRACK2_ORDER_ORACLE=/root/prof/m3_rounds.json TRACK2_ORDER=v2f:0.0` |
| K17 | 2.27 2.23 | 2.27 |  | `DRAFT=17` |
| K41 | 2.38 2.39 | 2.39 |  | `DRAFT=41` |
| X5_minus5 | 2.04 2.04 2.02 | 2.04 |  | |
| FG14 | 2.13 2.15 2.16 | 2.15 | text 97.08 / formula 95.25 / table 98.70 | `TRACK2_FORMULA_GLYPH=14` |
| FG16 | 2.15 2.18 2.15 | 2.15 | text 97.09 / formula 95.09 / table 99.07 | `TRACK2_FORMULA_GLYPH=16` |
| D1_dropasync | 2.13 2.19 2.09 | 2.13 | text 97.02 / formula 95.06 / table 98.14 | `SS_VIT_DROP=1` |
| D2_drop_notable | 2.16 2.13 2.15 | 2.15 | text 96.96 / formula 95.08 / table 98.71 | `SS_VIT_DROP=1 SS_VIT_DROP_TABLE=0` |
| FS1_fusedsilu | 2.17 2.13 2.16 | 2.16 | text 97.06 / formula 95.12 / table 98.72 | `SS_FUSED_SILU_VIT=1` |
| GM256 | 2.14 2.12 2.17 | 2.14 | text 96.95 / formula 95.19 / table 98.73 | `TRACK2_GLYPH_MIN_PATCHES=256` |
| C2_all4 | 2.04 2.06 2.10 | 2.06 | text 96.87 / formula 95.13 / table 98.69 | `TRACK2_FORMULA_GLYPH=14 SS_FUSED_SILU_VIT=1 SS_VIT_DROP=1 SS_VIT_DROP_TABLE=0 TRACK2_GLYPH_MIN_PATCHES=256` |
| LN1_33_17 | 2.18 2.23 | 2.23 |  | `TRACK2_LANES=table;text,formula TRACK2_LANE_DRAFT=33,17 TRACK2_LANE_MAXREQ=64,160 TRACK2_LANE_GRAPHBS=64,160 TRACK2_LANE_MEMFRAC=0.36,0.40 TRACK2_LANE_MM_WORKERS=2,4` |
| LN2_41_25 | 2.30 2.29 | 2.30 |  | `TRACK2_LANES=table;text,formula TRACK2_LANE_DRAFT=41,25 TRACK2_LANE_MAXREQ=64,160 TRACK2_LANE_GRAPHBS=64,160 TRACK2_LANE_MEMFRAC=0.36,0.40 TRACK2_LANE_MM_WORKERS=2,4` |
| LN3_33p1_17p0 | 2.12 2.14 | 2.14 |  | `TRACK2_LANES=table;text,formula TRACK2_LANE_DRAFT=33,17 TRACK2_LANE_MAXREQ=64,160 TRACK2_LANE_GRAPHBS=64,160 TRACK2_LANE_MEMFRAC=0.36,0.40 TRACK2_LANE_MM_WORKERS=2,4 TRACK2_LANE_MP` |
| LN4_25p1_13p0 | 2.09 2.11 | 2.11 |  | `TRACK2_LANES=table;text,formula TRACK2_LANE_DRAFT=25,13 TRACK2_LANE_MAXREQ=64,160 TRACK2_LANE_GRAPHBS=64,160 TRACK2_LANE_MEMFRAC=0.36,0.40 TRACK2_LANE_MM_WORKERS=2,4 TRACK2_LANE_MP` |
| LN5_33p1_13p0 | 2.07 2.11 2.08 | 2.08 |  | `TRACK2_LANES=table;text,formula TRACK2_LANE_DRAFT=33,13 TRACK2_LANE_MAXREQ=64,160 TRACK2_LANE_GRAPHBS=64,160 TRACK2_LANE_MEMFRAC=0.36,0.40 TRACK2_LANE_MM_WORKERS=2,4 TRACK2_LANE_MP` |
| LN6_25p1_13p0_C | 1.86 1.91 1.94 | 1.91 | text 96.81 / formula 95.12 / table 98.69 | `TRACK2_LANES=table;text,formula TRACK2_LANE_DRAFT=25,13 TRACK2_LANE_MAXREQ=64,160 TRACK2_LANE_GRAPHBS=64,160 TRACK2_LANE_MEMFRAC=0.36,0.40 TRACK2_LANE_MM_WORKERS=2,4 TRACK2_LANE_MP` |
| LN7_33p1_13p0_C | 1.90 1.91 1.91 | 1.91 | text 96.82 / formula 94.99 / table 99.10 | `TRACK2_LANES=table;text,formula TRACK2_LANE_DRAFT=33,13 TRACK2_LANE_MAXREQ=64,160 TRACK2_LANE_GRAPHBS=64,160 TRACK2_LANE_MEMFRAC=0.36,0.40 TRACK2_LANE_MM_WORKERS=2,4 TRACK2_LANE_MP` |
| P3_zip_m4 | 1.94 1.89 1.92 | 1.92 | text 96.91 / formula 95.19 / table 98.63 | |

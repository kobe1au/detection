# 2026 实验计划：base + generated overrides

所有实验共享 `config/base.yaml`。base 默认就是 full DBTA v2 20% continual 协议；override 只写相对 base 的关键差异。差异配置由 `scripts/make_ablation_configs.py` 生成到 `config/experiments/**.yaml`，入口清单是 `config/experiments/_manifest.yaml`。

当前协议要写清楚：2018-2021 作为 historical train，2022 validation，2023 只作为 recent adaptation pool，2024 final test。不要再把 2023 写成 final test，否则读者会质疑 adaptation/test 泄漏。

## 分组

| 组 | 数量 | 目的 |
|---|---:|---|
| `baselines` | 5 | API、Graph、Concat、Late Fusion、Cross Attention 的 historical ERM 对照 |
| `i1` / `i1_dbta` | 12 | 在 concat 上隔离 random/DBTA selection、static/dynamic_year_class/drift_matched replay、adaptation budget |
| `replay` | 4 | 固定 full model + DBTA 20%，比较 no/static/dynamic_year_class/drift_matched replay |
| `i2` | 5 | 固定 I1，固定 gate，拆解 semantic、class-aware、method-bias、local alignment |
| `i3` | 8 | 固定 I1+I2，拆解 fixed/learned gate、quality、q_time/q_drift、time features、uncertainty、confidence 输入 |
| `ratio` / `full` | 6 | full model 的 0%、5%、10%、20%、50%、100% recent adaptation budget |
| `final` | 1 | 单个最终模型 `M3_full_dbta_v2_020`，避免误跑完整 ratio sweep |
| `main` | 5 | 最短论文主线：concat ERM -> I1 -> I1+I2 -> full -> oracle class-balanced random100 static stress test |

## 主线

| ID | YAML | 解释 |
|---|---|---|
| M0 | `config/experiments/main_chain/M0_concat_erm.yaml` | historical concat ERM |
| M1 | `config/experiments/main_chain/M1_i1_dbta_v2_concat020.yaml` | concat + DBTA v2 20% + selected-adapt-relative drift-matched replay |
| M2 | `config/experiments/main_chain/M2_i1_i2_alignment_fixed_gate020.yaml` | M1 + hierarchical alignment，gate 固定 |
| M3 | `config/experiments/main_chain/M3_full_dbta_v2_020.yaml` | M2 + learned quality/q_time-q_drift/time-feature/uncertainty/confidence gate |
| M4 | `config/experiments/main_chain/M4_full_random_class_balanced100_static.yaml` | full model + oracle class-balanced random 100% + static replay，用来压力测试 DBTA v2 20% 的效率叙事 |

## 运行

先重建 YAML：

```bash
python scripts/make_ablation_configs.py
```

查看分组：

```bash
python run.py --list
python run.py main --dry-run
```

运行单组：

```bash
python run.py baselines
python run.py i1_dbta
python run.py replay
python run.py i2
python run.py i3
python run.py final
python run.py full
```

严苛判断标准：如果 `M3_full_dbta_v2_020` 不能稳定超过 `M4_full_random_class_balanced100_static` 或至少在显著更低 adaptation budget 下接近它，那么“少量 recent adaptation + DBTA”的贡献只能写成效率/成本优势，不能写成绝对性能优势。

写作边界：I1 只验证 budgeted adaptation / replay，因为 I1 配置固定 `fusion_mode=concat`、关闭 alignment、固定 gate；I2/I3 分别验证 drift/reliability signal 在 alignment 与 fusion 中的作用。

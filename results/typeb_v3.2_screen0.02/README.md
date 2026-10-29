# 内存 TypeB · v3.2 第一轮（`screen_margin=0.02`）

这是加入原始日志特征后的第一轮 TypeB 搜索，使用默认初筛门槛 `screen_margin=0.02`。最终结果和原因分析见 [typeb_v3.2](../typeb_v3.2/README.md)。保留这一轮，是因为它是排除"初筛偶然性"这一自检过程的证据。

- 12 组实验后，分析 Agent 主动停止（上限 13 组），没有挑战者被采纳，最终保留基线 trial_001；
- 基线模型和第二轮**完全相同**（配置、数据、随机种子都相同，模型文件哈希一致），所以确认窗口（F1 54.55%）和历史 TEST（F1 0）的结果也和第二轮完全一样；
- **关键记录**：检验"累积 CE 负担 → 故障"假设（H1）的实验 trial_003，因为选择窗口 F1 比基线低 4.2 个百分点，被初筛淘汰，没有进入 CV。选择窗口只有 10 台故障，差 1 台 F1 就相差约 4 个百分点。为了排除这种偶然性，第二轮改用 `screen_margin=1.0` 重跑；结果这个假设在 CV 上同样不成立（P(更好) = 0.091）。

| 路径 | 内容 |
|---|---|
| `session/` | 完整的搜索会话（假设 H1 和 H11 见 `session/hypotheses.json`，trial_003 的淘汰记录见 `session/state.json`） |
| `confirmation/`、`historical_test/` | 确认窗口和历史 TEST 的冻结记录与逐设备结果 |
| `summary.md`、`summary.json` | 比赛指标汇总（`tools/session_report.py` 生成） |
| `report.md`、`report.html`、`search_trials.csv`、`historical_test.csv`、`device_inventory.csv` | 运行时生成的原始报告和结果表 |
| `data_quality/`、`run_scripts/`、`logs/`、`audit/`、`workspace/` | 数据质量检查、实际执行的脚本（含 `raw_typeb.py`）、日志、审计记录 |

`report.md` 和 `audit/` 里的路径对应的是整理前的目录结构，文件内容没有改动。

# v3.1：修复审查中的第 2、3、4 项

本版是独立修复版本。已有 v3 会话、模型及采样文件保留原样；不能把本版程序覆盖到旧会话后重写 code_seal 冒充原运行。

## 已修复

1. **新数据入口**：`init` 保存映射，`prepare` 或 `run` 在会话内创建 TRAIN/VAL 缓存；默认 CV 模式同时在首个实验前确定选择/确认窗口和缓存路径。显式指定的共享缓存若缺失或损坏就报错，不回退到原始数据重建。已有缓存不重复构建。
2. **冻结采样下的设备权重**：保留同一批训练行、标签和采样补偿权重，仅在每台设备内部归一化，让每台保留训练设备的基础权重总和为 1。随后重新计算类别权重。CV 仅对该折的训练设备计算，不读取留出设备来决定权重。原有 sample 模式保留原采样补偿。
3. **训练失败分类**：子进程保存结构化异常记录。普通数据/训练错误标记为 failed 并清除 pending；真正的内存、磁盘、明确 CUDA 资源错误和超时保留任务并停止。包含 memory、cuda、disk 等字样的目录和普通异常文本不再自动触发资源重试。

这次检查还发现：仅补上 normalize 不足以跑通默认入口，因此一起补了原先只在旧会话初始化脚本中设置的 CV 窗口和缓存路径。

## 使用入口

```bash
python agent.py init --data /path/to/data --workspace /path/to/new_workspace --domain memory --mapping /path/to/mapping.json --config /path/to/runtime.json
python agent.py prepare --session /path/to/new_session
python agent.py run --session /path/to/new_session
```

`prepare` 不调用付费模型；如果 init 没有提供 mapping，映射阶段仍可能调用模型。运行前在环境设置所需 API 密钥。缓存构建按映射处理数值观测，不能代替 TypeB 原始错误日志的特征提取和语义确认。

从旧数据缓存新建修复版会话：

```bash
python search_step.py --initialize-from /path/to/old_session --session /path/to/new_session --arm arm_policy.json
```

该入口复用缓存与冻结采样，不导入历史模型，默认重新训练基线。旧版训练失败没有改变旧采样；不要靠删除 sampling_reference 或修改旧行号来绕过检查。已有会话如需保留全部历史再续跑，须另做明确的迁移，不能直接覆盖。

## 验证与范围

```bash
python -m unittest tests.test_review_regressions -v
python -m unittest discover -s tests -v
```

回归覆盖：无手工缓存的 CLI init/run、默认 CV、新会话 prepare、缺失共享缓存拒绝、原始输入改变拒绝、冻结采样下 sample/device 同行对照、每设备权重、真实 CV、采样篡改拒绝、目录名误判、真实子进程异常分类和资源故障暂停。端到端测试使用真实 XGBoost；需要 Agent 回复的部分使用测试替身。

本次没有把设备分组 CV 改成严格的时间外推，也没有补齐复杂度/推理时延/自主比例等评测要求。CV 的训练设备可能包含与留出设备评分期重叠的时间数据，属于同一 TRAIN 时间范围的跨设备验证；不能称为严格向未来外推。重复搜索得到的 CV 分数不是独立测试结论。

新交付的测试日志会脱敏；旧公开仓库的历史提交未在本次重写。完整测试结果和运行环境见交付的验证结果文件。

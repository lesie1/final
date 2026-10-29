# 特征 Agent 的作用与整体流程

框架里有三个 LLM 角色：**分析 Agent**决定下一步做什么，**代码 Agent**把被接受的计划写成训练配置，**特征 Agent**负责一切和"数据、特征"相关的工作。特征 Agent 在流程里出现在四个地方：

| # | 作用 | 什么时候 | 输入 | 输出 | 由谁把关 | 代码 / 记录位置 |
|---|---|---|---|---|---|---|
| ① | **数据适配** | 建立会话时（没有提供 `--mapping` 时） | 数据文件的表头和样例 | 字段映射：哪些列是特征、单位和含义、故障怎么定义、随访怎么算 | 控制器校验格式；故障定义和随访规则要人工核对 | `agent.py adapt` → `mapping.json` |
| ② | **训练前特征建议** | 任何训练开始之前 | 原始数据的字段和训练期样例、已有特征名；**看不到标签和结果** | 需要补充的特征清单：计算方法、因果性、机理、依据及可信度 | **人**逐条决定采纳与否，并负责实现 | `tools/propose_features.py` → `proposals.json`、`adoption.json`、`review.md` |
| ③ | **故障机理假设** | 基线训练完成后、第一次分析之前（每个会话一次） | 只有字段含义；**看不到标签、指标和数值** | 3～12 条可证伪的假设：机理、依据、涉及字段、预期方向、什么情况下算被否定 | 控制器校验字段必须存在；之后的特征请求必须指明检验哪条假设 | `hypotheses.json` |
| ④ | **生成特征程序** | 搜索中，分析 Agent 发出 `features` 请求时 | 现有特征的统计概况、字段含义、假设、父模型的误报/漏报诊断证据、受限的特征 API | 一段 `def transform(frame)` 因果特征代码 | 控制器静态检查、子进程试运行、CV 预筛，最后用 CV 配对比较决定是否采纳 | `feature_jobs/job_XXX/`、`feature_programs/<哈希>/transform.py` |

简单来说：
- ① 和 ② 是**准备数据**："数据怎么读""还缺什么特征"；
- ③ 是**提出理论**："故障可能由什么机理引起"；
- ④ 是**动手做实验**：针对某条假设和当前模型的错误，写出新特征来验证。

例如 TypeA 最终被采纳的 trial_008，就是 ④ 的产物：分析 Agent 发现大量漏报"有分数但低于阈值"，请特征 Agent 针对这个问题写特征；特征 Agent 用原始日志特征组合出了 4 个新特征（最近新出现的出错列占比、新错误模式速率、错误分散度、分散度 × 错误强度），通过了 CV 采纳检验。

## 全流程图

```mermaid
flowchart TD
    subgraph PREP["准备阶段（训练前）"]
        D["官方数据<br/>预处理特征 + 原始日志 + 故障工单"] --> MAP["① 数据适配<br/>特征 Agent 读表头，生成字段映射<br/>（也可以人工提供 --mapping）"]
        MAP --> NORM["规范化与时间划分<br/>agent.py init / prepare"]
        NORM --> PROP["② 训练前特征建议<br/>特征 Agent 根据领域知识<br/>提出需要补充的特征"]
        PROP --> REV{"人工审查<br/>采纳 / 拒绝 / 暂缓"}
        REV -->|采纳| IMPL["人工实现<br/>例：raw_cache.py 解析原始日志"]
        REV -->|拒绝或暂缓| CACHE
        IMPL --> CACHE["模型输入特征缓存"]
    end

    subgraph LOOP["Agent 闭环搜索"]
        CACHE --> BASE["基线训练 + 设备分组 CV"]
        BASE --> HYP["③ 故障机理假设<br/>特征 Agent 只看字段含义<br/>不看标签和结果"]
        HYP --> ANA["分析 Agent<br/>读取实验历史和诊断证据<br/>选择下一步动作"]
        ANA -->|diagnose| DIAG["误报 / 漏报诊断"]
        DIAG --> ANA
        ANA -->|experiment| CODE["代码 Agent<br/>写训练配置"]
        ANA -->|features| FEAT["④ 生成特征程序<br/>特征 Agent 针对某条假设<br/>写 transform(frame)"]
        FEAT --> CHECK{"控制器校验 + CV 预筛"}
        CHECK -->|通过| CODE
        CHECK -->|不通过| ANA
        CODE --> TRAIN["训练 + CV 配对比较"]
        TRAIN --> ACC{"P(更好) ≥ 0.9<br/>且 CV ΔF1 > 0？"}
        ACC -->|是| BEST["更新最优模型"]
        ACC -->|否| KEEP["保留当前最优"]
        BEST --> ANA
        KEEP --> ANA
    end

    ANA -->|stop 或预算用完| FIN["冻结 → 确认窗口评估 → 汇总报告<br/>confirm_search.py / tools/session_report.py"]
```

## ④ 生成特征程序的细节

```mermaid
flowchart TD
    REQ["分析 Agent 的 features 请求<br/>hypothesis_id · question · target（precision / recall / interaction / balanced）"] --> INPUT["交给特征 Agent 的输入<br/>· 现有特征的统计概况<br/>· 字段含义（映射）<br/>· 机理假设<br/>· 父模型的误报 / 漏报诊断证据<br/>· 受限特征 API"]
    INPUT --> WRITE["特征 Agent 写 def transform(frame)"]
    WRITE --> STATIC{"静态检查 + 子进程试运行<br/>只能用白名单操作<br/>只能用预测时刻之前的数据"}
    STATIC -->|失败：附上错误信息重写| WRITE
    STATIC -->|超过修复次数| FAIL["记录失败，交回分析 Agent"]
    STATIC -->|通过| SCREEN{"CV 预筛：新特征在折外样本上<br/>能否区分 真故障 / 误报<br/>或 漏报 / 正常？"}
    SCREEN -->|有信号| QUEUE["作为新实验排队训练"]
    SCREEN -->|没有信号，但给出了探索理由<br/>且探索名额没用完| QUEUE
    SCREEN -->|否则| FAIL
    QUEUE --> CV["训练 + CV 配对比较<br/>按采纳规则决定是否成为最优"]
```

**特征 Agent 不能做的事**（这些限制是为了安全和防止信息泄漏）：
- 读取故障标签、故障时间，或预测时刻之后的数据；
- 在搜索中读取任意文件、调用任意函数（`transform` 只能使用白名单里的 pandas/numpy 操作）；
- 自己解析原始日志：目前由人根据 ② 的建议来实现，由 Agent 自动编写解析器是下一步工作；
- 决定是否采纳：采纳只由预先确定的 CV 统计规则决定。

相关代码：`agent.py` 中的 `adapt()`、`hypotheses()`、`features()`；特征 API 定义在 `feature_contract.py`；检查和执行在 `feature_programs.py`、`feature_worker.py`；预筛在 `contrast.py`；提示词在 `prompts/feature.txt` 和 `tools/prompts/feature_proposal.txt`。

# 第五章图表建议

## 图 5-1 四分支自适应融合总体框架

文件：`docs/figures/ch5_four_branch_fusion_framework.svg`

英文题名：Figure 5-1 Overall Framework for Four-branch Adaptive Fusion

建议放置位置：5.1 “问题描述与总体框架”中，在介绍四个预测分支和门控融合流程之后。

## 图 5-2 门控证据与权重生成示意图

文件：`docs/figures/ch5_gate_evidence.svg`

英文题名：Figure 5-2 Illustration of Gate Evidence and Weight Generation

建议放置位置：5.3 “融合证据构建”末尾，或 5.4 “冲突感知门控融合”开头。

## 表 5-1 四分支结构与功能说明

英文题名：Table 5-1 Structure and Function of the Four Prediction Branches

建议放置位置：5.2 “四分支表示学习与分支预测”中，在介绍四个分支之后。

| 分支名称 | 输入信息 | 编码方式 | 输出结果 | 主要作用 |
|---|---|---|---|---|
| API 分支 | API 调用序列、API 类型、敏感 API 标记 | 序列编码器 | API 分支预测结果 | 学习代码中的显式行为模式，保留 API 模态的独立判别能力 |
| 调用图分支 | 方法节点特征、调用边、敏感节点标记 | 图神经网络 | 调用图分支预测结果 | 学习方法调用关系和结构化上下文信息 |
| Manifest 分支 | 权限、组件、意图过滤器、系统版本与统计特征 | 多层前馈网络 | Manifest 分支预测结果 | 学习应用声明能力和配置侧辅助证据 |
| 三模态综合分支 | API、调用图和 Manifest 的隐层表示 | 表示拼接与联合编码 | 综合分支预测结果 | 建模三类模态之间的互补信息，在多模态证据可靠时增强判别能力 |

## 表 5-2 融合证据组成及作用

英文题名：Table 5-2 Components and Roles of Fusion Evidence

建议放置位置：5.3 “融合证据构建”中，在列出融合证据向量前。

| 证据类型 | 主要字段 | 含义 | 在门控融合中的作用 |
|---|---|---|---|
| 模态质量 | q_api、q_graph、q_manifest | 描述各模态原始信息的完整性和可用程度 | 为分支权重提供基础质量判断 |
| 退化强度 | pert_api、pert_graph、pert_manifest | 描述各模态受到扰动、缺失或噪声影响的程度 | 在模态受损时降低对应分支的参考可信度 |
| 模态可靠性 | r_api、r_graph、r_manifest | 由质量和退化强度共同计算得到的样本级可靠性 | 作为门控网络判断分支可信程度的核心证据 |
| 代码侧对应质量 | q_align | 描述 API 调用与调用图方法节点之间的可观测对应程度 | 辅助判断 API 与调用图代码侧证据是否完整 |
| 跨源一致性 | c_api_manifest、c_graph_manifest、c_api_graph | 描述不同模态在统一安全语义类别空间中的类别相似程度 | 辅助识别 Manifest 与代码证据是否相互支持 |
| 分支置信度 | conf_api、conf_graph、conf_manifest、conf_joint | 描述各分支预测结果的置信程度 | 帮助门控网络结合分支自身判别状态分配权重 |
| 模态可用状态 | a_api、a_graph、a_manifest | 描述各模态是否存在有效输入 | 在模态缺失时抑制对应分支权重 |

## 表 5-3 训练目标组成

英文题名：Table 5-3 Components of the Training Objective

建议放置位置：5.5 “模型训练目标”末尾，在总损失公式之后。

| 损失项 | 作用对象 | 主要作用 | 章节来源 |
|---|---|---|---|
| 最终分类损失 | 四分支加权后的最终预测结果 | 优化模型总体检测性能 | 第 5 章 |
| 分支辅助损失 | API、调用图、Manifest 和综合分支 | 增强各分支独立判别能力，避免模型只依赖单一强分支 | 第 5 章 |
| 跨源软一致性损失 | API、调用图和 Manifest 的语义类别表示 | 引导可靠模态之间在功能类别层面保持适度一致 | 第 4 章 |
| 门控先验约束 | 门控网络输出的四分支权重 | 以较小权重引导门控网络关注可靠性和一致性证据 | 第 5 章 |


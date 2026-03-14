# BERT多任务训练Loss优化指南

## 核心结论

1. **BERT文本分类与NER的关系**：两者共享相同的编码器，但输出层不同——分类用[CLS]向量的全局表示，NER用所有token的序列表示。
2. **Loss选择**：分类用**交叉熵损失（Cross-Entropy Loss）**，NER用**序列标注的交叉熵损失**（本质也是CE，但应用于每个token）。
3. **简单相加Loss的致命问题**：
   - **梯度冲突**：不同任务的梯度方向可能相反，导致参数更新无效或震荡。
   - **任务主导**：数据量大或损失值大的任务会主导训练，其他任务被忽略。
   - **负迁移**：不相关任务会互相干扰，降低整体性能。
4. **训练不平衡的处理策略**：从**静态权重调整**→**动态权重调整**→**梯度手术**→**元学习**，层层递进。

---

## 一、BERT文本分类与NER的关系

### 架构层面的关系

| 维度 | 文本分类 | NER（命名实体识别） |
|------|----------|---------------------|
| **任务类型** | 序列级分类 | Token级分类 |
| **输出层输入** | `[CLS]` token的隐藏向量 | 所有token的隐藏向量 |
| **输出层结构** | 全连接层 → softmax | 全连接层 → CRF（可选） |
| **标签空间** | `{0, 1, ..., K-1}` | `{O, B-PER, I-PER, B-LOC, ...}` |
| **损失函数** | 交叉熵损失 | Token级交叉熵损失 |

**关键点**：
- 两者共享**相同的BERT编码器**，都从BERT的最后一层隐藏状态提取特征。
- 分类只关注全局语义（`[CLS]`），NER关注细粒度的每个token的语义。

### Loss的具体形式

**文本分类的Loss**：
```
L_seq = -∑_i y_i * log(p_i)  # 标准交叉熵
```

**NER的Loss**：
```
L_token = -∑_{t=1}^T ∑_i y_{t,i} * log(p_{t,i})  # 每个token的交叉熵求和
```

---

## 二、简单相加Loss的坏处

### 1. 梯度冲突（Gradient Interference）

当两个任务的梯度方向夹角接近90°甚至更大时，参数更新会互相抵消：

```
∇L_total = w_seq * ∇L_seq + w_token * ∇L_token
```

如果 `∇L_seq · ∇L_token < 0`，那么一个任务的进步会抵消另一个任务的进步。

### 2. 任务主导（Task Domination）

假设：
- 分类任务有100,000条样本
- NER任务只有10,000条样本
- 两者都使用标准交叉熵

即使你设置 `w_seq = w_token = 0.5`，分类任务的梯度仍然会主导训练，因为：
- **样本数量差异**导致反向传播的累积梯度大小不同
- **Loss值scale不同**（分类Loss通常比NER Loss大10-100倍）

### 3. 负迁移（Negative Transfer）

如果两个任务不相关（如情感分析 + NER），共享的编码器会同时学习两个矛盾的特征空间，导致：
- 某些任务性能下降
- 收敛速度变慢

---

## 三、训练不平衡的解决方案（从简单到高级）

### Level 1：静态权重调整（最简单，但不推荐）

```python
L_total = alpha * L_seq + (1-alpha) * L_token
```

**问题**：需要手动调参，且无法适应训练过程中的动态变化。

### Level 2：动态权重调整（基于任务难度）

**Uncertainty Weighting（Kendall et al., 2018）**：
```python
# 将任务权重建模为可学习的参数
L_total = 1/(2*σ1^2) * L_seq + 1/(2*σ2^2) * L_token + log(σ1) + log(σ2)
```

**原理**：自动学习每个任务的不确定性，不确定性高的任务权重低。

**GradNorm（Chen et al., 2018）**：
```python
# 让所有任务的梯度范数保持在同一水平
调整权重w_i，使得 ||∇L_i|| ≈ target_norm
```

**DWA（Dynamic Weight Averaging）**：
```python
# 根据任务损失的变化速度调整权重
w_i(t) = exp(r_i(t-1) / T) / ∑_j exp(r_j(t-1) / T)
# r_i(t) 是任务i的相对学习速度
```

### Level 3：梯度手术（Gradient Surgery）解决冲突

**PCGrad（Projecting Conflicting Gradients）**：
```python
if ∇L_seq · ∇L_token < 0:  # 梯度冲突
    ∇L_token = ∇L_token - (∇L_seq · ∇L_token / ||∇L_seq||^2) * ∇L_seq
∇L_total = ∇L_seq + ∇L_token
```

**原理**：当梯度冲突时，将一个任务的梯度投影到另一个任务梯度的法平面上，消除冲突分量。

### Level 4：元学习（Meta-Learning）自动调权

**MetaWeighting**：
```python
# 双层优化：
# 内层：更新模型参数θ
# 外层：更新任务权重w，使验证集损失最小化
min_w E_{i~tasks}[L_val(θ*(w))]
```

**UW-SO（Soft Optimal Uncertainty Weighting）**：
```python
# 对Uncertainty Weighting的理论改进
w_i = softmax(1/L_i / temperature)
```

### Level 5：样本级权重（最精细）

**SLGrad（Sample-Level Weighting）**：
```python
# 不只调整任务权重，还调整每个样本的权重
w_{sample} ∝ ∇L_sample · ∇M_val  # 样本梯度与验证指标梯度的对齐度
```

---

## 四、实战建议

如果你现在要解决这个具体问题（seq_loss + token_loss不平衡），我建议：

1. **先尝试最简单的方案**：
   ```python
   L_total = alpha * L_seq + beta * L_token
   ```
   - `alpha` 和 `beta` 通过网格搜索或贝叶斯优化找最优值
   - 初始建议：`alpha = 0.7, beta = 0.3`（分类通常更重要）

2. **如果效果不佳，用GradNorm**：
   - 自动平衡各任务的学习速度
   - PyTorch有现成实现

3. **如果任务冲突严重，用PCGrad**：
   - 特别适用于梯度方向相反的情况
   - GitHub上有开源实现

4. **如果追求极致性能，用元学习方法**：
   - 需要额外的验证集来指导权重更新
   - 计算成本较高

---

## 五、代码示例

```python
import torch
import torch.nn as nn

class MultiTaskBERT(nn.Module):
    def __init__(self, bert_model, num_classes, num_tags):
        super().__init__()
        self.bert = bert_model
        self.seq_classifier = nn.Linear(768, num_classes)
        self.token_classifier = nn.Linear(768, num_tags)
        
        # 可学习的任务权重
        self.log_var_seq = nn.Parameter(torch.zeros(1))
        self.log_var_token = nn.Parameter(torch.zeros(1))
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS]
        sequence_output = outputs.last_hidden_state  # all tokens
        
        seq_logits = self.seq_classifier(pooled_output)
        token_logits = self.token_classifier(sequence_output)
        
        return seq_logits, token_logits
    
    def compute_loss(self, seq_logits, seq_labels, token_logits, token_labels):
        # 序列分类损失
        seq_loss = nn.CrossEntropyLoss()(seq_logits, seq_labels)
        
        # Token级损失（忽略padding tokens）
        token_loss = nn.CrossEntropyLoss(reduction='none')(token_logits.view(-1, token_logits.size(-1)), token_labels.view(-1))
        token_loss = token_loss[token_labels.view(-1) != -100].mean()
        
        # Uncertainty Weighting
        precision_seq = torch.exp(-self.log_var_seq)
        precision_token = torch.exp(-self.log_var_token)
        
        total_loss = precision_seq * seq_loss + precision_token * token_loss + self.log_var_seq + self.log_var_token
        return total_loss
```

---

## 六、参考文献

1. Kendall, A., et al. (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR.
2. Chen, Z., et al. (2018). "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." ICML.
3. Yu, T., et al. (2020). "Gradient Surgery for Multi-Task Learning." NeurIPS.
4. Liu, X., et al. (2019). "Dynamic Task Prioritization for Multitask Learning." ECCV.
5. Liu, Y., et al. (2022). "MetaWeighting: Learning to Weight Tasks in Multi-Task Learning." ACL Findings.

---

## 附录：快速对比表

| 方法 | 复杂度 | 适用场景 | 优点 | 缺点 |
|------|--------|----------|------|------|
| 静态权重 | ⭐ | 任务重要性明确 | 简单直接 | 需要调参，无法适应动态变化 |
| Uncertainty Weighting | ⭐⭐ | 任务难度差异大 | 自动学习权重 | 假设过强，可能过拟合 |
| GradNorm | ⭐⭐ | 任务学习速度不均 | 平衡学习速度 | 需要调target norm |
| PCGrad | ⭐⭐⭐ | 梯度冲突严重 | 直接解决冲突 | 计算成本增加 |
| MetaWeighting | ⭐⭐⭐⭐ | 追求极致性能 | 优化泛化性能 | 计算成本高，需要验证集 |
| SLGrad | ⭐⭐⭐⭐⭐ | 样本噪声大 | 最精细的粒度 | 计算成本极高 |

:::tips
1️⃣【本周第七周作业：】

作业1: 阅读 02-joint-bert-training-only 代码，并回答以下问题：

    ◦ bert 文本分类和 实体识别有什么关系，分别使用什么loss？

    ◦ 多任务训练  loss = seq_loss + token_loss 有什么坏处，如果存在训练不平衡的情况，如何处理？

作业2: 基础 02-joint-bert-training-only  中的数据集，希望你自己写一个提示词能完成任务（信息解析的智能对话系统）





（作业按时完成，在周五之前进行提交，加油👏有问题及时沟通，老师永远是你最坚实的后盾。[拥抱][拥抱]）

:::

## 一、BERT 做文本分类（意图识别）
这个是之前学习到的做意图识别，比如播放音乐，我想导航到天安门等等

1. 将文本输入 BERT；

2.取[CLS]为节点



## 二、BERT 做命名实体识别（槽位填充）
本节课学习的实体识别



<font style="color:rgb(34, 35, 37);">我 想 听 周 杰 伦 的 歌 </font>

<font style="color:rgb(34, 35, 37);">O O O B-artist I-artist I-artist O O</font>



1. 输入文本到 BERT；
2. 获取**每个 token 的隐藏状态**（包括 `[CLS]`, `[SEP]`）；





## 使用什么LOSS(千问回答)
在 BERT 联合模型（Joint BERT）中，**文本分类（意图识别）** 和 **命名实体识别（槽位填充）** 使用的损失函数虽然底层都是 **交叉熵（Cross-Entropy Loss）**，但**输入形式和计算方式不同**。下面是详细对比：

---

## ✅ 1. 文本分类（意图识别） → **标准 Cross-Entropy**
### 🎯 任务特点
+ 每个样本只有一个标签（如 `"play_music"`）
+ 输出是一个类别概率分布

### 🔧 损失函数
```python
import torch.nn as nn

criterion_cls = nn.CrossEntropyLoss()
# 输入: logits (batch_size, num_classes)
# 标签: labels (batch_size,) → 长整型（如 [2, 0, 1]）

loss_intent = criterion_cls(intent_logits, intent_labels)
```

### 📌 示例
```python
intent_logits.shape    # torch.Size([8, 10])  ← 8个样本，10个意图类别
intent_labels.shape    # torch.Size([8])      ← 每个样本一个真实类别ID
```

✅ 这是**多分类单标签**的标准用法。

---

## ✅ 2. 命名实体识别（槽位填充） → **Token-level Cross-Entropy（带 Mask）**
### 🎯 任务特点
+ 每个 token 都有一个标签（如 `['O', 'B-PER', 'I-PER', ...]`）
+ 需要忽略 `[PAD]`、`[CLS]`、`[SEP]` 等位置的 loss

### 🔧 损失函数（关键：**mask padding 位置**）
```python
# 方法 1：使用 ignore_index（推荐）
criterion_ner = nn.CrossEntropyLoss(ignore_index=-100)

# 在构造 token_label_ids 时，把 [PAD] 位置设为 -100
# 例如：原始标签 [0, 2, 3, 0, 0] → 加 [CLS]/[SEP] 后 → [0, 2, 3, 0, 0, 0]
# padding 到长度 10 → [0, 2, 3, 0, 0, 0, -100, -100, -100, -100]

loss_slot = criterion_ner(slot_logits.view(-1, num_labels), slot_labels.view(-1))
```

### 或者手动 mask（更灵活）：
```python
# 方法 2：手动 mask
active_loss = attention_mask.view(-1) == 1          # (batch * seq_len,)
active_logits = slot_logits.view(-1, num_labels)[active_loss]   # 只保留有效 token
active_labels = slot_labels.view(-1)[active_loss]

loss_slot = criterion_ner(active_logits, active_labels)
```

⚠️ 注意：`slot_logits` 是 `(batch, seq_len, num_labels)`，必须 `view(-1, num_labels)` 扁平化。

### 📌 示例
```python
slot_logits.shape     # torch.Size([8, 64, 50])  ← 8样本，64长度，50个NER标签
slot_labels.shape     # torch.Size([8, 64])       ← 每个token一个标签ID

# 经过 view(-1, ...) 后：
slot_logits_flat      # (512, 50)
slot_labels_flat      # (512,)  ← 其中部分为 -100（被 ignore）
```

---

## ✅ 3. 联合总损失（Joint Loss）
通常简单相加（也可加权重）：

```python
total_loss = loss_intent + loss_slot
# 或
total_loss = α * loss_intent + β * loss_slot  # α=1.0, β=1.0 常见
```

---

## 📊 对比总结
| 任务 | 损失函数 | 输入形状 | 标签形状 | 特殊处理 |
| --- | --- | --- | --- | --- |
| **文本分类** | `CrossEntropyLoss()` | `(N, C)` | `(N,)` | 无 |
| **NER** | `CrossEntropyLoss(ignore_index=-100)` | `(N×L, C)` | `(N×L,)` | **mask padding / special tokens** |


+ `N` = batch size 
+ `L` = sequence length 
+ `C` = number of classes

---

## 💡 实践建议
1. **构造 NER 标签时，padding 位置设为 **`**-100**`（PyTorch 默认 ignore_index）：

```python
# 假设 max_len=64，实际文本长度=5
token_label_ids = [0] + original_labels + [0]  # 加 [CLS]/[SEP]
token_label_ids += [-100] * (64 - len(token_label_ids))  # padding 用 -100
```

2. **验证前几个样本的标签是否对齐**（打印 input_ids 和 labels）：

```python
print("Text:", tokenizer.convert_ids_to_tokens(input_ids))
print("Labels:", [id2label[i] if i != -100 else "PAD" for i in token_label_ids])
```

3. **避免使用 **`**eval()**`** 读数据，改用 **`**json.load()**`** + **`**encoding='utf-8'**`

---

这样，你的 Joint BERT 模型就能正确计算两种任务的损失，并联合优化！如果还有具体代码问题，欢迎贴出 loss 计算部分。



### ❌ 1. **量级不平衡（Scale Imbalance）**
+ `seq_loss` 是 **单样本一个标量**（如 batch=32 → 32 个 loss 值平均后 ≈ 1~3）
+ `token_loss` 是 **每个 token 一个 loss**（如 batch=32, len=50 → 1600 个 token，平均后可能 ≈ 0.5~2）

### ❌ 2. **任务难度差异未被考虑**
+ 某些数据集中，意图分类很简单（如只有 5 类），而 NER 很难（实体边界模糊）；
+ 或反过来：意图类别多且相似，NER 却简单。

👉 简单相加无法动态调整学习重点。







# 作业二
你是一个智能对话系统的信息解析模块。请根据以下规则，从用户输入中提取结构化信息：

+ 识别用户的**意图（intent）**，可能的意图包括：play_music, check_weather, search_poi
+ 同时提取相关的**槽位（slots）**，槽位类型包括：
    - check_weather: date（日期）, city（城市）
    - play_music: artist（歌手）, song（歌曲）
    - search_poi: location（地点）

请以严格的 JSON 格式输出，不要包含任何解释、markdown 或额外文字。

示例：  
输入：我想听周杰伦的稻香  
输出：{"intent": "play_music", "slots": {"artist": "周杰伦", "song": "稻香"}}

输入：后天上海天气如何？  
输出：{"intent": "check_weather", "slots": {"date": "后天", "city": "上海"}}

现在请解析以下输入：  
<font style="color:#1DC0C9;">输入：我今天去北海公园转了一圈，明天我们去颐和园吧。  
</font><font style="color:#1DC0C9;">输出：{"intent": "search_poi", "slots": {"location": "北海公园", "location": "颐和园"}}</font>

<font style="color:#1DC0C9;">{"intent": "check_weather", "slots": {"date": "今天", "date": "明天"}}</font>


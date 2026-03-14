## bert 文本分类和 实体识别有什么关系，分别使用什么loss？
文本分类是对整个句子做分类，实体识别是对句子中的每个token做分类
文本分类主要用到句向量（在bert1中是[CLS]对应的向量），实体识别主要用到token向量
两者都可以用交叉熵loss

## 多任务训练  loss = seq_loss + token_loss 有什么坏处，如果存在训练不平衡的情况，如何处理？
seq_loss和token_loss不是一个量级的，所以直接相加会导致训练不平衡
可以给token_loss和seq_loss各乘一个系数，比如token_loss的系数设为1，seq_loss的系数设为0.5
"""
    BERT文本分类使用的损失函数主要是交叉熵损失函数(nn.CrossEntropyLoss())，参见代码main.py的line48。
    seq_loss = self.criterion(seq_output, seq_label_ids)
    BERT实体识别的损失函数也是交叉熵损失函数(nn.CrossEntropyLoss())，但仅对有效token进行计算，而不是所有位置都计算，参见代码main.py的line49。
    active_loss = attention_mask.view(-1) == 1
    active_logits = token_output.view(-1, token_output.shape[2])[active_loss]
    active_labels = token_label_ids.view(-1)[active_loss]
    token_loss = self.criterion(active_logits, active_labels)
  
    多任务训练 loss = seq_loss + token_loss的坏处是任务冲突，主要在于两个训练任务的收敛过程可能不一样，有的可能是快速收敛，loss快速下降，有的可能是缓慢收敛，需要训练的轮次更多，叠加在一起训练可能导致训练结果比较差。
    主要的解决办法是：超参数调整，调整不同损失函数的权重、设置不同的学习率等。
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# 读取数据
dataset = pd.read_csv ("dataset.csv" , sep="\t" , header=None)
texts = dataset[0].tolist ()
string_labels = dataset[1].tolist ()

# 创建标签映射
label_to_index = {label: i for i , label in enumerate (set (string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label , i in label_to_index.items ()}

# 创建字符映射
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len (char_to_index)

index_to_char = {i: char for char , i in char_to_index.items ()}
vocab_size = len (char_to_index)

# 最大文本长度
max_len = 40


# 自定义数据集
class CharRNNDataset (Dataset):
    def __init__(self , texts , labels , char_to_index , max_len):
        self.texts = texts
        self.labels = torch.tensor (labels , dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len (self.texts)

    def __getitem__(self , idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get (char , 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len (indices))
        return torch.tensor (indices , dtype=torch.long) , self.labels[idx]


# 分割训练集和测试集
train_texts , test_texts , train_labels , test_labels = train_test_split (
    texts , numerical_labels , test_size=0.2 , random_state=42 , stratify=numerical_labels
)

# 创建训练和测试数据集
train_dataset = CharRNNDataset (train_texts , train_labels , char_to_index , max_len)
test_dataset = CharRNNDataset (test_texts , test_labels , char_to_index , max_len)


# LSTM模型
class LSTMClassifier (nn.Module):
    def __init__(self , vocab_size , embedding_dim , hidden_dim , output_dim , num_layers=1 , dropout=0.0):
        super (LSTMClassifier , self).__init__ ()
        self.embedding = nn.Embedding (vocab_size , embedding_dim)
        self.lstm = nn.LSTM (embedding_dim , hidden_dim , num_layers=num_layers ,
                             batch_first=True , dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear (hidden_dim , output_dim)

    def forward(self , x):
        embedded = self.embedding (x)
        lstm_out , (hidden_state , cell_state) = self.lstm (embedded)
        out = self.fc (hidden_state[-1])  # 取最后一层的hidden state
        return out

#双向LSTM模型
class BiLSTMClassifier (nn.Module):
    def __init__(self , vocab_size , embedding_dim , hidden_dim , output_dim , num_layers=1 , dropout=0.0):
        super (BiLSTMClassifier , self).__init__ ()

        # 保持与原始LSTM相同的嵌入层
        self.embedding = nn.Embedding (vocab_size , embedding_dim)

        # 将单向LSTM改为双向LSTM
        # hidden_dim保持相同，但实际每个方向使用hidden_dim/2，以保持总参数数量相对一致
        # 或者保持hidden_dim不变，这样总隐藏维度会翻倍
        self.lstm = nn.LSTM (
            embedding_dim ,
            hidden_dim ,  # 保持相同的隐藏维度
            num_layers=num_layers ,
            batch_first=True ,
            dropout=dropout if num_layers > 1 else 0.0 ,
            bidirectional=True  # 关键修改：变为双向
        )

        # 对于双向LSTM，最后一个隐藏状态有两个（前向和后向）
        # 需要将两者连接起来，所以输入维度变为 hidden_dim * 2
        self.fc = nn.Linear (hidden_dim * 2 , output_dim)

    def forward(self , x):
        embedded = self.embedding (x)

        # LSTM输出
        lstm_out , (hidden_state , cell_state) = self.lstm (embedded)

        # 对于双向LSTM，hidden_state的形状是 (num_layers * 2, batch_size, hidden_dim)
        # 我们需要连接最后两个隐藏状态（前向和后向）
        # 取最后两层（双向所以是最后两层）
        forward_hidden = hidden_state[-2 , : , :]  # 前向最后一个隐藏状态
        backward_hidden = hidden_state[-1 , : , :]  # 后向最后一个隐藏状态

        # 连接前向和后向的隐藏状态
        concatenated = torch.cat ((forward_hidden , backward_hidden) , dim=1)

        # 全连接层分类
        out = self.fc (concatenated)

        return out

# RNN模型
class RNNClassifier (nn.Module):
    def __init__(self , vocab_size , embedding_dim , hidden_dim , output_dim , num_layers=2 , dropout=0.3):
        super (RNNClassifier , self).__init__ ()

        self.embedding = nn.Embedding (vocab_size , embedding_dim , padding_idx=0)

        self.rnn = nn.RNN (
            embedding_dim ,
            hidden_dim ,
            num_layers=num_layers ,
            batch_first=True ,
            dropout=dropout if num_layers > 1 else 0.0 ,
            nonlinearity='tanh'  # 明确指定tanh激活
        )

        self.batch_norm = nn.BatchNorm1d (hidden_dim)
        self.dropout = nn.Dropout (dropout)
        self.fc1 = nn.Linear (hidden_dim , hidden_dim // 2)
        self.fc2 = nn.Linear (hidden_dim // 2 , output_dim)
        self.relu = nn.ReLU ()
        self._init_weights ()

    def _init_weights(self):
        """改进权重初始化方法"""
        # 初始化嵌入层权重
        nn.init.xavier_uniform_ (self.embedding.weight)

        for name , param in self.rnn.named_parameters ():
            if 'weight' in name:
                # 使用正交初始化，有助于缓解梯度消失/爆炸
                nn.init.orthogonal_ (param)
            elif 'bias' in name:
                # 偏置初始化为0
                nn.init.constant_ (param , 0)

    def forward(self , x):
        embedded = self.embedding (x)

        rnn_out , hidden_state = self.rnn (embedded)

        last_hidden = hidden_state[-1]
        normalized = self.batch_norm (last_hidden)
        dropped = self.dropout (normalized)

        fc1_out = self.fc1 (dropped)
        fc1_activated = self.relu (fc1_out)

        fc1_dropped = self.dropout (fc1_activated)
        out = self.fc2 (fc1_dropped)

        return out


# GRU模型
class GRUClassifier (nn.Module):
    def __init__(self , vocab_size , embedding_dim , hidden_dim , output_dim , num_layers=1 , dropout=0.0):
        super (GRUClassifier , self).__init__ ()
        self.embedding = nn.Embedding (vocab_size , embedding_dim)
        self.gru = nn.GRU (embedding_dim , hidden_dim , num_layers=num_layers ,
                           batch_first=True , dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear (hidden_dim , output_dim)

    def forward(self , x):
        embedded = self.embedding (x)
        gru_out , hidden_state = self.gru (embedded)
        out = self.fc (hidden_state[-1])  # 取最后一层的hidden state
        return out


# 训练函数
def train_model(model , train_loader , test_loader , model_name , num_epochs=10 , lr=0.001):
    criterion = nn.CrossEntropyLoss ()
    optimizer = optim.Adam (model.parameters () , lr=lr)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print (f"\n{'=' * 50}")
    print (f"开始训练 {model_name}")
    print (f"{'=' * 50}")

    for epoch in range (num_epochs):
        model.train ()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx , (inputs , labels) in enumerate (train_loader):
            optimizer.zero_grad ()
            outputs = model (inputs)
            loss = criterion (outputs , labels)
            loss.backward ()
            optimizer.step ()

            running_loss += loss.item ()
            _ , predicted = torch.max (outputs , 1)
            total += labels.size (0)
            correct += (predicted == labels).sum ().item ()

            if batch_idx % 50 == 0:
                print (f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len (train_loader)}], "
                       f"Loss: {loss.item ():.4f}")

        epoch_loss = running_loss / len (train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append (epoch_loss)
        train_accuracies.append (train_accuracy)

        # 测试集评估
        test_accuracy = evaluate_model (model , test_loader)
        test_accuracies.append (test_accuracy)

        print (f"Epoch [{epoch + 1}/{num_epochs}] - "
               f"Loss: {epoch_loss:.4f}, "
               f"Train Acc: {train_accuracy:.2f}%, "
               f"Test Acc: {test_accuracy:.2f}%")

    return train_losses , train_accuracies , test_accuracies


# 评估函数
def evaluate_model(model , test_loader):
    model.eval ()
    correct = 0
    total = 0

    with torch.no_grad ():
        for inputs , labels in test_loader:
            outputs = model (inputs)
            _ , predicted = torch.max (outputs , 1)
            total += labels.size (0)
            correct += (predicted == labels).sum ().item ()

    return 100 * correct / total


# 预测函数
def classify_text(text , model , char_to_index , max_len , index_to_label):
    indices = [char_to_index.get (char , 0) for char in text[:max_len]]
    indices += [0] * (max_len - len (indices))
    input_tensor = torch.tensor (indices , dtype=torch.long).unsqueeze (0)

    model.eval ()
    with torch.no_grad ():
        output = model (input_tensor)

    _ , predicted_index = torch.max (output , 1)
    predicted_index = predicted_index.item ()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 主函数
def main():
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader (train_dataset , batch_size=batch_size , shuffle=True)
    test_loader = DataLoader (test_dataset , batch_size=batch_size , shuffle=False)

    # 模型参数
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len (label_to_index)
    num_layers = 2
    dropout = 0.2
    num_epochs = 10

    # 初始化三个模型
    models = {
        "LSTM": LSTMClassifier (vocab_size , embedding_dim , hidden_dim , output_dim , num_layers , dropout) ,
        "BiLSTM": BiLSTMClassifier (vocab_size , embedding_dim , hidden_dim , output_dim , num_layers , dropout) ,
        "RNN": RNNClassifier (vocab_size , embedding_dim , hidden_dim , output_dim , num_layers , dropout) ,
        "GRU": GRUClassifier (vocab_size , embedding_dim , hidden_dim , output_dim , num_layers , dropout)
    }

    results = {}

    # 训练并评估每个模型
    for model_name , model in models.items ():
        print (f"\n{'=' * 60}")
        print (f"训练 {model_name} 模型")
        print (f"{'=' * 60}")

        train_losses , train_accuracies , test_accuracies = train_model (
            model , train_loader , test_loader , model_name , num_epochs
        )

        results[model_name] = {
            'train_losses': train_losses ,
            'train_accuracies': train_accuracies ,
            'test_accuracies': test_accuracies ,
            'final_test_accuracy': test_accuracies[-1] ,
            'model': model
        }

    # 打印结果对比
    print (f"\n{'=' * 60}")
    print ("模型性能对比")
    print (f"{'=' * 60}")

    for model_name , result in results.items ():
        print (f"{model_name}:")
        print (f"  训练准确率: {result['train_accuracies'][-1]:.2f}%")
        print (f"  测试准确率: {result['final_test_accuracy']:.2f}%")
        print (f"  最终训练损失: {result['train_losses'][-1]:.4f}")
        print ()

    # 找出最佳模型
    best_model_name = max (results.keys () , key=lambda x: results[x]['final_test_accuracy'])
    best_model = results[best_model_name]['model']
    print (f"最佳模型: {best_model_name} (测试准确率: {results[best_model_name]['final_test_accuracy']:.2f}%)")

    # 用最佳模型进行预测
    print (f"\n{'=' * 60}")
    print ("使用最佳模型进行预测")
    print (f"{'=' * 60}")

    test_samples = [
        "帮我导航到北京" ,
        "查询明天北京的天气" ,
        "播放周杰伦的歌" ,
        "打开计算器" ,
        "今天股市怎么样"
    ]

    for text in test_samples:
        predicted_class = classify_text (text , best_model , char_to_index , max_len , index_to_label)
        print (f"输入 '{text}' 预测为: '{predicted_class}'")

        # 保存最佳模型
        torch.save (best_model.state_dict () , f"best_{best_model_name.lower ()}_model.pth")
        print (f"\n最佳模型已保存为: best_{best_model_name.lower ()}_model.pth")

        return results

if __name__ == "__main__":
    results = main ()

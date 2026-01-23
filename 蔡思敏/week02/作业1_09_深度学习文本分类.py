"""
    改动点：
    1. 修改隐藏层数量：1-》3
    2. 修改隐藏层节点数量：64,128,32
    3. 修改epoch：10 -》100 （增加网络复杂度后，10个epoch的loss仍然很大，未充分训练，加大epoch）

    修改后的运行结果 附在该文件最后
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


# 自定义数据集 custom dataset
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        # pad 和 crop
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# 调整层数
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

hidden_dim1 = 64
hidden_dim2 = 128
hidden_dim3 = 32

output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

# Batch 个数 0, 当前Batch Loss: 2.4379465579986572
# Batch 个数 50, 当前Batch Loss: 2.4675259590148926
# Batch 个数 100, 当前Batch Loss: 2.42108416557312
# Batch 个数 150, 当前Batch Loss: 2.450716018676758
# Batch 个数 200, 当前Batch Loss: 2.415167808532715
# Batch 个数 250, 当前Batch Loss: 2.4192421436309814
# Batch 个数 300, 当前Batch Loss: 2.447007417678833
# Batch 个数 350, 当前Batch Loss: 2.4188387393951416
# Epoch [1/100], Loss: 2.4430
# Batch 个数 0, 当前Batch Loss: 2.431652784347534
# Batch 个数 50, 当前Batch Loss: 2.413646697998047
# Batch 个数 100, 当前Batch Loss: 2.3926782608032227
# Batch 个数 150, 当前Batch Loss: 2.4331061840057373
# Batch 个数 200, 当前Batch Loss: 2.416121244430542
# Batch 个数 250, 当前Batch Loss: 2.4159605503082275
# Batch 个数 300, 当前Batch Loss: 2.410677671432495
# Batch 个数 350, 当前Batch Loss: 2.3888254165649414
# Epoch [2/100], Loss: 2.4066
# Batch 个数 0, 当前Batch Loss: 2.362417221069336
# Batch 个数 50, 当前Batch Loss: 2.3613758087158203
# Batch 个数 100, 当前Batch Loss: 2.3792309761047363
# Batch 个数 150, 当前Batch Loss: 2.366511583328247
# Batch 个数 200, 当前Batch Loss: 2.3400487899780273
# Batch 个数 250, 当前Batch Loss: 2.35500431060791
# Batch 个数 300, 当前Batch Loss: 2.353102207183838
# Batch 个数 350, 当前Batch Loss: 2.38850998878479
# Epoch [3/100], Loss: 2.3847
# Batch 个数 0, 当前Batch Loss: 2.410914897918701
# Batch 个数 50, 当前Batch Loss: 2.373826503753662
# Batch 个数 100, 当前Batch Loss: 2.305582284927368
# Batch 个数 150, 当前Batch Loss: 2.313488245010376
# Batch 个数 200, 当前Batch Loss: 2.4023959636688232
# Batch 个数 250, 当前Batch Loss: 2.405454635620117
# Batch 个数 300, 当前Batch Loss: 2.312070846557617
# Batch 个数 350, 当前Batch Loss: 2.3349978923797607
# Epoch [4/100], Loss: 2.3677
# Batch 个数 0, 当前Batch Loss: 2.394540786743164
# Batch 个数 50, 当前Batch Loss: 2.373711347579956
# Batch 个数 100, 当前Batch Loss: 2.365694522857666
# Batch 个数 150, 当前Batch Loss: 2.3198375701904297
# Batch 个数 200, 当前Batch Loss: 2.352747917175293
# Batch 个数 250, 当前Batch Loss: 2.4152770042419434
# Batch 个数 300, 当前Batch Loss: 2.29183292388916
# Batch 个数 350, 当前Batch Loss: 2.3883564472198486
# Epoch [5/100], Loss: 2.3531
# Batch 个数 0, 当前Batch Loss: 2.360175132751465
# Batch 个数 50, 当前Batch Loss: 2.3886449337005615
# Batch 个数 100, 当前Batch Loss: 2.3870325088500977
# Batch 个数 150, 当前Batch Loss: 2.2768824100494385
# Batch 个数 200, 当前Batch Loss: 2.3020434379577637
# Batch 个数 250, 当前Batch Loss: 2.249648332595825
# Batch 个数 300, 当前Batch Loss: 2.4073357582092285
# Batch 个数 350, 当前Batch Loss: 2.262042760848999
# Epoch [6/100], Loss: 2.3368
# Batch 个数 0, 当前Batch Loss: 2.33768367767334
# Batch 个数 50, 当前Batch Loss: 2.2930126190185547
# Batch 个数 100, 当前Batch Loss: 2.309260606765747
# Batch 个数 150, 当前Batch Loss: 2.3315625190734863
# Batch 个数 200, 当前Batch Loss: 2.2650539875030518
# Batch 个数 250, 当前Batch Loss: 2.3128645420074463
# Batch 个数 300, 当前Batch Loss: 2.449352502822876
# Batch 个数 350, 当前Batch Loss: 2.335829734802246
# Epoch [7/100], Loss: 2.3113
# Batch 个数 0, 当前Batch Loss: 2.374667167663574
# Batch 个数 50, 当前Batch Loss: 2.2665622234344482
# Batch 个数 100, 当前Batch Loss: 2.1858277320861816
# Batch 个数 150, 当前Batch Loss: 2.3323686122894287
# Batch 个数 200, 当前Batch Loss: 2.236818790435791
# Batch 个数 250, 当前Batch Loss: 2.21268630027771
# Batch 个数 300, 当前Batch Loss: 2.3020317554473877
# Batch 个数 350, 当前Batch Loss: 2.2477355003356934
# Epoch [8/100], Loss: 2.2515
# Batch 个数 0, 当前Batch Loss: 2.3375072479248047
# Batch 个数 50, 当前Batch Loss: 2.130338668823242
# Batch 个数 100, 当前Batch Loss: 2.1072757244110107
# Batch 个数 150, 当前Batch Loss: 2.1560182571411133
# Batch 个数 200, 当前Batch Loss: 2.1684012413024902
# Batch 个数 250, 当前Batch Loss: 2.0142550468444824
# Batch 个数 300, 当前Batch Loss: 1.967712163925171
# Batch 个数 350, 当前Batch Loss: 1.9187517166137695
# Epoch [9/100], Loss: 2.0739
# Batch 个数 0, 当前Batch Loss: 1.9682444334030151
# Batch 个数 50, 当前Batch Loss: 1.7505825757980347
# Batch 个数 100, 当前Batch Loss: 1.7318381071090698
# Batch 个数 150, 当前Batch Loss: 1.7820080518722534
# Batch 个数 200, 当前Batch Loss: 1.7371279001235962
# Batch 个数 250, 当前Batch Loss: 1.4050414562225342
# Batch 个数 300, 当前Batch Loss: 1.5533969402313232
# Batch 个数 350, 当前Batch Loss: 1.5782420635223389
# Epoch [10/100], Loss: 1.6544
# Batch 个数 0, 当前Batch Loss: 1.4739787578582764
# Batch 个数 50, 当前Batch Loss: 1.6040918827056885
# Batch 个数 100, 当前Batch Loss: 1.3121601343154907
# Batch 个数 150, 当前Batch Loss: 1.3750494718551636
# Batch 个数 200, 当前Batch Loss: 1.3543959856033325
# Batch 个数 250, 当前Batch Loss: 1.2730977535247803
# Batch 个数 300, 当前Batch Loss: 1.3371864557266235
# Batch 个数 350, 当前Batch Loss: 0.7118768095970154
# Epoch [11/100], Loss: 1.2536
# Batch 个数 0, 当前Batch Loss: 0.9337895512580872
# Batch 个数 50, 当前Batch Loss: 1.1099903583526611
# Batch 个数 100, 当前Batch Loss: 0.9785836338996887
# Batch 个数 150, 当前Batch Loss: 1.1606498956680298
# Batch 个数 200, 当前Batch Loss: 0.652070164680481
# Batch 个数 250, 当前Batch Loss: 0.8857564330101013
# Batch 个数 300, 当前Batch Loss: 0.7738965749740601
# Batch 个数 350, 当前Batch Loss: 0.9310120940208435
# Epoch [12/100], Loss: 1.0143
# Batch 个数 0, 当前Batch Loss: 0.7761024832725525
# Batch 个数 50, 当前Batch Loss: 0.7710438966751099
# Batch 个数 100, 当前Batch Loss: 1.061794638633728
# Batch 个数 150, 当前Batch Loss: 0.4921899735927582
# Batch 个数 200, 当前Batch Loss: 0.6776322722434998
# Batch 个数 250, 当前Batch Loss: 0.684217095375061
# Batch 个数 300, 当前Batch Loss: 0.7380944490432739
# Batch 个数 350, 当前Batch Loss: 0.7262867093086243
# Epoch [13/100], Loss: 0.8444
# Batch 个数 0, 当前Batch Loss: 0.6678687334060669
# Batch 个数 50, 当前Batch Loss: 0.6386932134628296
# Batch 个数 100, 当前Batch Loss: 0.726004958152771
# Batch 个数 150, 当前Batch Loss: 0.554800271987915
# Batch 个数 200, 当前Batch Loss: 0.5346648097038269
# Batch 个数 250, 当前Batch Loss: 0.6731013059616089
# Batch 个数 300, 当前Batch Loss: 0.6076700091362
# Batch 个数 350, 当前Batch Loss: 0.9650055766105652
# Epoch [14/100], Loss: 0.7131
# Batch 个数 0, 当前Batch Loss: 0.39155468344688416
# Batch 个数 50, 当前Batch Loss: 1.014702320098877
# Batch 个数 100, 当前Batch Loss: 0.749821662902832
# Batch 个数 150, 当前Batch Loss: 0.44389578700065613
# Batch 个数 200, 当前Batch Loss: 0.40620097517967224
# Batch 个数 250, 当前Batch Loss: 0.9009126424789429
# Batch 个数 300, 当前Batch Loss: 0.7386623620986938
# Batch 个数 350, 当前Batch Loss: 0.6558021903038025
# Epoch [15/100], Loss: 0.6195
# Batch 个数 0, 当前Batch Loss: 0.8109229803085327
# Batch 个数 50, 当前Batch Loss: 0.5051597952842712
# Batch 个数 100, 当前Batch Loss: 0.7447684407234192
# Batch 个数 150, 当前Batch Loss: 0.4188520312309265
# Batch 个数 200, 当前Batch Loss: 0.9177455902099609
# Batch 个数 250, 当前Batch Loss: 0.24445892870426178
# Batch 个数 300, 当前Batch Loss: 0.3767852187156677
# Batch 个数 350, 当前Batch Loss: 0.4626505970954895
# Epoch [16/100], Loss: 0.5510
# Batch 个数 0, 当前Batch Loss: 0.24782265722751617
# Batch 个数 50, 当前Batch Loss: 0.76517653465271
# Batch 个数 100, 当前Batch Loss: 0.7235251069068909
# Batch 个数 150, 当前Batch Loss: 0.5817258954048157
# Batch 个数 200, 当前Batch Loss: 0.3040839731693268
# Batch 个数 250, 当前Batch Loss: 0.5739542245864868
# Batch 个数 300, 当前Batch Loss: 0.23738929629325867
# Batch 个数 350, 当前Batch Loss: 0.16959860920906067
# Epoch [17/100], Loss: 0.5026
# Batch 个数 0, 当前Batch Loss: 0.7212584018707275
# Batch 个数 50, 当前Batch Loss: 0.44458451867103577
# Batch 个数 100, 当前Batch Loss: 0.3998424708843231
# Batch 个数 150, 当前Batch Loss: 0.5502519607543945
# Batch 个数 200, 当前Batch Loss: 0.2511916756629944
# Batch 个数 250, 当前Batch Loss: 0.23141060769557953
# Batch 个数 300, 当前Batch Loss: 0.342604398727417
# Batch 个数 350, 当前Batch Loss: 0.31599321961402893
# Epoch [18/100], Loss: 0.4643
# Batch 个数 0, 当前Batch Loss: 0.4339126944541931
# Batch 个数 50, 当前Batch Loss: 0.6682218313217163
# Batch 个数 100, 当前Batch Loss: 0.25784969329833984
# Batch 个数 150, 当前Batch Loss: 0.49775004386901855
# Batch 个数 200, 当前Batch Loss: 0.1394825428724289
# Batch 个数 250, 当前Batch Loss: 0.42790353298187256
# Batch 个数 300, 当前Batch Loss: 0.3020549416542053
# Batch 个数 350, 当前Batch Loss: 0.36487048864364624
# Epoch [19/100], Loss: 0.4346
# Batch 个数 0, 当前Batch Loss: 0.6168382167816162
# Batch 个数 50, 当前Batch Loss: 0.20343922078609467
# Batch 个数 100, 当前Batch Loss: 0.28289294242858887
# Batch 个数 150, 当前Batch Loss: 0.5668777823448181
# Batch 个数 200, 当前Batch Loss: 0.4604061245918274
# Batch 个数 250, 当前Batch Loss: 0.7136673927307129
# Batch 个数 300, 当前Batch Loss: 0.4688064754009247
# Batch 个数 350, 当前Batch Loss: 0.46382683515548706
# Epoch [20/100], Loss: 0.4100
# Batch 个数 0, 当前Batch Loss: 0.3575233519077301
# Batch 个数 50, 当前Batch Loss: 0.3294640779495239
# Batch 个数 100, 当前Batch Loss: 0.4944291114807129
# Batch 个数 150, 当前Batch Loss: 0.31042635440826416
# Batch 个数 200, 当前Batch Loss: 0.3145695924758911
# Batch 个数 250, 当前Batch Loss: 0.23720987141132355
# Batch 个数 300, 当前Batch Loss: 0.3486143946647644
# Batch 个数 350, 当前Batch Loss: 0.42670509219169617
# Epoch [21/100], Loss: 0.3885
# Batch 个数 0, 当前Batch Loss: 0.16068638861179352
# Batch 个数 50, 当前Batch Loss: 0.3933359682559967
# Batch 个数 100, 当前Batch Loss: 0.3323290944099426
# Batch 个数 150, 当前Batch Loss: 0.7213345170021057
# Batch 个数 200, 当前Batch Loss: 0.328299880027771
# Batch 个数 250, 当前Batch Loss: 0.2167879343032837
# Batch 个数 300, 当前Batch Loss: 0.8810442090034485
# Batch 个数 350, 当前Batch Loss: 0.1825309842824936
# Epoch [22/100], Loss: 0.3699
# Batch 个数 0, 当前Batch Loss: 0.3900284171104431
# Batch 个数 50, 当前Batch Loss: 0.3076286315917969
# Batch 个数 100, 当前Batch Loss: 0.6158479452133179
# Batch 个数 150, 当前Batch Loss: 0.1938442587852478
# Batch 个数 200, 当前Batch Loss: 0.4109046459197998
# Batch 个数 250, 当前Batch Loss: 0.4796006977558136
# Batch 个数 300, 当前Batch Loss: 0.47641468048095703
# Batch 个数 350, 当前Batch Loss: 0.4420762360095978
# Epoch [23/100], Loss: 0.3542
# Batch 个数 0, 当前Batch Loss: 0.36578240990638733
# Batch 个数 50, 当前Batch Loss: 0.2876906991004944
# Batch 个数 100, 当前Batch Loss: 0.13204346597194672
# Batch 个数 150, 当前Batch Loss: 0.5134927034378052
# Batch 个数 200, 当前Batch Loss: 0.36338508129119873
# Batch 个数 250, 当前Batch Loss: 0.20466117560863495
# Batch 个数 300, 当前Batch Loss: 0.38915884494781494
# Batch 个数 350, 当前Batch Loss: 0.34856367111206055
# Epoch [24/100], Loss: 0.3377
# Batch 个数 0, 当前Batch Loss: 0.11968154460191727
# Batch 个数 50, 当前Batch Loss: 0.2216263860464096
# Batch 个数 100, 当前Batch Loss: 0.3315093517303467
# Batch 个数 150, 当前Batch Loss: 0.12220685929059982
# Batch 个数 200, 当前Batch Loss: 0.21897707879543304
# Batch 个数 250, 当前Batch Loss: 0.40145692229270935
# Batch 个数 300, 当前Batch Loss: 0.2109527438879013
# Batch 个数 350, 当前Batch Loss: 0.3000561594963074
# Epoch [25/100], Loss: 0.3236
# Batch 个数 0, 当前Batch Loss: 0.33491215109825134
# Batch 个数 50, 当前Batch Loss: 0.20773226022720337
# Batch 个数 100, 当前Batch Loss: 0.1959894448518753
# Batch 个数 150, 当前Batch Loss: 0.3029839098453522
# Batch 个数 200, 当前Batch Loss: 0.2536674737930298
# Batch 个数 250, 当前Batch Loss: 0.3242896497249603
# Batch 个数 300, 当前Batch Loss: 0.42461395263671875
# Batch 个数 350, 当前Batch Loss: 0.24773895740509033
# Epoch [26/100], Loss: 0.3083
# Batch 个数 0, 当前Batch Loss: 0.26902392506599426
# Batch 个数 50, 当前Batch Loss: 0.30190178751945496
# Batch 个数 100, 当前Batch Loss: 0.23040470480918884
# Batch 个数 150, 当前Batch Loss: 0.40358293056488037
# Batch 个数 200, 当前Batch Loss: 0.14194157719612122
# Batch 个数 250, 当前Batch Loss: 0.4657672345638275
# Batch 个数 300, 当前Batch Loss: 0.22077234089374542
# Batch 个数 350, 当前Batch Loss: 0.11294826120138168
# Epoch [27/100], Loss: 0.2945
# Batch 个数 0, 当前Batch Loss: 0.25972187519073486
# Batch 个数 50, 当前Batch Loss: 0.27183735370635986
# Batch 个数 100, 当前Batch Loss: 0.455879271030426
# Batch 个数 150, 当前Batch Loss: 0.23563507199287415
# Batch 个数 200, 当前Batch Loss: 0.3387600779533386
# Batch 个数 250, 当前Batch Loss: 0.05581251159310341
# Batch 个数 300, 当前Batch Loss: 0.5215531587600708
# Batch 个数 350, 当前Batch Loss: 0.301522433757782
# Epoch [28/100], Loss: 0.2823
# Batch 个数 0, 当前Batch Loss: 0.33882033824920654
# Batch 个数 50, 当前Batch Loss: 0.30413272976875305
# Batch 个数 100, 当前Batch Loss: 0.4386496841907501
# Batch 个数 150, 当前Batch Loss: 0.2943127155303955
# Batch 个数 200, 当前Batch Loss: 0.3182680606842041
# Batch 个数 250, 当前Batch Loss: 0.2658689022064209
# Batch 个数 300, 当前Batch Loss: 0.1928309053182602
# Batch 个数 350, 当前Batch Loss: 0.08897876739501953
# Epoch [29/100], Loss: 0.2691
# Batch 个数 0, 当前Batch Loss: 0.23179759085178375
# Batch 个数 50, 当前Batch Loss: 0.2757122814655304
# Batch 个数 100, 当前Batch Loss: 0.22503064572811127
# Batch 个数 150, 当前Batch Loss: 0.29535749554634094
# Batch 个数 200, 当前Batch Loss: 0.28184154629707336
# Batch 个数 250, 当前Batch Loss: 0.26661643385887146
# Batch 个数 300, 当前Batch Loss: 0.35445597767829895
# Batch 个数 350, 当前Batch Loss: 0.3161884844303131
# Epoch [30/100], Loss: 0.2590
# Batch 个数 0, 当前Batch Loss: 0.0893523097038269
# Batch 个数 50, 当前Batch Loss: 0.17758186161518097
# Batch 个数 100, 当前Batch Loss: 0.3601057827472687
# Batch 个数 150, 当前Batch Loss: 0.2830218970775604
# Batch 个数 200, 当前Batch Loss: 0.46619701385498047
# Batch 个数 250, 当前Batch Loss: 0.2505369782447815
# Batch 个数 300, 当前Batch Loss: 0.2993249297142029
# Batch 个数 350, 当前Batch Loss: 0.06030435860157013
# Epoch [31/100], Loss: 0.2466
# Batch 个数 0, 当前Batch Loss: 0.30342981219291687
# Batch 个数 50, 当前Batch Loss: 0.28284159302711487
# Batch 个数 100, 当前Batch Loss: 0.1783525049686432
# Batch 个数 150, 当前Batch Loss: 0.25828832387924194
# Batch 个数 200, 当前Batch Loss: 0.3421090841293335
# Batch 个数 250, 当前Batch Loss: 0.2953757047653198
# Batch 个数 300, 当前Batch Loss: 0.10165901482105255
# Batch 个数 350, 当前Batch Loss: 0.2965856194496155
# Epoch [32/100], Loss: 0.2369
# Batch 个数 0, 当前Batch Loss: 0.08890638500452042
# Batch 个数 50, 当前Batch Loss: 0.1634513884782791
# Batch 个数 100, 当前Batch Loss: 0.12807036936283112
# Batch 个数 150, 当前Batch Loss: 0.12818247079849243
# Batch 个数 200, 当前Batch Loss: 0.06854815036058426
# Batch 个数 250, 当前Batch Loss: 0.17469513416290283
# Batch 个数 300, 当前Batch Loss: 0.25984856486320496
# Batch 个数 350, 当前Batch Loss: 0.5757458209991455
# Epoch [33/100], Loss: 0.2266
# Batch 个数 0, 当前Batch Loss: 0.36934664845466614
# Batch 个数 50, 当前Batch Loss: 0.06646503508090973
# Batch 个数 100, 当前Batch Loss: 0.17902769148349762
# Batch 个数 150, 当前Batch Loss: 0.3509852886199951
# Batch 个数 200, 当前Batch Loss: 0.20530501008033752
# Batch 个数 250, 当前Batch Loss: 0.08040410280227661
# Batch 个数 300, 当前Batch Loss: 0.4091116189956665
# Batch 个数 350, 当前Batch Loss: 0.21639995276927948
# Epoch [34/100], Loss: 0.2140
# Batch 个数 0, 当前Batch Loss: 0.15719792246818542
# Batch 个数 50, 当前Batch Loss: 0.19733890891075134
# Batch 个数 100, 当前Batch Loss: 0.13857033848762512
# Batch 个数 150, 当前Batch Loss: 0.10266614705324173
# Batch 个数 200, 当前Batch Loss: 0.32938152551651
# Batch 个数 250, 当前Batch Loss: 0.028013082221150398
# Batch 个数 300, 当前Batch Loss: 0.18706995248794556
# Batch 个数 350, 当前Batch Loss: 0.13481405377388
# Epoch [35/100], Loss: 0.2039
# Batch 个数 0, 当前Batch Loss: 0.15168166160583496
# Batch 个数 50, 当前Batch Loss: 0.1099214032292366
# Batch 个数 100, 当前Batch Loss: 0.13274385035037994
# Batch 个数 150, 当前Batch Loss: 0.355985164642334
# Batch 个数 200, 当前Batch Loss: 0.17188771069049835
# Batch 个数 250, 当前Batch Loss: 0.2740517258644104
# Batch 个数 300, 当前Batch Loss: 0.11864995211362839
# Batch 个数 350, 当前Batch Loss: 0.2653442323207855
# Epoch [36/100], Loss: 0.1944
# Batch 个数 0, 当前Batch Loss: 0.27760422229766846
# Batch 个数 50, 当前Batch Loss: 0.07862875610589981
# Batch 个数 100, 当前Batch Loss: 0.21157000958919525
# Batch 个数 150, 当前Batch Loss: 0.08424755185842514
# Batch 个数 200, 当前Batch Loss: 0.11791430413722992
# Batch 个数 250, 当前Batch Loss: 0.11788573861122131
# Batch 个数 300, 当前Batch Loss: 0.19051715731620789
# Batch 个数 350, 当前Batch Loss: 0.05661071464419365
# Epoch [37/100], Loss: 0.1831
# Batch 个数 0, 当前Batch Loss: 0.2509574592113495
# Batch 个数 50, 当前Batch Loss: 0.175357386469841
# Batch 个数 100, 当前Batch Loss: 0.09259942173957825
# Batch 个数 150, 当前Batch Loss: 0.21495400369167328
# Batch 个数 200, 当前Batch Loss: 0.03166072815656662
# Batch 个数 250, 当前Batch Loss: 0.07795347273349762
# Batch 个数 300, 当前Batch Loss: 0.22043591737747192
# Batch 个数 350, 当前Batch Loss: 0.22133514285087585
# Epoch [38/100], Loss: 0.1735
# Batch 个数 0, 当前Batch Loss: 0.13312354683876038
# Batch 个数 50, 当前Batch Loss: 0.2330729067325592
# Batch 个数 100, 当前Batch Loss: 0.19538255035877228
# Batch 个数 150, 当前Batch Loss: 0.11118558049201965
# Batch 个数 200, 当前Batch Loss: 0.1779356449842453
# Batch 个数 250, 当前Batch Loss: 0.08796138316392899
# Batch 个数 300, 当前Batch Loss: 0.21519942581653595
# Batch 个数 350, 当前Batch Loss: 0.06566751003265381
# Epoch [39/100], Loss: 0.1646
# Batch 个数 0, 当前Batch Loss: 0.08992312103509903
# Batch 个数 50, 当前Batch Loss: 0.04664008691906929
# Batch 个数 100, 当前Batch Loss: 0.21998251974582672
# Batch 个数 150, 当前Batch Loss: 0.12658388912677765
# Batch 个数 200, 当前Batch Loss: 0.1784764677286148
# Batch 个数 250, 当前Batch Loss: 0.20672789216041565
# Batch 个数 300, 当前Batch Loss: 0.2214282602071762
# Batch 个数 350, 当前Batch Loss: 0.10402728617191315
# Epoch [40/100], Loss: 0.1558
# Batch 个数 0, 当前Batch Loss: 0.08204128593206406
# Batch 个数 50, 当前Batch Loss: 0.1019197478890419
# Batch 个数 100, 当前Batch Loss: 0.11968405544757843
# Batch 个数 150, 当前Batch Loss: 0.13185814023017883
# Batch 个数 200, 当前Batch Loss: 0.16147945821285248
# Batch 个数 250, 当前Batch Loss: 0.16919218003749847
# Batch 个数 300, 当前Batch Loss: 0.1394028216600418
# Batch 个数 350, 当前Batch Loss: 0.225441575050354
# Epoch [41/100], Loss: 0.1485
# Batch 个数 0, 当前Batch Loss: 0.10590518265962601
# Batch 个数 50, 当前Batch Loss: 0.17323507368564606
# Batch 个数 100, 当前Batch Loss: 0.15882034599781036
# Batch 个数 150, 当前Batch Loss: 0.2730979323387146
# Batch 个数 200, 当前Batch Loss: 0.12196779996156693
# Batch 个数 250, 当前Batch Loss: 0.04816773533821106
# Batch 个数 300, 当前Batch Loss: 0.11136685311794281
# Batch 个数 350, 当前Batch Loss: 0.3200698792934418
# Epoch [42/100], Loss: 0.1403
# Batch 个数 0, 当前Batch Loss: 0.22021493315696716
# Batch 个数 50, 当前Batch Loss: 0.20014344155788422
# Batch 个数 100, 当前Batch Loss: 0.2615910768508911
# Batch 个数 150, 当前Batch Loss: 0.12146749347448349
# Batch 个数 200, 当前Batch Loss: 0.1290719360113144
# Batch 个数 250, 当前Batch Loss: 0.08570767939090729
# Batch 个数 300, 当前Batch Loss: 0.18993981182575226
# Batch 个数 350, 当前Batch Loss: 0.13574057817459106
# Epoch [43/100], Loss: 0.1329
# Batch 个数 0, 当前Batch Loss: 0.16351944208145142
# Batch 个数 50, 当前Batch Loss: 0.20869910717010498
# Batch 个数 100, 当前Batch Loss: 0.11480239033699036
# Batch 个数 150, 当前Batch Loss: 0.179573655128479
# Batch 个数 200, 当前Batch Loss: 0.11873805522918701
# Batch 个数 250, 当前Batch Loss: 0.11422117799520493
# Batch 个数 300, 当前Batch Loss: 0.06500262767076492
# Batch 个数 350, 当前Batch Loss: 0.19829973578453064
# Epoch [44/100], Loss: 0.1264
# Batch 个数 0, 当前Batch Loss: 0.1982058882713318
# Batch 个数 50, 当前Batch Loss: 0.08366867154836655
# Batch 个数 100, 当前Batch Loss: 0.09514118731021881
# Batch 个数 150, 当前Batch Loss: 0.09679348021745682
# Batch 个数 200, 当前Batch Loss: 0.08287496119737625
# Batch 个数 250, 当前Batch Loss: 0.15390916168689728
# Batch 个数 300, 当前Batch Loss: 0.049075935035943985
# Batch 个数 350, 当前Batch Loss: 0.19248956441879272
# Epoch [45/100], Loss: 0.1199
# Batch 个数 0, 当前Batch Loss: 0.13059592247009277
# Batch 个数 50, 当前Batch Loss: 0.09480516612529755
# Batch 个数 100, 当前Batch Loss: 0.09659140557050705
# Batch 个数 150, 当前Batch Loss: 0.15954507887363434
# Batch 个数 200, 当前Batch Loss: 0.31335684657096863
# Batch 个数 250, 当前Batch Loss: 0.07966966181993484
# Batch 个数 300, 当前Batch Loss: 0.14114439487457275
# Batch 个数 350, 当前Batch Loss: 0.0764307975769043
# Epoch [46/100], Loss: 0.1147
# Batch 个数 0, 当前Batch Loss: 0.16898371279239655
# Batch 个数 50, 当前Batch Loss: 0.09884732216596603
# Batch 个数 100, 当前Batch Loss: 0.14682702720165253
# Batch 个数 150, 当前Batch Loss: 0.035010479390621185
# Batch 个数 200, 当前Batch Loss: 0.11496251076459885
# Batch 个数 250, 当前Batch Loss: 0.07919418066740036
# Batch 个数 300, 当前Batch Loss: 0.032749149948358536
# Batch 个数 350, 当前Batch Loss: 0.2198428064584732
# Epoch [47/100], Loss: 0.1077
# Batch 个数 0, 当前Batch Loss: 0.08537359535694122
# Batch 个数 50, 当前Batch Loss: 0.03299638256430626
# Batch 个数 100, 当前Batch Loss: 0.07300303876399994
# Batch 个数 150, 当前Batch Loss: 0.053232546895742416
# Batch 个数 200, 当前Batch Loss: 0.11897888779640198
# Batch 个数 250, 当前Batch Loss: 0.08827099204063416
# Batch 个数 300, 当前Batch Loss: 0.10830001533031464
# Batch 个数 350, 当前Batch Loss: 0.045199450105428696
# Epoch [48/100], Loss: 0.1010
# Batch 个数 0, 当前Batch Loss: 0.05019295960664749
# Batch 个数 50, 当前Batch Loss: 0.14528612792491913
# Batch 个数 100, 当前Batch Loss: 0.08169534057378769
# Batch 个数 150, 当前Batch Loss: 0.10183216631412506
# Batch 个数 200, 当前Batch Loss: 0.17880390584468842
# Batch 个数 250, 当前Batch Loss: 0.03909305855631828
# Batch 个数 300, 当前Batch Loss: 0.09310569614171982
# Batch 个数 350, 当前Batch Loss: 0.03856303170323372
# Epoch [49/100], Loss: 0.0961
# Batch 个数 0, 当前Batch Loss: 0.027096010744571686
# Batch 个数 50, 当前Batch Loss: 0.05056433007121086
# Batch 个数 100, 当前Batch Loss: 0.04752852022647858
# Batch 个数 150, 当前Batch Loss: 0.08694897592067719
# Batch 个数 200, 当前Batch Loss: 0.04625670984387398
# Batch 个数 250, 当前Batch Loss: 0.09010656923055649
# Batch 个数 300, 当前Batch Loss: 0.052053432911634445
# Batch 个数 350, 当前Batch Loss: 0.07493497431278229
# Epoch [50/100], Loss: 0.0907
# Batch 个数 0, 当前Batch Loss: 0.060791704803705215
# Batch 个数 50, 当前Batch Loss: 0.07621429860591888
# Batch 个数 100, 当前Batch Loss: 0.2410898655653
# Batch 个数 150, 当前Batch Loss: 0.1453336626291275
# Batch 个数 200, 当前Batch Loss: 0.024584658443927765
# Batch 个数 250, 当前Batch Loss: 0.08824609220027924
# Batch 个数 300, 当前Batch Loss: 0.03209839388728142
# Batch 个数 350, 当前Batch Loss: 0.09675949811935425
# Epoch [51/100], Loss: 0.0849
# Batch 个数 0, 当前Batch Loss: 0.13795910775661469
# Batch 个数 50, 当前Batch Loss: 0.15281663835048676
# Batch 个数 100, 当前Batch Loss: 0.031019816175103188
# Batch 个数 150, 当前Batch Loss: 0.05561654642224312
# Batch 个数 200, 当前Batch Loss: 0.014413729310035706
# Batch 个数 250, 当前Batch Loss: 0.10981844365596771
# Batch 个数 300, 当前Batch Loss: 0.16667267680168152
# Batch 个数 350, 当前Batch Loss: 0.053764667361974716
# Epoch [52/100], Loss: 0.0809
# Batch 个数 0, 当前Batch Loss: 0.05926799401640892
# Batch 个数 50, 当前Batch Loss: 0.07451330125331879
# Batch 个数 100, 当前Batch Loss: 0.059936825186014175
# Batch 个数 150, 当前Batch Loss: 0.02090388722717762
# Batch 个数 200, 当前Batch Loss: 0.08877195417881012
# Batch 个数 250, 当前Batch Loss: 0.017507333308458328
# Batch 个数 300, 当前Batch Loss: 0.2536541819572449
# Batch 个数 350, 当前Batch Loss: 0.03502262011170387
# Epoch [53/100], Loss: 0.0767
# Batch 个数 0, 当前Batch Loss: 0.029518384486436844
# Batch 个数 50, 当前Batch Loss: 0.12988048791885376
# Batch 个数 100, 当前Batch Loss: 0.04252931848168373
# Batch 个数 150, 当前Batch Loss: 0.022017400711774826
# Batch 个数 200, 当前Batch Loss: 0.027134591713547707
# Batch 个数 250, 当前Batch Loss: 0.1946054846048355
# Batch 个数 300, 当前Batch Loss: 0.09613825380802155
# Batch 个数 350, 当前Batch Loss: 0.024716323241591454
# Epoch [54/100], Loss: 0.0733
# Batch 个数 0, 当前Batch Loss: 0.06359685957431793
# Batch 个数 50, 当前Batch Loss: 0.00460374541580677
# Batch 个数 100, 当前Batch Loss: 0.10242209583520889
# Batch 个数 150, 当前Batch Loss: 0.027679618448019028
# Batch 个数 200, 当前Batch Loss: 0.1791859269142151
# Batch 个数 250, 当前Batch Loss: 0.05385003983974457
# Batch 个数 300, 当前Batch Loss: 0.2912743091583252
# Batch 个数 350, 当前Batch Loss: 0.007635742891579866
# Epoch [55/100], Loss: 0.0695
# Batch 个数 0, 当前Batch Loss: 0.21255479753017426
# Batch 个数 50, 当前Batch Loss: 0.036870162934064865
# Batch 个数 100, 当前Batch Loss: 0.02926049567759037
# Batch 个数 150, 当前Batch Loss: 0.027757937088608742
# Batch 个数 200, 当前Batch Loss: 0.04573964700102806
# Batch 个数 250, 当前Batch Loss: 0.16035105288028717
# Batch 个数 300, 当前Batch Loss: 0.07257111370563507
# Batch 个数 350, 当前Batch Loss: 0.09767428040504456
# Epoch [56/100], Loss: 0.0667
# Batch 个数 0, 当前Batch Loss: 0.12827132642269135
# Batch 个数 50, 当前Batch Loss: 0.0309380404651165
# Batch 个数 100, 当前Batch Loss: 0.06706157326698303
# Batch 个数 150, 当前Batch Loss: 0.21575504541397095
# Batch 个数 200, 当前Batch Loss: 0.06314673274755478
# Batch 个数 250, 当前Batch Loss: 0.18804778158664703
# Batch 个数 300, 当前Batch Loss: 0.016299661248922348
# Batch 个数 350, 当前Batch Loss: 0.0665152296423912
# Epoch [57/100], Loss: 0.0619
# Batch 个数 0, 当前Batch Loss: 0.064040407538414
# Batch 个数 50, 当前Batch Loss: 0.01181928813457489
# Batch 个数 100, 当前Batch Loss: 0.022968623787164688
# Batch 个数 150, 当前Batch Loss: 0.03494146093726158
# Batch 个数 200, 当前Batch Loss: 0.04886658117175102
# Batch 个数 250, 当前Batch Loss: 0.09521409124135971
# Batch 个数 300, 当前Batch Loss: 0.04140496253967285
# Batch 个数 350, 当前Batch Loss: 0.07340347021818161
# Epoch [58/100], Loss: 0.0578
# Batch 个数 0, 当前Batch Loss: 0.07105938345193863
# Batch 个数 50, 当前Batch Loss: 0.03193778172135353
# Batch 个数 100, 当前Batch Loss: 0.05783020332455635
# Batch 个数 150, 当前Batch Loss: 0.05959153175354004
# Batch 个数 200, 当前Batch Loss: 0.030743513256311417
# Batch 个数 250, 当前Batch Loss: 0.015635715797543526
# Batch 个数 300, 当前Batch Loss: 0.052314989268779755
# Batch 个数 350, 当前Batch Loss: 0.043889936059713364
# Epoch [59/100], Loss: 0.0547
# Batch 个数 0, 当前Batch Loss: 0.030339159071445465
# Batch 个数 50, 当前Batch Loss: 0.010938615538179874
# Batch 个数 100, 当前Batch Loss: 0.06166266277432442
# Batch 个数 150, 当前Batch Loss: 0.07738720625638962
# Batch 个数 200, 当前Batch Loss: 0.033799927681684494
# Batch 个数 250, 当前Batch Loss: 0.050910551100969315
# Batch 个数 300, 当前Batch Loss: 0.055622514337301254
# Batch 个数 350, 当前Batch Loss: 0.04190057888627052
# Epoch [60/100], Loss: 0.0514
# Batch 个数 0, 当前Batch Loss: 0.12867844104766846
# Batch 个数 50, 当前Batch Loss: 0.08734460175037384
# Batch 个数 100, 当前Batch Loss: 0.04834304004907608
# Batch 个数 150, 当前Batch Loss: 0.04175504297018051
# Batch 个数 200, 当前Batch Loss: 0.044477615505456924
# Batch 个数 250, 当前Batch Loss: 0.07431115955114365
# Batch 个数 300, 当前Batch Loss: 0.05496678501367569
# Batch 个数 350, 当前Batch Loss: 0.042560696601867676
# Epoch [61/100], Loss: 0.0487
# Batch 个数 0, 当前Batch Loss: 0.010129866190254688
# Batch 个数 50, 当前Batch Loss: 0.12338557839393616
# Batch 个数 100, 当前Batch Loss: 0.06711126118898392
# Batch 个数 150, 当前Batch Loss: 0.06160852313041687
# Batch 个数 200, 当前Batch Loss: 0.05905969813466072
# Batch 个数 250, 当前Batch Loss: 0.03375896066427231
# Batch 个数 300, 当前Batch Loss: 0.03152529150247574
# Batch 个数 350, 当前Batch Loss: 0.0804065614938736
# Epoch [62/100], Loss: 0.0460
# Batch 个数 0, 当前Batch Loss: 0.08063770085573196
# Batch 个数 50, 当前Batch Loss: 0.04770790413022041
# Batch 个数 100, 当前Batch Loss: 0.02764834277331829
# Batch 个数 150, 当前Batch Loss: 0.032875098288059235
# Batch 个数 200, 当前Batch Loss: 0.047694720327854156
# Batch 个数 250, 当前Batch Loss: 0.06212962418794632
# Batch 个数 300, 当前Batch Loss: 0.0361945815384388
# Batch 个数 350, 当前Batch Loss: 0.014376958832144737
# Epoch [63/100], Loss: 0.0441
# Batch 个数 0, 当前Batch Loss: 0.015459470450878143
# Batch 个数 50, 当前Batch Loss: 0.029999420046806335
# Batch 个数 100, 当前Batch Loss: 0.0753595381975174
# Batch 个数 150, 当前Batch Loss: 0.04037103429436684
# Batch 个数 200, 当前Batch Loss: 0.05777711048722267
# Batch 个数 250, 当前Batch Loss: 0.02978665940463543
# Batch 个数 300, 当前Batch Loss: 0.02949218638241291
# Batch 个数 350, 当前Batch Loss: 0.05084694176912308
# Epoch [64/100], Loss: 0.0411
# Batch 个数 0, 当前Batch Loss: 0.022148773074150085
# Batch 个数 50, 当前Batch Loss: 0.017779948189854622
# Batch 个数 100, 当前Batch Loss: 0.014974139630794525
# Batch 个数 150, 当前Batch Loss: 0.029235053807497025
# Batch 个数 200, 当前Batch Loss: 0.042129576206207275
# Batch 个数 250, 当前Batch Loss: 0.05096343159675598
# Batch 个数 300, 当前Batch Loss: 0.037489697337150574
# Batch 个数 350, 当前Batch Loss: 0.02809283882379532
# Epoch [65/100], Loss: 0.0389
# Batch 个数 0, 当前Batch Loss: 0.03599181026220322
# Batch 个数 50, 当前Batch Loss: 0.023271042853593826
# Batch 个数 100, 当前Batch Loss: 0.014392598532140255
# Batch 个数 150, 当前Batch Loss: 0.045835912227630615
# Batch 个数 200, 当前Batch Loss: 0.03656010329723358
# Batch 个数 250, 当前Batch Loss: 0.05140237882733345
# Batch 个数 300, 当前Batch Loss: 0.016797097399830818
# Batch 个数 350, 当前Batch Loss: 0.016959086060523987
# Epoch [66/100], Loss: 0.0372
# Batch 个数 0, 当前Batch Loss: 0.02781151980161667
# Batch 个数 50, 当前Batch Loss: 0.02639954350888729
# Batch 个数 100, 当前Batch Loss: 0.01663305051624775
# Batch 个数 150, 当前Batch Loss: 0.038898222148418427
# Batch 个数 200, 当前Batch Loss: 0.0630224272608757
# Batch 个数 250, 当前Batch Loss: 0.017910845577716827
# Batch 个数 300, 当前Batch Loss: 0.023022573441267014
# Batch 个数 350, 当前Batch Loss: 0.03390964865684509
# Epoch [67/100], Loss: 0.0351
# Batch 个数 0, 当前Batch Loss: 0.020892757922410965
# Batch 个数 50, 当前Batch Loss: 0.038981955498456955
# Batch 个数 100, 当前Batch Loss: 0.014155574142932892
# Batch 个数 150, 当前Batch Loss: 0.06674444675445557
# Batch 个数 200, 当前Batch Loss: 0.04120471701025963
# Batch 个数 250, 当前Batch Loss: 0.06410521268844604
# Batch 个数 300, 当前Batch Loss: 0.03632913902401924
# Batch 个数 350, 当前Batch Loss: 0.022946564480662346
# Epoch [68/100], Loss: 0.0331
# Batch 个数 0, 当前Batch Loss: 0.018037131056189537
# Batch 个数 50, 当前Batch Loss: 0.04073026776313782
# Batch 个数 100, 当前Batch Loss: 0.04578966274857521
# Batch 个数 150, 当前Batch Loss: 0.012798556126654148
# Batch 个数 200, 当前Batch Loss: 0.02619241178035736
# Batch 个数 250, 当前Batch Loss: 0.060364242643117905
# Batch 个数 300, 当前Batch Loss: 0.06324228644371033
# Batch 个数 350, 当前Batch Loss: 0.02394012361764908
# Epoch [69/100], Loss: 0.0314
# Batch 个数 0, 当前Batch Loss: 0.031541671603918076
# Batch 个数 50, 当前Batch Loss: 0.017274437472224236
# Batch 个数 100, 当前Batch Loss: 0.016396578401327133
# Batch 个数 150, 当前Batch Loss: 0.02138293907046318
# Batch 个数 200, 当前Batch Loss: 0.00643707113340497
# Batch 个数 250, 当前Batch Loss: 0.00945593323558569
# Batch 个数 300, 当前Batch Loss: 0.006734990514814854
# Batch 个数 350, 当前Batch Loss: 0.15766283869743347
# Epoch [70/100], Loss: 0.0300
# Batch 个数 0, 当前Batch Loss: 0.01826506480574608
# Batch 个数 50, 当前Batch Loss: 0.024980487301945686
# Batch 个数 100, 当前Batch Loss: 0.005998398642987013
# Batch 个数 150, 当前Batch Loss: 0.007178429048508406
# Batch 个数 200, 当前Batch Loss: 0.03577110543847084
# Batch 个数 250, 当前Batch Loss: 0.023578986525535583
# Batch 个数 300, 当前Batch Loss: 0.01698228530585766
# Batch 个数 350, 当前Batch Loss: 0.018022051081061363
# Epoch [71/100], Loss: 0.0284
# Batch 个数 0, 当前Batch Loss: 0.01765727810561657
# Batch 个数 50, 当前Batch Loss: 0.007256967481225729
# Batch 个数 100, 当前Batch Loss: 0.004039156716316938
# Batch 个数 150, 当前Batch Loss: 0.027500636875629425
# Batch 个数 200, 当前Batch Loss: 0.025911331176757812
# Batch 个数 250, 当前Batch Loss: 0.01771540567278862
# Batch 个数 300, 当前Batch Loss: 0.02454565279185772
# Batch 个数 350, 当前Batch Loss: 0.010525279678404331
# Epoch [72/100], Loss: 0.0266
# Batch 个数 0, 当前Batch Loss: 0.011875932104885578
# Batch 个数 50, 当前Batch Loss: 0.015385091304779053
# Batch 个数 100, 当前Batch Loss: 0.022090397775173187
# Batch 个数 150, 当前Batch Loss: 0.011065197177231312
# Batch 个数 200, 当前Batch Loss: 0.004281758330762386
# Batch 个数 250, 当前Batch Loss: 0.01199060957878828
# Batch 个数 300, 当前Batch Loss: 0.011105111800134182
# Batch 个数 350, 当前Batch Loss: 0.012432420626282692
# Epoch [73/100], Loss: 0.0256
# Batch 个数 0, 当前Batch Loss: 0.010805157013237476
# Batch 个数 50, 当前Batch Loss: 0.04050952568650246
# Batch 个数 100, 当前Batch Loss: 0.009083378128707409
# Batch 个数 150, 当前Batch Loss: 0.056672707200050354
# Batch 个数 200, 当前Batch Loss: 0.021480392664670944
# Batch 个数 250, 当前Batch Loss: 0.03019062615931034
# Batch 个数 300, 当前Batch Loss: 0.026096699759364128
# Batch 个数 350, 当前Batch Loss: 0.024851134046912193
# Epoch [74/100], Loss: 0.0242
# Batch 个数 0, 当前Batch Loss: 0.009622573852539062
# Batch 个数 50, 当前Batch Loss: 0.052709225565195084
# Batch 个数 100, 当前Batch Loss: 0.01704995147883892
# Batch 个数 150, 当前Batch Loss: 0.005442617926746607
# Batch 个数 200, 当前Batch Loss: 0.03213861957192421
# Batch 个数 250, 当前Batch Loss: 0.0067522721365094185
# Batch 个数 300, 当前Batch Loss: 0.018033696338534355
# Batch 个数 350, 当前Batch Loss: 0.022404128685593605
# Epoch [75/100], Loss: 0.0230
# Batch 个数 0, 当前Batch Loss: 0.007185718044638634
# Batch 个数 50, 当前Batch Loss: 0.018565597012639046
# Batch 个数 100, 当前Batch Loss: 0.019101206213235855
# Batch 个数 150, 当前Batch Loss: 0.023896291851997375
# Batch 个数 200, 当前Batch Loss: 0.02237575314939022
# Batch 个数 250, 当前Batch Loss: 0.025397460907697678
# Batch 个数 300, 当前Batch Loss: 0.013491254299879074
# Batch 个数 350, 当前Batch Loss: 0.010374770499765873
# Epoch [76/100], Loss: 0.0217
# Batch 个数 0, 当前Batch Loss: 0.018331443890929222
# Batch 个数 50, 当前Batch Loss: 0.009848237037658691
# Batch 个数 100, 当前Batch Loss: 0.01183745265007019
# Batch 个数 150, 当前Batch Loss: 0.009397026151418686
# Batch 个数 200, 当前Batch Loss: 0.0078119863756000996
# Batch 个数 250, 当前Batch Loss: 0.017950547859072685
# Batch 个数 300, 当前Batch Loss: 0.021537283435463905
# Batch 个数 350, 当前Batch Loss: 0.043895602226257324
# Epoch [77/100], Loss: 0.0206
# Batch 个数 0, 当前Batch Loss: 0.011381263844668865
# Batch 个数 50, 当前Batch Loss: 0.007079525850713253
# Batch 个数 100, 当前Batch Loss: 0.013672619126737118
# Batch 个数 150, 当前Batch Loss: 0.01104460284113884
# Batch 个数 200, 当前Batch Loss: 0.015140766277909279
# Batch 个数 250, 当前Batch Loss: 0.01371536310762167
# Batch 个数 300, 当前Batch Loss: 0.04188144579529762
# Batch 个数 350, 当前Batch Loss: 0.020259637385606766
# Epoch [78/100], Loss: 0.0198
# Batch 个数 0, 当前Batch Loss: 0.007530198432505131
# Batch 个数 50, 当前Batch Loss: 0.009914649650454521
# Batch 个数 100, 当前Batch Loss: 0.017512254416942596
# Batch 个数 150, 当前Batch Loss: 0.009945911355316639
# Batch 个数 200, 当前Batch Loss: 0.016145063564181328
# Batch 个数 250, 当前Batch Loss: 0.21173985302448273
# Batch 个数 300, 当前Batch Loss: 0.03892192244529724
# Batch 个数 350, 当前Batch Loss: 0.010713903233408928
# Epoch [79/100], Loss: 0.0189
# Batch 个数 0, 当前Batch Loss: 0.008734243921935558
# Batch 个数 50, 当前Batch Loss: 0.009662664495408535
# Batch 个数 100, 当前Batch Loss: 0.01914137415587902
# Batch 个数 150, 当前Batch Loss: 0.023904699832201004
# Batch 个数 200, 当前Batch Loss: 0.009160922840237617
# Batch 个数 250, 当前Batch Loss: 0.02538815326988697
# Batch 个数 300, 当前Batch Loss: 0.009654861874878407
# Batch 个数 350, 当前Batch Loss: 0.009729656390845776
# Epoch [80/100], Loss: 0.0178
# Batch 个数 0, 当前Batch Loss: 0.002984330989420414
# Batch 个数 50, 当前Batch Loss: 0.033889975398778915
# Batch 个数 100, 当前Batch Loss: 0.025339799001812935
# Batch 个数 150, 当前Batch Loss: 0.018003297969698906
# Batch 个数 200, 当前Batch Loss: 0.009177600964903831
# Batch 个数 250, 当前Batch Loss: 0.019989807158708572
# Batch 个数 300, 当前Batch Loss: 0.013412263244390488
# Batch 个数 350, 当前Batch Loss: 0.025450658053159714
# Epoch [81/100], Loss: 0.0169
# Batch 个数 0, 当前Batch Loss: 0.009283216670155525
# Batch 个数 50, 当前Batch Loss: 0.00874702911823988
# Batch 个数 100, 当前Batch Loss: 0.01155214011669159
# Batch 个数 150, 当前Batch Loss: 0.007882331497967243
# Batch 个数 200, 当前Batch Loss: 0.01566862314939499
# Batch 个数 250, 当前Batch Loss: 0.0142521308735013
# Batch 个数 300, 当前Batch Loss: 0.009633014909923077
# Batch 个数 350, 当前Batch Loss: 0.01991916634142399
# Epoch [82/100], Loss: 0.0162
# Batch 个数 0, 当前Batch Loss: 0.010491486638784409
# Batch 个数 50, 当前Batch Loss: 0.023671485483646393
# Batch 个数 100, 当前Batch Loss: 0.0065428102388978004
# Batch 个数 150, 当前Batch Loss: 0.014201988466084003
# Batch 个数 200, 当前Batch Loss: 0.01357588917016983
# Batch 个数 250, 当前Batch Loss: 0.010712136514484882
# Batch 个数 300, 当前Batch Loss: 0.005670805461704731
# Batch 个数 350, 当前Batch Loss: 0.011905662715435028
# Epoch [83/100], Loss: 0.0154
# Batch 个数 0, 当前Batch Loss: 0.023973627015948296
# Batch 个数 50, 当前Batch Loss: 0.0029871577862650156
# Batch 个数 100, 当前Batch Loss: 0.006049968767911196
# Batch 个数 150, 当前Batch Loss: 0.011425563134253025
# Batch 个数 200, 当前Batch Loss: 0.00936494767665863
# Batch 个数 250, 当前Batch Loss: 0.002761788433417678
# Batch 个数 300, 当前Batch Loss: 0.013429575599730015
# Batch 个数 350, 当前Batch Loss: 0.010592994280159473
# Epoch [84/100], Loss: 0.0148
# Batch 个数 0, 当前Batch Loss: 0.00733000785112381
# Batch 个数 50, 当前Batch Loss: 0.0010809096274897456
# Batch 个数 100, 当前Batch Loss: 0.011844088323414326
# Batch 个数 150, 当前Batch Loss: 0.019736235961318016
# Batch 个数 200, 当前Batch Loss: 0.01488301157951355
# Batch 个数 250, 当前Batch Loss: 0.025262128561735153
# Batch 个数 300, 当前Batch Loss: 0.017642751336097717
# Batch 个数 350, 当前Batch Loss: 0.013267610222101212
# Epoch [85/100], Loss: 0.0141
# Batch 个数 0, 当前Batch Loss: 0.0058220187202095985
# Batch 个数 50, 当前Batch Loss: 0.008038376457989216
# Batch 个数 100, 当前Batch Loss: 0.01636500656604767
# Batch 个数 150, 当前Batch Loss: 0.0038120634853839874
# Batch 个数 200, 当前Batch Loss: 0.012089141644537449
# Batch 个数 250, 当前Batch Loss: 0.05950810760259628
# Batch 个数 300, 当前Batch Loss: 0.016051068902015686
# Batch 个数 350, 当前Batch Loss: 0.02022438868880272
# Epoch [86/100], Loss: 0.0137
# Batch 个数 0, 当前Batch Loss: 0.025527553632855415
# Batch 个数 50, 当前Batch Loss: 0.007632682099938393
# Batch 个数 100, 当前Batch Loss: 0.007109399884939194
# Batch 个数 150, 当前Batch Loss: 0.00838300958275795
# Batch 个数 200, 当前Batch Loss: 0.007429866585880518
# Batch 个数 250, 当前Batch Loss: 0.008041713386774063
# Batch 个数 300, 当前Batch Loss: 0.014658570289611816
# Batch 个数 350, 当前Batch Loss: 0.002976180985569954
# Epoch [87/100], Loss: 0.0130
# Batch 个数 0, 当前Batch Loss: 0.0077131339348852634
# Batch 个数 50, 当前Batch Loss: 0.04020344093441963
# Batch 个数 100, 当前Batch Loss: 0.009060148149728775
# Batch 个数 150, 当前Batch Loss: 0.006904440000653267
# Batch 个数 200, 当前Batch Loss: 0.007663294207304716
# Batch 个数 250, 当前Batch Loss: 0.008246324956417084
# Batch 个数 300, 当前Batch Loss: 0.028197018429636955
# Batch 个数 350, 当前Batch Loss: 0.006940035615116358
# Epoch [88/100], Loss: 0.0124
# Batch 个数 0, 当前Batch Loss: 0.007735145278275013
# Batch 个数 50, 当前Batch Loss: 0.005724285263568163
# Batch 个数 100, 当前Batch Loss: 0.005599556490778923
# Batch 个数 150, 当前Batch Loss: 0.016336161643266678
# Batch 个数 200, 当前Batch Loss: 0.0193085428327322
# Batch 个数 250, 当前Batch Loss: 0.011811683885753155
# Batch 个数 300, 当前Batch Loss: 0.006376261822879314
# Batch 个数 350, 当前Batch Loss: 0.0010186416329815984
# Epoch [89/100], Loss: 0.0119
# Batch 个数 0, 当前Batch Loss: 0.00594313582405448
# Batch 个数 50, 当前Batch Loss: 0.012339374050498009
# Batch 个数 100, 当前Batch Loss: 0.014445175416767597
# Batch 个数 150, 当前Batch Loss: 0.0028518247418105602
# Batch 个数 200, 当前Batch Loss: 0.03053869493305683
# Batch 个数 250, 当前Batch Loss: 0.0010048598051071167
# Batch 个数 300, 当前Batch Loss: 0.007153400685638189
# Batch 个数 350, 当前Batch Loss: 0.009448491036891937
# Epoch [90/100], Loss: 0.0114
# Batch 个数 0, 当前Batch Loss: 0.014992866665124893
# Batch 个数 50, 当前Batch Loss: 0.008313311263918877
# Batch 个数 100, 当前Batch Loss: 0.005880874115973711
# Batch 个数 150, 当前Batch Loss: 0.004873082973062992
# Batch 个数 200, 当前Batch Loss: 0.006175456568598747
# Batch 个数 250, 当前Batch Loss: 0.008494198322296143
# Batch 个数 300, 当前Batch Loss: 0.005879412870854139
# Batch 个数 350, 当前Batch Loss: 0.004300981294363737
# Epoch [91/100], Loss: 0.0110
# Batch 个数 0, 当前Batch Loss: 0.017486276105046272
# Batch 个数 50, 当前Batch Loss: 0.013181004673242569
# Batch 个数 100, 当前Batch Loss: 0.008458701893687248
# Batch 个数 150, 当前Batch Loss: 0.0069205788895487785
# Batch 个数 200, 当前Batch Loss: 0.006714135408401489
# Batch 个数 250, 当前Batch Loss: 0.0053213913924992085
# Batch 个数 300, 当前Batch Loss: 0.011805250309407711
# Batch 个数 350, 当前Batch Loss: 0.007324756123125553
# Epoch [92/100], Loss: 0.0105
# Batch 个数 0, 当前Batch Loss: 0.005766317248344421
# Batch 个数 50, 当前Batch Loss: 0.011682704091072083
# Batch 个数 100, 当前Batch Loss: 0.004314587451517582
# Batch 个数 150, 当前Batch Loss: 0.026730166748166084
# Batch 个数 200, 当前Batch Loss: 0.012297473847866058
# Batch 个数 250, 当前Batch Loss: 0.0170297808945179
# Batch 个数 300, 当前Batch Loss: 0.004340590909123421
# Batch 个数 350, 当前Batch Loss: 0.009730154648423195
# Epoch [93/100], Loss: 0.0101
# Batch 个数 0, 当前Batch Loss: 0.01679292321205139
# Batch 个数 50, 当前Batch Loss: 0.004411904141306877
# Batch 个数 100, 当前Batch Loss: 0.00666993111371994
# Batch 个数 150, 当前Batch Loss: 0.014193372800946236
# Batch 个数 200, 当前Batch Loss: 0.011732712388038635
# Batch 个数 250, 当前Batch Loss: 0.012618797831237316
# Batch 个数 300, 当前Batch Loss: 0.006849908269941807
# Batch 个数 350, 当前Batch Loss: 0.010692904703319073
# Epoch [94/100], Loss: 0.0097
# Batch 个数 0, 当前Batch Loss: 0.006746961735188961
# Batch 个数 50, 当前Batch Loss: 0.014214199967682362
# Batch 个数 100, 当前Batch Loss: 0.0058969613164663315
# Batch 个数 150, 当前Batch Loss: 0.004774064756929874
# Batch 个数 200, 当前Batch Loss: 0.01641843095421791
# Batch 个数 250, 当前Batch Loss: 0.006495635490864515
# Batch 个数 300, 当前Batch Loss: 0.009415543638169765
# Batch 个数 350, 当前Batch Loss: 0.008248352445662022
# Epoch [95/100], Loss: 0.0094
# Batch 个数 0, 当前Batch Loss: 0.009729652665555477
# Batch 个数 50, 当前Batch Loss: 0.004374057985842228
# Batch 个数 100, 当前Batch Loss: 0.02197312004864216
# Batch 个数 150, 当前Batch Loss: 0.02774765156209469
# Batch 个数 200, 当前Batch Loss: 0.017624057829380035
# Batch 个数 250, 当前Batch Loss: 0.001677674357779324
# Batch 个数 300, 当前Batch Loss: 0.007587771862745285
# Batch 个数 350, 当前Batch Loss: 0.002582262735813856
# Epoch [96/100], Loss: 0.0090
# Batch 个数 0, 当前Batch Loss: 0.00642466451972723
# Batch 个数 50, 当前Batch Loss: 0.005130593664944172
# Batch 个数 100, 当前Batch Loss: 0.003871894907206297
# Batch 个数 150, 当前Batch Loss: 0.016821064054965973
# Batch 个数 200, 当前Batch Loss: 0.008798981085419655
# Batch 个数 250, 当前Batch Loss: 0.01174863614141941
# Batch 个数 300, 当前Batch Loss: 0.010248837061226368
# Batch 个数 350, 当前Batch Loss: 0.0025188110303133726
# Epoch [97/100], Loss: 0.0087
# Batch 个数 0, 当前Batch Loss: 0.002705798950046301
# Batch 个数 50, 当前Batch Loss: 0.009980837814509869
# Batch 个数 100, 当前Batch Loss: 0.0008338618208654225
# Batch 个数 150, 当前Batch Loss: 0.005108801182359457
# Batch 个数 200, 当前Batch Loss: 0.0031177178025245667
# Batch 个数 250, 当前Batch Loss: 0.009322207421064377
# Batch 个数 300, 当前Batch Loss: 0.025123540312051773
# Batch 个数 350, 当前Batch Loss: 0.0019972058944404125
# Epoch [98/100], Loss: 0.0084
# Batch 个数 0, 当前Batch Loss: 0.008289442397654057
# Batch 个数 50, 当前Batch Loss: 0.014101161621510983
# Batch 个数 100, 当前Batch Loss: 0.01338329166173935
# Batch 个数 150, 当前Batch Loss: 0.00866643711924553
# Batch 个数 200, 当前Batch Loss: 0.005650359205901623
# Batch 个数 250, 当前Batch Loss: 0.005093724466860294
# Batch 个数 300, 当前Batch Loss: 0.009242543019354343
# Batch 个数 350, 当前Batch Loss: 0.010008756071329117
# Epoch [99/100], Loss: 0.0080
# Batch 个数 0, 当前Batch Loss: 0.00814784411340952
# Batch 个数 50, 当前Batch Loss: 0.009635833092033863
# Batch 个数 100, 当前Batch Loss: 0.002368793822824955
# Batch 个数 150, 当前Batch Loss: 0.009710336104035378
# Batch 个数 200, 当前Batch Loss: 0.00857623666524887
# Batch 个数 250, 当前Batch Loss: 0.0075587136670947075
# Batch 个数 300, 当前Batch Loss: 0.005398079752922058
# Batch 个数 350, 当前Batch Loss: 0.0041574519127607346
# Epoch [100/100], Loss: 0.0078
# 输入 '帮我导航到北京' 预测为: 'Travel-Query'
# 输入 '查询明天北京的天气' 预测为: 'Weather-Query'



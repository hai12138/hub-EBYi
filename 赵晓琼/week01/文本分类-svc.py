from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC # 导入SVC分类器
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import jieba

'''
文本分类方法一：基于SVC的文本分类
SVM(Support Vector Machine)支持向量机，是一个通用的机器学习算法框架
SVC(Support Vector Classification)支持向量机分类器，是SVM框架在分类任务上的具体实现
SVR(Support Vector Regression)支持向量机回归器，专门处理预测连续值的回归问题
当用SVM解决分类问题时，这个模型就是SVC
'''
# 搜集数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset[1].value_counts())

# 中文需要先进行jieba分词
# jieba分词：根据词表把所有可能的切分方式切出来，计算哪种切分方式总词频最高，词频事先根据分词后语料统计出来
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # dataset[0].apply对数据集中的每一个元素批量应用自定义函数

# 计算tf-idf tf-词频，某个词在某类别中出现的次数/该类别词总数 idf-log(语料库的文档总数/(包含该词的文档数+1))
# tf-idf越高，该词对于该领域重要程度越高
vectorizer = TfidfVectorizer( # 把非结构化的文本（字符串）转换成机器学习模型（如SVC、决策树）能处理的数值型特征矩阵
    max_features=2000, # 限制特征数，避免维度灾难
    norm='l2', # 归一化，对SVC至关重要
    analyzer='word'
)
X = vectorizer.fit_transform(input_sentence.values) # fit_transform先“学习”语料库的词汇表和IDF值(fit)再把文本转换成特征矩阵，预测新文本时只能用transform()必须复用训练时的词汇表和IDF值，否则特征维度会不一致
Y = dataset[1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y # stratify 保证标签分布均匀
)

# 建模
svc_model = SVC( # SVC的优化目标是：最大化间隔 同时 最小化分类错误
    kernel='linear',
    C=0.8, # 正则化参数：C越小正则化越强，防止过拟合
    class_weight='balanced',
    random_state=42,
    tol=1e-4,  # 降低收敛阈值，提升精度
    max_iter=10000  # 增加迭代次数
)
svc_model.fit(X_train, y_train)

# 模型评估
y_pred = svc_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型测试集准确率：{accuracy:.2f}\n")
print('详细分析报告')
print(classification_report(y_test, y_pred))

# 根据用户输入的内容预测是哪个类型
while True:
    text = input('请输入句子')
    input_sentence = " ".join(jieba.lcut(str(text)))
    # print(cut_input, '查看cut_input')
    new_X = vectorizer.transform([input_sentence])
    new_pred = svc_model.predict(new_X)
    print(f'分类结果:{new_pred}')
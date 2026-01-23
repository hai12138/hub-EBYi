import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

stop_words = [
    '的', '了', '在', '是', '我', '有', '和', '就',
    '不', '人', '都', '一', '一个', '上', '也', '很',
    '到', '说', '要', '去', '你', '会', '着', '没有',
    '好', '自己', '这', '那', '中', '为', '与',
    '对', '但', '而', '或', '且', '之', '这', '那',
    '啊', '呀', '呢', '吧', '吗', '啦', '哇', '哦',
    '嗯', '呃', '唉', '呀', '嘛', '咧', '呗'
]

# 带停用词过滤的分词函数
def cut_with_stopwords(text: str) -> str :
    # 使用jieba分词
    words = jieba.lcut(text)
    # 过滤停用词
    filtered_words = [word for word in words if word not in stop_words]
    # if(len(filtered_words)==0):
    #     print(f"{words} 过滤后为空")
    return " ".join(filtered_words)

# 1. 数据加载
dataFrame = pd.read_table("dataset.csv", header=None, nrows=20000)
dataFrame.columns = ['text', 'label']
# 2. 中文分词
print("正在进行中文分词...")
dataFrame['cut_text'] = dataFrame['text'].apply(lambda x: cut_with_stopwords(x))
# print(dataFrame['cut_text'])
# 3. 特征提取
print("正在提取文本特征...")
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(dataFrame['cut_text'].values)
labels = dataFrame['label'].values
# 4. 模型训练
print("正在训练KNN模型...")
model = KNeighborsClassifier(
    # metric='cosine'  # 对文本数据，余弦距离通常效果更好
)
model.fit(features, labels)

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 5. 使用模型进行预测
def predictML(text):
    """预测单个文本"""
    # 分词
    cut_text = cut_with_stopwords(str(text))
    text_features = vectorizer.transform([cut_text])
    # 预测
    return model.predict(text_features)[0]
#
def predictLLM(text):
    response = client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": f"""给定以下标签：
                FilmTele-Play            
                Video-Play               
                Music-Play              
                Radio-Listen           
                Alarm-Update        
                Travel-Query        
                HomeAppliance-Control  
                Weather-Query          
                Calendar-Query      
                TVProgram-Play      
                Audio-Play       
                Other
                帮我给"{text}"分个类，只给出标签，不要有多余的输出
                """}],
    )
    return response.choices[0].message.content

#
# 示例：预测新文本
test_text = "今天是什么天气啊"
ML_label = predictML(test_text)
LLM_label = predictLLM(test_text)
print(f"文本: {test_text}")
print(f"ML预测标签: {ML_label}")
print(f"LLM预测标签: {LLM_label}")

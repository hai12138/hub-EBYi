import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

# 加载数据集
DATASET_PATH = "dataset.csv"
try:
    df = pd.read_csv(
        DATASET_PATH,
        sep="\t",
        header=None
    )
except FileNotFoundError:
    print(f"未找到数据集文件 {DATASET_PATH}")
    df = None


# 使用jieba分词，并用空格连接
def preprocess_chinese_text(text_series):
    """
    中文文本预处理函数：对文本序列进行分词并格式转换
    :param text_series: pandas Series，包含待处理的中文文本
    :return: 处理后的文本列表（每个文本已分词并空格连接）
    """
    return text_series.apply(lambda x: " ".join(jieba.lcut(x))).tolist()


# 执行文本预处理
processed_texts = None
text_features = None
knn_model = KNeighborsClassifier()
labels = None

if df is not None and not df.empty:
    processed_texts = preprocess_chinese_text(df[0])

    # 文本特征提取：使用CountVectorizer
    count_vectorizer = CountVectorizer()
    # 拟合词表并转换为特征矩阵
    text_features = count_vectorizer.fit_transform(processed_texts)

    # 训练Knn分类模型
    # 提取标签列并完成模型训练
    labels = df[1].values
    knn_model.fit(text_features, labels)


# ---------------------- 文本分类功能函数 ----------------------
def text_classify_ml(text: str) -> str:
    """
    基于机器学习（K近邻+词袋模型）的文本分类
    :param text: 待分类的中文文本
    :return: 分类结果标签
    """
    if df is None or count_vectorizer is None:
        return "机器学习分类不可用：数据集未加载或预处理失败"

    # 对输入文本进行相同的预处理
    processed_text = " ".join(jieba.lcut(text))
    # 转换为模型可识别的特征格式
    text_feature = count_vectorizer.transform([processed_text])
    # 模型预测并返回结果
    return knn_model.predict(text_feature)[0]


def text_classify_llm(text: str) -> str:
    """
    基于GLM的文本分类
    :param text: 待分类的中文文本
    :return: 分类结果标签
    """
    # 修正1：配置正确的base_url（仅保留到v4版本前缀，不追加重复路径）
    client = OpenAI(
        api_key="fcc4370e59464245bc75ba9c65661c7e.MdUUgRkGPvwV7SZU",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    # 提示词
    system_prompt = """你需要完成中文文本分类任务，输出结果必须严格从以下类别中选择，且仅输出类别名称，不添加任何额外内容：
    Video-Play、FilmTele-Play、Music-Play、Radio-Listen、Alarm-Update、
    Travel-Query、HomeAppliance-Control、Weather-Query、Calendar-Query、
    TVProgram-Play、Audio-Play、Other
    """

    user_prompt = f"请对以下文本进行分类：{text}"

    # 调用大模型完成分类
    try:
        completion = client.chat.completions.create(
            model="GLM-4.5-Flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        # 提取并返回分类结果
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM分类失败：{str(e)}"


# ---------------------- 测试运行 ----------------------
if __name__ == '__main__':

    test_text = "帮我看下明天的天气"

    ml_result = text_classify_ml(test_text)
    llm_result = text_classify_llm(test_text)

    print(f"机器学习（K近邻）分类结果：{ml_result}")
    print(f"大语言模型（GLM）分类结果：{llm_result}")


    '''
    作业二————文本分类：
        输出：
        机器学习（K近邻）分类结果：Weather-Query
        大语言模型（GLM）分类结果：Weather-Query
    
    作业一————环境搭建：
        (nlp_learn) guozhijia@guozhijiadeMacBook-Pro Week01 % pip list
        Package            Version
        ------------------ -----------
        accelerate         1.12.0
        annotated-doc      0.0.4
        annotated-types    0.7.0
        anyio              4.12.0
        certifi            2025.11.12
        charset-normalizer 3.4.4
        contourpy          1.3.3
        cycler             0.12.1
        distro             1.9.0
        fastapi            0.124.4
        filelock           3.20.0
        fonttools          4.61.1
        fsspec             2025.12.0
        h11                0.16.0
        hf-xet             1.2.0
        httpcore           1.0.9
        httpx              0.28.1
        huggingface-hub    0.36.0
        idna               3.11
        jieba              0.42.1
        Jinja2             3.1.6
        jiter              0.12.0
        joblib             1.5.2
        kiwisolver         1.4.9
        librt              0.7.3
        MarkupSafe         3.0.3
        matplotlib         3.9.2
        mpmath             1.3.0
        mypy               1.19.0
        mypy_extensions    1.1.0
        networkx           3.6.1
        numpy              1.26.4
        openai             2.15.0
        packaging          25.0
        pandas             2.2.2
        pathspec           0.12.1
        peft               0.15.0
        pillow             12.0.0
        pip                25.3
        psutil             7.1.3
        pydantic           2.12.5
        pydantic_core      2.41.5
        pyparsing          3.2.5
        python-dateutil    2.9.0.post0
        pytz               2025.2
        PyYAML             6.0.3
        regex              2025.11.3
        requests           2.32.5
        safetensors        0.7.0
        scikit-learn       1.5.1
        scipy              1.16.3
        setuptools         80.9.0
        six                1.17.0
        sniffio            1.3.1
        starlette          0.50.0
        sympy              1.13.1
        threadpoolctl      3.6.0
        tokenizers         0.22.1
        torch              2.6.0
        tqdm               4.67.1
        transformers       4.56.2
        typing_extensions  4.15.0
        typing-inspection  0.4.2
        tzdata             2025.2
        urllib3            2.6.2
        wheel              0.45.1

    '''

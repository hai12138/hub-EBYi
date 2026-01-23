
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from openai import OpenAI

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 进行转换 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-6df0270104f64e359c37561bc3980c3c", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
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
"""},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    print("=" * 50)
    print("          文本分类交互式程序")
    print("=" * 50)
    print("输入说明：")
    print("1. 输入需要分类的文本，按回车即可同时调用两个模型预测")
    print("2. 输入 'exit' 或 'quit' 可退出程序")
    print("-" * 50)

    while True:
        # 获取用户输入（去除首尾空格）
        user_input = input("请输入待分类的文本：").strip()

        # 退出条件
        if user_input.lower() in ["exit", "quit"]:
            print("程序已退出，感谢使用！")
            break

        # 空输入处理
        if not user_input:
            print("错误：输入不能为空，请重新输入！")
            continue

        # 调用模型并输出结果
        print("\n【预测结果】")
        ml_result = text_calssify_using_ml(user_input)
        llm_result = text_calssify_using_llm(user_input)
        print(f"机器学习模型(KNN)：{ml_result}")
        print(f"大语言模型(LLM)：{llm_result}")
        print("-" * 50)

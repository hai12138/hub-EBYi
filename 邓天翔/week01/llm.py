from openai import OpenAI

client = OpenAI(
    api_key="sk-e3f7921e1aac4b48a206384ac86f9cdc",
    # 模型地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_classify(sentence: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    :param sentence: 输入文本
    :return: 文本类型
    """
    result = client.chat.completions.create(
        # 模型代号
        model="qwen-flash",
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{sentence}
输出的类型只能为以下类型，除了以下类型，不要输出其他内容。
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

"""},
        ]
    )
    return result.choices[0].message.content

if __name__ == '__main__':
    while True:
        sentence = input("请输入要判定的语句：")
        print(text_classify(sentence))

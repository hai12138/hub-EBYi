import llm_text_classify as llm_helper
import ml_text_classify as ml_helper

if __name__ == '__main__':
    text = "帮我导航到天安门"
    print("机器学习", ml_helper.text_classify_use_ml(text))
    print("大模型", llm_helper.text_calssify_use_llm(text))

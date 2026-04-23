from agents import Agent, Runner
from transformers import pipeline
import spacy


# Hugging Face 情感分析模型
sentiment_analysis = pipeline("sentiment-analysis")

# spaCy 英文实体识别模型
nlp = spacy.load("en_core_web_sm")


def sentiment_agent(request):
    text = request["text"]
    result = sentiment_analysis(text)
    return {
        "task": "sentiment",
        "text": text,
        "sentiment": result[0]["label"],
        "score": float(result[0]["score"])
    }


def entity_recognition_agent(request):
    text = request["text"]
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return {
        "task": "entity_recognition",
        "text": text,
        "entities": entities
    }


class MainAgent:
    def __init__(self):
        self.sentiment_agent = sentiment_agent
        self.entity_recognition_agent = entity_recognition_agent

    def respond(self, request):
        task_type = request.get("task_type", "").strip().lower()

        if task_type == "sentiment":
            return self.sentiment_agent(request)
        elif task_type == "entity_recognition":
            return self.entity_recognition_agent(request)
        else:
            return {
                "error": 'Unknown task type. Please choose "sentiment" or "entity_recognition".'
            }

def main():
    agent = MainAgent()

    print("Welcome! What would you like to do?")
    print("1. Sentiment Analysis (type: sentiment)")
    print("2. Entity Recognition (type: entity_recognition)")

    task_type = input("Please enter your choice: ").strip()
    user_input = input("Please enter your text: ").strip()

    request = {
        "text": user_input,
        "task_type": task_type
    }

    result = agent.respond(request)
    print("Result:")
    print(result)


if __name__ == "__main__":
    main()

import os

# 强制离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset
import warnings

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class EmotionClassifier:
    def __init__(self, model_path='D:/models/google-bert/bert-base-chinese'):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.data_collator = None

    def load_data(self, file_path='simple.xlsx'):
        """加载数据"""
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            print(f"数据集加载成功，共有 {len(df)} 条数据")

            texts = df['text'].astype(str).tolist()
            labels = df['label'].tolist()

            encoded_labels = self.label_encoder.fit_transform(labels)
            self.num_classes = len(self.label_encoder.classes_)

            print(f"情感类别: {list(self.label_encoder.classes_)}")
            print(f"类别数量: {self.num_classes}")
            print("标签分布:")
            print(pd.Series(labels).value_counts())

            return texts, encoded_labels

        except Exception as e:
            print(f"数据加载错误: {e}")
            return [], []

    def setup_model(self):
        """设置模型"""
        print("从本地加载模型...")

        try:
            # 检查模型路径
            import os
            if not os.path.exists(self.model_path):
                print(f"错误: 模型路径不存在: {self.model_path}")
                return False

            print("模型目录内容:")
            for file in os.listdir(self.model_path):
                print(f"  {file}")

            # 加载分词器和模型
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=self.num_classes,
                local_files_only=True
            )
            self.model.to(device)

            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            print("✓ 模型加载成功!")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def prepare_data(self, texts, labels, test_size=0.2):
        """准备训练数据"""
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        print(f"\n数据分割结果:")
        print(f"训练集: {len(train_texts)} 条")
        print(f"测试集: {len(test_texts)} 条")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=128
            )

        train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
        test_dataset = Dataset.from_dict({'text': test_texts, 'labels': test_labels})

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_test = test_dataset.map(tokenize_function, batched=True)

        return tokenized_train, tokenized_test

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    def train(self, train_dataset, test_dataset, output_dir='./results'):
        """训练模型"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        print("开始训练...")
        train_result = trainer.train()

        # 保存模型
        trainer.save_model(f'{output_dir}/best_model')
        self.tokenizer.save_pretrained(f'{output_dir}/best_model')

        # 保存标签编码器
        import joblib
        joblib.dump(self.label_encoder, f'{output_dir}/best_model/label_encoder.pkl')

        return trainer, train_result

    def evaluate_model(self, trainer, test_dataset):
        """评估模型性能"""
        print("\n" + "=" * 50)
        print("模型评估结果")
        print("=" * 50)

        eval_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"测试集准确率: {eval_results['eval_accuracy']:.4f}")

        # 详细预测分析
        test_predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(test_predictions.predictions, axis=1)
        true_labels = test_predictions.label_ids

        print("\n详细分类报告:")
        print(classification_report(true_labels, pred_labels,
                                    target_names=self.label_encoder.classes_))

        return eval_results

    def predict_single(self, text):
        """预测单条文本的情感"""
        self.model.eval()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class_id = probabilities.argmax().item()
            confidence = probabilities[0][predicted_class_id].item()

        predicted_label = self.label_encoder.inverse_transform([predicted_class_id])[0]

        return predicted_label, confidence

    def batch_predict(self, texts):
        """批量预测文本情感"""
        results = []
        for text in texts:
            emotion, confidence = self.predict_single(text)
            results.append({
                'text': text,
                'predicted_emotion': emotion,
                'confidence': confidence
            })
        return results

    def test_predictions(self, test_cases):
        """测试预测功能"""
        print("\n" + "=" * 60)
        print("模型预测效果测试")
        print("=" * 60)

        results = self.batch_predict(test_cases)

        print("预测结果:")
        print("-" * 80)

        for i, result in enumerate(results, 1):
            print(f"{i:2d}. 文本: {result['text']}")
            print(f"    预测情感: {result['predicted_emotion']} (置信度: {result['confidence']:.4f})")
            print("-" * 80)

        # 保存预测结果
        results_df = pd.DataFrame(results)
        results_df.to_csv('prediction_test_results.csv', index=False, encoding='utf-8')
        print(f"\n预测结果已保存到 'prediction_test_results.csv'")

        return results


def main():
    # 初始化情感分类器
    classifier = EmotionClassifier(model_path='D:/models/google-bert/bert-base-chinese')

    # 加载数据
    texts, labels = classifier.load_data('simple.xlsx')
    if not texts:
        print("数据加载失败，程序退出")
        return

    # 设置模型
    if not classifier.setup_model():
        print("模型设置失败，程序退出")
        return

    # 准备数据
    train_dataset, test_dataset = classifier.prepare_data(texts, labels)

    # 训练模型
    trainer, train_result = classifier.train(train_dataset, test_dataset)

    # 评估模型
    classifier.evaluate_model(trainer, test_dataset)

    # 测试预测功能
    test_cases = [
        "我今天特别开心，考试得了满分！真是太棒了！",
        "真的很生气，他们怎么能这样对待我！完全不能接受！",
        "这个消息让我非常惊讶，完全没想到会这样发展。",
        "心情很平静，一切都很好，没有什么特别的事情。",
        "我觉得有点不舒服，需要关心一下我的健康状况。",
        "这个电影太让人伤心了，我看哭了整整一个小时。",
        "我对这种行为感到非常厌恶！简直无法忍受！",
        "这是怎么回事？我有点疑问，需要进一步了解。",
        "今天的天气真好啊，心情特别愉快！",
        "这个决定让我感到非常失望，完全不符合预期。",
        "哇！这个惊喜太让人兴奋了！",
        "我对这个结果感到很满意，一切都很顺利。",
        "这个问题让我很困惑，不知道该怎么解决。",
        "听到这个消息我真的很伤心，需要时间平复心情。",
        "这种态度让我很生气，完全不负责任！"
    ]

    # 运行预测测试
    prediction_results = classifier.test_predictions(test_cases)

    # 分析预测结果
    print("\n" + "=" * 60)
    print("预测结果统计分析")
    print("=" * 60)

    emotions = [result['predicted_emotion'] for result in prediction_results]
    confidences = [result['confidence'] for result in prediction_results]

    emotion_counts = pd.Series(emotions).value_counts()
    print("预测情感分布:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}次")

    print(f"\n平均置信度: {np.mean(confidences):.4f}")
    print(f"置信度标准差: {np.std(confidences):.4f}")
    print(f"最高置信度: {max(confidences):.4f}")
    print(f"最低置信度: {min(confidences):.4f}")

    # 高置信度预测示例
    high_confidence = [r for r in prediction_results if r['confidence'] > 0.9]
    if high_confidence:
        print(f"\n高置信度预测 (>{0.9}):")
        for result in high_confidence:
            print(f"  {result['text'][:30]}... -> {result['predicted_emotion']} ({result['confidence']:.4f})")


if __name__ == "__main__":
    main()

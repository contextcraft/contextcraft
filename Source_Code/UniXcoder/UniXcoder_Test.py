import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class UniXcoder_Test:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')

    def predict(self, functional_description):
        inputs = self.tokenizer(functional_description, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.item()

    def evaluate(self, test_df):
        test_df['Predicted Method Name'] = test_df['Functional Description'].apply(self.predict)
        return test_df

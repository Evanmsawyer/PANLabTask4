import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import glob
import json
from sklearn.metrics import confusion_matrix, classification_report

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, ids, tokenizer: AutoTokenizer):
        self.texts = texts
        self.labels = labels
        self.ids = ids  
        self.encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=512)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['text'] = self.texts[idx]
        item['id'] = self.ids[idx]  
        return item

    def __len__(self):
        return len(self.labels)


def load_data(human_path, machine_path):
    with open(human_path, 'r', encoding='utf-8') as file:
        human_data = [json.loads(line) for line in file]
    
    machine_data = []
    for path in glob.glob(f'{machine_path}/*.jsonl'):
        with open(path, 'r', encoding='utf-8') as file:
            machine_data.extend([json.loads(line) for line in file])
    
    texts = [item['text'] for item in human_data + machine_data]
    ids = [item['id'] for item in human_data + machine_data]  # Extract IDs
    labels = [0] * len(human_data) + [1] * len(machine_data)  # 0 for human, 1 for AI
    
    return texts, labels, ids


def evaluate_model(model, data_loader):
    model.eval()
    misclassified_ids = [] 
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            inputs = {key: val.to(model.device) for key, val in batch.items() if key != 'labels' and key != 'text' and key != 'id'}
            labels = batch['labels'].to(model.device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

            # Collect misclassified IDs
            for i, (pred, label) in enumerate(zip(preds, labels.cpu().numpy())):
                if pred != label:
                    misclassified_ids.append(batch['id'][i])  

    return predictions, true_labels, misclassified_ids


def main():
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('./roberta_finetuned', num_labels=2)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    human_path = './data/testing/human.jsonl'
    machine_path = './data/testing/machines'
    texts, labels, ids = load_data(human_path, machine_path)
    
    test_dataset = CustomDataset(texts, labels, ids, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    predictions, true_labels, misclassified_ids = evaluate_model(model, test_loader)
    f1 = f1_score(true_labels, predictions)
    print(f'F1 Score: {f1}')

    cm = confusion_matrix(true_labels, predictions)
    cr = classification_report(true_labels, predictions)

    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", cr)
    
    print("\nMisclassified Text IDs:")
    for text_id in misclassified_ids:
        print(text_id)

if __name__ == '__main__':
    main()


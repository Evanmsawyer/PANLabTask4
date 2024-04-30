import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import evaluate
import json
import glob
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer: AutoTokenizer):
        self.encodings = tokenizer(
            dataframe['text'].tolist(), 
            truncation=True, 
            padding=True, 
            return_tensors="pt",
            max_length=512
        )
        self.labels = dataframe['label'].apply(lambda x: 1 if x == "AI" else 0).tolist()

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)

def read_jsonl_files(base_path):
    df_human = pd.read_json(os.path.join(base_path, 'training/human.jsonl'), lines=True)
    df_human.drop(columns=[df_human.columns[0]], inplace=True) 
    df_human['label'] = 'Human'
    
    machine_files = glob.glob(os.path.join(base_path, 'training/machines/*.jsonl'))
    df_machine = pd.concat([pd.read_json(f, lines=True) for f in machine_files], ignore_index=True)
    df_machine.drop(columns=[df_machine.columns[0]], inplace=True) 
    df_machine['label'] = 'AI'
    
    df_train = pd.concat([df_human, df_machine], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_eval = pd.read_json(os.path.join(base_path, 'eval/human.jsonl'), lines=True)
    df_eval.drop(columns=[df_eval.columns[0]], inplace=True) 
    df_eval['label'] = 'Human'
    
    eval_machine_files = glob.glob(os.path.join(base_path, 'eval/machines/*.jsonl'))
    df_eval_machine = pd.concat([pd.read_json(f, lines=True) for f in eval_machine_files], ignore_index=True)
    df_eval_machine.drop(columns=[df_eval_machine.columns[0]], inplace=True)
    df_eval_machine['label'] = 'AI'

    df_eval = pd.concat([df_eval, df_eval_machine], ignore_index=True)

    return df_train, df_eval


def compute_metrics(eval_pred):
    metric = evaluate.load('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def finetune_model(data_dict: dict, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=data_dict['name'],
        learning_rate=1e-5,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="no"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_dict['train'],
        eval_dataset=data_dict['eval'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model()

def main():
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    train_df, eval_df = read_jsonl_files('./data')
    
    print(train_df)
    data_dict = {
        'name': 'roberta_finetuned',
        'train': CustomDataset(train_df, tokenizer), 
        'eval': CustomDataset(eval_df, tokenizer)
    }
    finetune_model(data_dict, tokenizer)

if __name__ == "__main__":
    main()

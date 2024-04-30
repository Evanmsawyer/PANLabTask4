import transformers
import torch
import pandas as pd
import os
import glob

model_id = "mistralai/Mistral-7B-v0.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformers.AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
pipe = transformers.pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device=device, torch_dtype=torch.bfloat16
)

def read_jsonl_files(base_path):
    df_human = pd.read_json(os.path.join(base_path, 'training/human.jsonl'), lines=True)
    df_human['label'] = 'Human'
    
    df_machine = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(os.path.join(base_path, 'training/machines/*.jsonl'))], ignore_index=True)
    df_machine['label'] = 'AI'
    
    df_train = pd.concat([df_human, df_machine], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_eval = pd.read_json(os.path.join(base_path, 'eval/human.jsonl'), lines=True)
    df_eval['label'] = 'Human'
    
    df_eval_machine = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(os.path.join(base_path, 'eval/machines/*.jsonl'))], ignore_index=True)
    df_eval_machine['label'] = 'AI'

    df_eval = pd.concat([df_eval, df_eval_machine], ignore_index=True)

    return df_train, df_eval

def classify_text(text):
    prompt = f"""
    Role: AI Evaluator
    Task: Determine if the following text was generated by an AI or written by a human.
    Criteria:
      1. Analyze linguistic patterns and typical errors made by AI.
      2. Consider the complexity and depth of emotion in the text.
      3. Evaluate the logical coherence and flow of ideas.
    Text: "{text}"
    Evaluation: [Human/AI]
    """
    output = pipe(prompt, max_new_tokens=50, top_k=50, top_p=0.95, num_return_sequences=1, do_sample=True, temperature=0.7)
    response = output[0]['generated_text'].strip()
    return 'AI' if 'AI' in response else 'Human'

def evaluate_model(df):
    df['predicted_label'] = df['text'].apply(classify_text)
    accuracy = (df['predicted_label'] == df['label']).mean()
    return accuracy

def main():
    train_df, eval_df = read_jsonl_files('./data') 
    
    accuracy = evaluate_model(eval_df)
    print(f"Accuracy of the Mistral model as a classifier: {accuracy}")

if __name__ == "__main__":
    main()

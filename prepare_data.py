import json
import os
import argparse
import numpy as np
import yaml
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

def load_local_tokenizer(tokenizer_dir):
    try:
        json_path = os.path.join(tokenizer_dir, "tokenizer.json")
        if os.path.exists(json_path):
             return PreTrainedTokenizerFast(
                tokenizer_file=json_path,
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]"
            )
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_dir}: {e}")

def prepare_data(args):
    data_dir = getattr(args, "data_dir", "data")
    tokenizer_dir = "tokenizer" # Assumed local based on user's setup
    
    # 1. Load Tokenizer
    print("Loading custom tokenizer...")
    if not os.path.exists(tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer directory not found at {os.path.abspath(tokenizer_dir)}. Please ensure 'tokenizer/' is in the bobgpt folder.")
    
    tokenizer = load_local_tokenizer(tokenizer_dir)
    print(f"✓ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # 2. Check Input Data
    input_file = os.path.join(data_dir, "cosmopedia.jsonl")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}. Please ensure 'cosmopedia.jsonl' is in the data folder.")

    # 3. Process Data
    print(f"Processing {input_file}...")
    
    tokenized_samples = []
    max_len = 512
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Just count lines first for progress bar
        print("Counting lines...")
        lines = f.readlines()
        
    print(f"Found {len(lines)} lines. Tokenizing & Padding...")
    
    for line in tqdm(lines):
        try:
            entry = json.loads(line)
            text = entry.get("text", "")
            if not text:
                continue
            
            # The cosmopedia.py script already ensured len <= 512, but we double check/re-tokenize 
            # effectively just to map to IDs and pad.
            encoding = tokenizer(
                text, 
                truncation=True, 
                max_length=max_len, 
                padding="max_length", 
                add_special_tokens=True,
                return_attention_mask=False
            )
            
            input_ids = encoding["input_ids"]
            tokenized_samples.append(input_ids)
            
        except json.JSONDecodeError:
            continue
            
    if not tokenized_samples:
        raise ValueError("No valid samples found in cosmopedia.jsonl!")
        
    # 4. Save to NPY using uint16 (vocab 35000 fits in uint16, saves RAM)
    # Actually vocab 35000 fits in uint16 (max 65535).
    all_data = np.array(tokenized_samples, dtype=np.uint16)
    print(f"Total samples: {len(all_data)}")
    
    # Shuffle
    np.random.shuffle(all_data)
    
    # Split 90/10
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    valid_data = all_data[split_idx:]
    
    train_path = os.path.join(data_dir, "train.npy")
    valid_path = os.path.join(data_dir, "valid.npy")
    
    np.save(train_path, train_data)
    np.save(valid_path, valid_data)
    
    print(f"✓ Saved {len(train_data)} training samples to {train_path}")
    print(f"✓ Saved {len(valid_data)} validation samples to {valid_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    prepare_data(args)

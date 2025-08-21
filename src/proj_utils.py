#from transformers import AutoTokenizer
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from collections import Counter

#MODEL_NAME = "bert-base-uncased"
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_text(text):
    #return tokenizer.tokenize(text)
    words = text.split()
    return words

def process_chunk(chunk):
    X_chunk = []
    y_chunk = []
    for tokens in chunk:
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                X_chunk.append(tokens[:i+1])
                y_chunk.append(tokens[i+1])
    return X_chunk, y_chunk

def create_training_examples(tokenized_texts):
    num_processes = cpu_count()
    
    print(f"Используется {num_processes} процессов для создания примеров...")
    
    chunk_size = max(1, len(tokenized_texts) // num_processes)
    chunks = [tokenized_texts[i:i + chunk_size] for i in range(0, len(tokenized_texts), chunk_size)]
    
    with Pool(num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    X = []
    y = []
    for X_chunk, y_chunk in results:
        X.extend(X_chunk)
        y.extend(y_chunk)
    
    return X, y

def prepare_datasets(X, y, test_size=0.1, val_size=0.1, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state, shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_vocabulary(token_lists, min_freq=1):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {'<PAD>':0, '<UNK>':1}

    idx = 2
    for token, count in counter.items():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1

    return vocab

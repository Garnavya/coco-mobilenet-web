import json
import re
from collections import Counter
from pathlib import Path
from pycocotools.coco import COCO

from config_caption import config_cap as config

def tokenize(text):
    """Standardizes text by lowercasing and stripping punctuation."""
    return re.findall(r'\w+', text.lower())

def build_vocab():
    print(f"[INFO] Loading COCO Captions from: {config.TRAIN_CAPTIONS_FILE}")
    coco = COCO(str(config.TRAIN_CAPTIONS_FILE))
    
    counter = Counter()
    anns = coco.dataset['annotations']
    total_captions = len(anns)
    
    print(f"[INFO] Tokenizing {total_captions} captions...")
    for i, ann in enumerate(anns):
        if i % 10000 == 0 or i == total_captions - 1:
            print(f"\rProcessing caption {i+1}/{total_captions}", end="", flush=True)
            
        tokens = tokenize(ann['caption'])
        counter.update(tokens)
        
    print("\n")
    
    # 1. Discard rare words (typos, highly obscure references)
    words = [word for word, count in counter.items() if count >= config.VOCAB_FREQ_THRESHOLD]
    print(f"[INFO] Total unique words: {len(counter)}")
    print(f"[INFO] Words kept (frequency >= {config.VOCAB_FREQ_THRESHOLD}): {len(words)}")
    
    # 2. Add the four mandatory NLP special tokens
    word2idx = {
        "<PAD>": 0,    # Used to pad short captions to exactly 20 words
        "<START>": 1,  # The input token that tells the LSTM to start generating
        "<END>": 2,    # The output token where the LSTM decides the sentence is over
        "<UNK>": 3     # Replaces any word not in our dictionary
    }
    
    # 3. Assign integers to the remaining words
    for idx, word in enumerate(words, start=4):
        word2idx[word] = idx
        
    # Create the reverse mapping (for human-readable output)
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    vocab = {
        "word2idx": word2idx,
        "idx2word": idx2word
    }
    
    vocab_path = config.BASE_DIR / "vocab.json"
    print(f"[INFO] Saving dictionary to: {vocab_path}")
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
        
    print(f"[SUCCESS] Vocabulary built successfully! Dictionary size: {len(word2idx)} tokens.")

if __name__ == "__main__":
    build_vocab()
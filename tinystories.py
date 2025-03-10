import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset # pip install datasets
from tqdm import tqdm 
from transformers import AutoTokenizer

data_dir_name = "tinystories"
shard_size = int(1e8) # 100M tokens per shard

# download the dataset
data_dir = os.path.join(os.path.dirname(__file__), data_dir_name)
os.makedirs(data_dir, exist_ok=True)
ts = load_dataset("roneneldan/TinyStories", cache_dir=data_dir)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
def tokenize(doc):
  # tokenizes a single document and returns a numpy array of uint16 tokens
  tokens = [tokenizer.eos_token_id]  # use the tokenizer's EOS token ID
  tokens.extend(tokenizer(doc["text"])["input_ids"])
  tokens_np = np.array(tokens)
  assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
  return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
  np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
n_procs = max(1, os.cpu_count()//2)
for split in ["train", "validation"]:
  with mp.Pool(n_procs) as pool:
    # Reset for each split
    token_ctr    = 0
    shard_index  = 0
    progress_bar = None
    # preallocate buffer to hold current shard
    shard_tokens = np.empty((shard_size,), dtype=np.uint16)
    for tokens in pool.imap(tokenize, ts[split], chunksize=16):
      # is there enough space in the current shard for the new tokens?
      if token_ctr + len(tokens) < shard_size:
        # append tokens to current shard
        shard_tokens[token_ctr:token_ctr+len(tokens)] = tokens
        token_ctr += len(tokens)
        # update progress bar
        if progress_bar is None:
          progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
      else:
        # write the current shard and start a new one
        filename = os.path.join(data_dir, f"tinystories_{split}_{shard_index:06d}")
        # split the document into whatever fits in this shard; the remainder goes to next one
        remainder = shard_size - token_ctr
        progress_bar.update(remainder)
        shard_tokens[token_ctr:token_ctr+remainder] = tokens[:remainder]
        write_datafile(filename, shard_tokens)
        shard_index += 1
        progress_bar = None
        # populate the next shard with the leftovers of the current doc
        shard_tokens[0:len(tokens)-remainder] = tokens[remainder:]
        token_ctr = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_ctr != 0:
      filename = os.path.join(data_dir, f"tinystories_{split}_{shard_index:06d}")
      write_datafile(filename, shard_tokens[:token_ctr])
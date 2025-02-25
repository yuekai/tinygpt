################
# data loading #
################

import numpy as np
import os
import random

import torch

def load_npy(filename):
  npa = np.load(filename)
  ptt = torch.tensor(npa.astype(np.int32), dtype=torch.long) # torch doesn't like int16, so convert to int32 before passing to torch
  return ptt

class TinyStoriesDataLoader:
  
  def __init__(self, B, T, rank, world_size, split):
    self.B = B
    self.T = T
    self.rank = rank
    self.world_size = world_size
    assert split in {"train", "val"}

    data_dir = "tinystories"
    shards = os.listdir(data_dir)
    shards = [os.path.join(data_dir,s) for s in shards if split in s]
    random.shuffle(shards)
    self.shards = shards
    if MAIN_PROC_FLAG:
      print(f"loading {len(shards)} shards in {split} split")
    self.reset()

  def reset(self):

    self.shard_idx = 0
    self.toks = load_npy(self.shards[self.shard_idx])
    self.pos = self.B * self.T * self.rank

  def get_batch(self):
    B, T = self.B, self.T
    bffr = self.toks[self.pos : self.pos + B * T + 1]
    x = bffr[:-1].view(B,T)
    y = bffr[1:].view(B,T)
    self.pos += B * T * self.world_size
    if (self.pos + B * T * self.world_size + 1) > len(self.toks):
      self.shard_idx = (self.shard_idx + 1) % len(self.shards) 
      self.toks = load_npy(self.shards[self.shard_idx])
      self.pos = B * T * self.rank
    return x, y


########
# main #
#######################################################
# torchrun --standalone --nproc_per_node=4 tinygpt.py #
#######################################################

from dataclasses import dataclass
import math
import time

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from torchinfo import summary
from gpt import GPT

# set up training environment
DDP_FLAG = (int(os.environ.get('RANK',-1)) != -1)
if DDP_FLAG:
  assert torch.cuda.is_available()
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  device = f"cuda:{ddp_local_rank}"
  torch.cuda.set_device(device)
  MAIN_PROC_FLAG = (ddp_rank == 0)
  print(f"setting up GPU {ddp_rank+1} of {ddp_world_size}")
else:
  ddp_rank = 0
  ddp_local_rank = ddp_rank
  ddp_world_size = 1
  MAIN_PROC_FLAG = True
  # autodetect GPU
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
  print(f"using {device}")

# model configuration
@dataclass
class GPTConfig:
  block_size: int = 512
  vocab_size: int = 50257
  n_layer: int = 12
  n_head: int = 4
  kv_heads: int = 2
  n_embd: int = 512

# training hyperparameters
batch_toks = 524288
B, T = 64, 512
weight_decay, max_lr, betas = 1e-2, 1e-3, (0.9, 0.95)
device_type = "cuda" if device.startswith("cuda") else "cpu"

# set random seeds for reproducibility
torch.manual_seed(8888)
if torch.cuda.is_available():
  torch.cuda.manual_seed(8888)
  torch.set_float32_matmul_precision("high")

# gradient accumulation
assert batch_toks % (B * T * ddp_world_size) == 0
grad_accum_steps = batch_toks // (B * T * ddp_world_size)
if MAIN_PROC_FLAG:
  print(f"{grad_accum_steps:4d} grad accum steps to simulate desired batch size ({batch_toks:4d})")

# data loaders
train_loader = TinyStoriesDataLoader(B=B, T=T, rank=ddp_rank, world_size=ddp_world_size, split="train")
val_loader = TinyStoriesDataLoader(B=B, T=T, rank=ddp_rank, world_size=ddp_world_size, split="val")

# instantiate model
model = GPT(GPTConfig(vocab_size=50304 ))
if MAIN_PROC_FLAG:
  summary(model)
model.to(device)
model = torch.compile(model)
if DDP_FLAG:
  model = DDP(model, device_ids=[ddp_local_rank])
base_model = model.module if DDP_FLAG else model

# instantiate optimizer
min_lr = max_lr / 10.
max_itr = 9500
warmup_itr = max_itr // 100
max_decay_itrs = max_itr
def get_lr(itr):
  if itr < warmup_itr:
    lr = max_lr * (itr+1) / warmup_itr
  elif itr > max_decay_itrs:
    lr = min_lr
  else:  # cosine decay for remaining 90%
    decay_frac = (itr - warmup_itr) / (max_itr - warmup_itr)
    lr = min_lr + (max_lr - min_lr) * 0.5 * (math.cos(math.pi * decay_frac) + 1.) 
  return lr
optim = base_model.configure_optimizer(weight_decay=weight_decay, lr=min_lr, betas=betas, device_type=device_type)

# set up logging
if MAIN_PROC_FLAG:
  log_dir = "log"
  os.makedirs(log_dir, exist_ok=True)
  train_log = os.path.join(log_dir, f"train_log.txt")
  with open(train_log, "w") as f:
    f.write(f" iter | train loss | walltime (sec) | ktoks/sec\n")
  val_log = os.path.join(log_dir, f"val_log.txt")
  with open(val_log, "w") as f:
    f.write(f" iter | val loss\n")

# training loop
checkpoint_int, print_int = 1000, 10
eval_int = print_int
for itr in range(max_itr):

  # train
  if MAIN_PROC_FLAG:
    min_t0 = time.time()
  model.train()
  optim.zero_grad()
  cum_loss = 0.
  for min_itr in range(grad_accum_steps):
    x, y = train_loader.get_batch()
    x, y = x.to(device), y.to(device)
    if torch.cuda.is_available():
      with torch.autocast(device_type=device_type, dtype=torch.bfloat16): 
        logits, loss = model(x,y)
    else:
      logits, loss = model(x,y)
    loss = loss / grad_accum_steps
    cum_loss += loss.detach()
    if DDP_FLAG:
      if (min_itr < grad_accum_steps - 1):
        with model.no_sync():
          loss.backward()
      else:
        loss.backward()
  if DDP_FLAG:
    dist.all_reduce(cum_loss, op=dist.ReduceOp.AVG)
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  lr = get_lr(itr)
  for param_group in optim.param_groups:
    param_group['lr'] = lr
  optim.step()
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  if (itr % print_int == 0) and MAIN_PROC_FLAG:
    min_t1 = time.time()
    min_dt = min_t1 - min_t0
    toks_per_sec = ddp_world_size * batch_toks / min_dt 
    print(f"iter {itr} | train loss: {cum_loss.item():.4f} | walltime: {min_dt:.2f} sec | ktoks/sec: {toks_per_sec/1e3:.2f}")
    with open(train_log, "a") as f:
      f.write(f"{itr:5d} |    {cum_loss.item():.4f} |           {min_dt:.2f} |      {toks_per_sec/1e3:.2f}\n")

  # evaluate
  FNL_ITR_FLAG = (itr == max_itr - 1)
  if (itr % eval_int == 0) or FNL_ITR_FLAG:
    model.eval()
    val_loader.reset()
    with torch.no_grad():
      cum_val_loss = 0.
      for min_itr in range(grad_accum_steps):
        x, y = val_loader.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
          logits, loss = model(x,y)
        loss /= grad_accum_steps
        cum_val_loss += loss.detach()
    if DDP_FLAG:
      dist.all_reduce(cum_val_loss, op=dist.ReduceOp.AVG)
    if MAIN_PROC_FLAG:
      with open(val_log, "a") as f:
        f.write(f"{itr:5d} |  {cum_val_loss.item():.4f}\n")
      if (itr % checkpoint_int == 0) or FNL_ITR_FLAG:
        checkpoint_path = os.path.join(log_dir, f"checkpoint_{itr:05d}.pt")
        checkpoint = {
          "device_type": device_type,
          "model_config": base_model.config,
          "optim_config": {
            "batch_toks": batch_toks,
            "B": B,
            "T": T,
            "max_lr": max_lr,
            "weight_decay": weight_decay,
            "betas": betas,
          },
          "iter": itr,
          "model_state": base_model.state_dict(),
          "optim_state": optim.state_dict(),
          "train loss": cum_loss.item(),
          "val loss": cum_val_loss.item(),
        }
        torch.save(checkpoint, checkpoint_path)

if DDP_FLAG:
  destroy_process_group()

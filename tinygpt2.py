################
# data loading #
################

import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# set random seeds for reproducibility
torch.manual_seed(8888)
if torch.cuda.is_available():
  torch.cuda.manual_seed(8888)
  torch.set_float32_matmul_precision("high")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

def load_npy(filename):
  npa = np.load(filename)
  ptt = torch.tensor(npa.astype(np.int32), dtype=torch.long) # torch doesn't like int16, so convert to int32 before passing to torch
  return ptt

def get_pos_ids(ids, block_size):
  bos_mask = F.pad(ids == 0, (1, 0), value=1)[:-1] # shift to the right by one
  doc_ids = bos_mask.cumsum(dim=-1) - 1
  pos_ids = (~bos_mask).cumsum(dim=-1)
  pos_ids = pos_ids - pos_ids[bos_mask][doc_ids]
  while any(pos_ids >= block_size):
    pos_ids[pos_ids >= block_size] -= block_size
  return pos_ids

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
    if IS_MAIN_PROC:
      print(f"loading {len(shards)} shards in {split} split")
    self.reset()

  def reset(self):
    self.shard_idx = 0
    self.tokens = load_npy(self.shards[self.shard_idx])
    self.pos = self.B * self.T * self.rank

  def get_batch(self):
    B, T = self.B, self.T
    tmp = self.tokens[self.pos : self.pos + B * T + 1]
    x = tmp[:-1].view(B,T)
    pos_ids = get_pos_ids(tmp[:-1], 512).view(B,T)
    y = tmp[1:].view(B,T)
    self.pos += B * T * self.world_size
    if (self.pos + B * T * self.world_size + 1) > len(self.tokens):
      self.shard_idx = (self.shard_idx + 1) % len(self.shards) 
      self.tokens = load_npy(self.shards[self.shard_idx])
      self.pos = B * T * self.rank
    return x, y, pos_ids

########
# main #
########################################################
# torchrun --standalone --nproc_per_node=2 tinygpt2.py #
########################################################

from dataclasses import dataclass
import math
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from gpt import GPT

# set up training environment
USE_DDP = (int(os.environ.get('RANK',-1)) != -1)
if USE_DDP:
  assert torch.cuda.is_available()
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  device = f"cuda:{ddp_local_rank}"
  torch.cuda.set_device(device)
  IS_MAIN_PROC = (ddp_rank == 0)
  print(f"setting up GPU {ddp_rank+1} of {ddp_world_size}")
else:
  ddp_rank = 0
  ddp_local_rank = ddp_rank
  ddp_world_size = 1
  IS_MAIN_PROC = True
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
  n_embed: int = 512

@dataclass
class OptimConfig:
  device: str 
  world_size: int
  B: int = 32 # per device batch size (B, T = 32, 512 takes up 13GB VRAM)
  batch_tokens: int = 524288
  betas: tuple = (0.9, 0.95)
  max_itr: int = 9500
  max_lr: float = 1e-3
  T: int = 512 # block size / context length
  weight_decay: float = 1e-2

  def __post_init__(self):
    self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
    assert self.batch_tokens % (self.B * self.T * self.world_size) == 0
    self.grad_accum_steps = self.batch_tokens // (self.B * self.T * self.world_size)
    self.max_decay_itr = self.max_itr
    self.min_lr = self.max_lr / 10.
    self.warmup_itr = self.max_itr // 100
  
  def cosine_decay(self, itr):
    if itr < self.warmup_itr:
      lr = self.max_lr * (itr+1) / self.warmup_itr
    elif itr > self.max_decay_itr:
      lr = self.min_lr
    else:  # cosine decay for remaining 90%
      decay_frac = (itr - self.warmup_itr) / (self.max_itr - self.warmup_itr)
      lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (math.cos(math.pi * decay_frac) + 1.)
    return lr

# instantiate model
model = GPT(GPTConfig(vocab_size=50304 ))
model.to(device)
# model = torch.compile(model)
if USE_DDP:
  model = DDP(model, device_ids=[ddp_local_rank])
base_model = model.module if USE_DDP else model

# instantiate optimizer
opt_config = OptimConfig(device=device, world_size=ddp_world_size)
optimizer = base_model.configure_optimizer(opt_config)

# data loaders
train_loader = TinyStoriesDataLoader(B=opt_config.B, T=opt_config.T, rank=ddp_rank, world_size=ddp_world_size, split="train")
val_loader = TinyStoriesDataLoader(B=opt_config.B, T=opt_config.T, rank=ddp_rank, world_size=ddp_world_size, split="val")

# set up logging
if IS_MAIN_PROC:
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
for itr in range(opt_config.max_itr):

  # train
  if IS_MAIN_PROC:
    min_t0 = time.time()
  model.train()
  optimizer.zero_grad()
  avg_loss = 0.
  for min_itr in range(opt_config.grad_accum_steps):
    x, y, pos_ids = train_loader.get_batch()
    x, y, pos_ids = x.to(device), y.to(device), pos_ids.to(device)
    if torch.cuda.is_available():
      with torch.autocast(device_type=opt_config.device_type, dtype=torch.bfloat16): 
        logits, loss = model(x, y, pos_ids)
    else:
      logits, loss = model(x, y, pos_ids)
    loss = loss / opt_config.grad_accum_steps
    avg_loss += loss.detach()
    if USE_DDP:
      if (min_itr < opt_config.grad_accum_steps - 1):
        with model.no_sync():
          loss.backward()
      else:
        loss.backward()
    else:
      loss.backward()
  if USE_DDP:
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  lr = opt_config.cosine_decay(itr)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  if (itr % print_int == 0) and IS_MAIN_PROC:
    min_t1 = time.time()
    min_dt = min_t1 - min_t0
    TPS = ddp_world_size * opt_config.batch_tokens / min_dt 
    print(f"iter {itr} | train loss: {avg_loss.item():.4f} | walltime: {min_dt:.2f} sec | ktoks/sec: {TPS/1e3:.2f}")
    with open(train_log, "a") as f:
      f.write(f"{itr:5d} |    {avg_loss.item():.4f} |           {min_dt:.2f} |      {TPS/1e3:.2f}\n")

  # evaluate
  IS_MAX_ITR = (itr == opt_config.max_itr - 1)
  if (itr % eval_int == 0) or IS_MAX_ITR:
    model.eval()
    val_loader.reset()
    with torch.no_grad():
      avg_val_loss = 0.
      for min_itr in range(opt_config.grad_accum_steps):
        x, y, pos_ids = val_loader.get_batch()
        x, y, pos_ids = x.to(device), y.to(device), pos_ids.to(device)
        if torch.cuda.is_available():
          with torch.autocast(device_type=opt_config.device_type, dtype=torch.bfloat16): 
            logits, loss = model(x, y, pos_ids)
        else:
          logits, loss = model(x, y, pos_ids)
        loss /= opt_config.grad_accum_steps
        avg_val_loss += loss.detach()
    if USE_DDP:
      dist.all_reduce(avg_val_loss, op=dist.ReduceOp.AVG)
    if IS_MAIN_PROC:
      with open(val_log, "a") as f:
        f.write(f"{itr:5d} |  {avg_val_loss.item():.4f}\n")
      if (itr % checkpoint_int == 0) or IS_MAX_ITR:
        checkpoint_path = os.path.join(log_dir, f"checkpoint_{itr:05d}.pt")
        checkpoint = {
          "iter": itr,
          "model_config": base_model.config,
          "model_state": base_model.state_dict(),
          "optimizer_config": opt_config,
          "optimizer_state": optimizer.state_dict(),
          "train loss": avg_loss.item(),
          "val loss": avg_val_loss.item(),
        }
        torch.save(checkpoint, checkpoint_path)

if USE_DDP:
  destroy_process_group()

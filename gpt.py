###############
# GPT-2 model #
###############

import inspect
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0 # n_embd must be divisible by n_head
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False) # q, k, v projections all in one op
    self.c_proj = nn.Linear(self.n_embd, self.n_embd)
    self.c_proj.RESIDUAL_LAYER = True
    # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # not needed for flash attention
  
  def forward(self, x, attention_mask=None):
    B, T, C = x.size() # C is n_head * head_size here
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # q, k, v are (B, T, C)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, head_size = C / n_head)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) 
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
    # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)
    # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    # att = F.softmax(att, dim=-1) 
    # x = att @ v # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
    if attention_mask is None:
      x = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention (replaces the preceding 4 lines of code)
    else:
      x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.view(B,1,T,T))
    x = x.transpose(1,2).contiguous().view(B, T, C) # stack head outputs
    x = self.c_proj(x)
    return x

class GroupedQueryAttention(nn.Module):

  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0 # n_embd must be divisible by n_head
    assert config.n_head % config.kv_heads == 0 # n_head must be divisible by kv_heads
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.kv_heads = config.kv_heads
    self.gqa_sz = self.n_head // self.kv_heads
    self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.kv_attn = nn.Linear(self.n_embd, 2 * self.n_embd // self.gqa_sz, bias=False) # k, v projections all in one op
    self.c_proj = nn.Linear(self.n_embd, self.n_embd)
    self.c_proj.RESIDUAL_LAYER = True
  
  def forward(self, x, attention_mask=None):
    B, T, C = x.size() # C is n_head * head_size here
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, head_size = C / n_head)
    k, v = self.kv_attn(x).split(self.n_embd // self.gqa_sz, dim=2) # k, v are (B, T, C)
    k = k.view(B, T, self.kv_heads, C // self.n_head).transpose(1,2)  # (B, kv_heads, T, head_size = C / n_head)
    v = v.view(B, T, self.kv_heads, C // self.n_head).transpose(1,2) 
    if attention_mask is None:
      x = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True) # flash GQA 
    else:
      x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.view(B,1,T,T), enable_gqa=True)
    x = x.transpose(1,2).contiguous().view(B, T, C) # stack head outputs
    x = self.c_proj(x)
    return x

class MLP(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.RESIDUAL_LAYER = True

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

class TransformerBlock(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    # self.attn = MultiHeadAttention(config)
    self.attn = GroupedQueryAttention(config) # use GQA for smaller models
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x, attention_mask=None):
    x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
    x = x + self.mlp(self.ln_2(x))
    return x

class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd),
      wpe = nn.Embedding(config.block_size, config.n_embd),
      h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
      ln_f = nn.LayerNorm(config.n_embd)
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight # share wte and lm_head weights

    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    std = self.config.n_embd ** -0.5
    if isinstance(module, nn.Linear):
      if hasattr(module, 'RESIDUAL_LAYER'):
        std *= (2 * self.config.n_layer) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0., std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0., std=std)

  def forward(self, ids, targets=None, attention_mask=None):
    B, T = ids.size()
    if T > self.config.block_size:
      ids = ids[:, :self.config.block_size]
      warnings.warn(
        f"The input length ({T}) exceeds the model's block size ({self.config.block_size}). The inputs have been truncated to the block size.",
        UserWarning
      )
    pos = torch.arange(T, dtype=torch.long, device=ids.device)
    pos_emb = self.transformer.wpe(pos)
    tok_emb = self.transformer.wte(ids)
    x = pos_emb + tok_emb
    for block in self.transformer.h:
      x = block(x, attention_mask=attention_mask)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    if targets is None:
      return logits
    else:
      loss = F.cross_entropy(logits.view(B*T,-1), targets.view(B*T))
      return logits, loss
  
  def generate(self, ids, max_new_tokens=100, temp=1., topk=50):
    B, T = ids.size()
    for _ in range(max_new_tokens):
      if T > self.config.block_size:
        ids = ids[:, :self.config.block_size]
        warnings.warn(
          f"The inputs have been truncated to the block size ({self.config.block_size}) because their length ({T}) exceeds the block size.",
          UserWarning
        )
      logits = self(ids)[:,-1,:] / temp
      probs = F.softmax(logits, dim=-1)
      topk_probs, topk_ids = torch.topk(probs, topk, dim=-1)
      new_ids = torch.multinomial(topk_probs, 1)
      new_ids = torch.gather(topk_ids, -1, new_ids)
      ids = torch.cat([ids, new_ids], dim=1)
    return ids      

  def configure_optimizer(self, config):
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # weight decay 2D params (ie all params except biases and layernorm)
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
      {'params': decay_params, 'weight_decay': config.weight_decay},
      {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    # use the fused AdamW if it's available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and config.device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optim = torch.optim.AdamW(optim_groups, betas=config.betas, **extra_args)
    # print(f"using fused AdamW: {use_fused}")
    return optim

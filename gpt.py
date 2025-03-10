###############
# GPT-2 model #
###############
import inspect
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.nn.attention as attn
# import torch.nn.attention.flex_attention as FA

class MultiHeadAttention(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    assert config.n_embed % config.n_head == 0 # n_embed must be divisible by n_head
    self.n_head = config.n_head
    self.n_embed = config.n_embed
    self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed, bias=False) # q, k, v projections all in one op
    self.c_proj = nn.Linear(self.n_embed, self.n_embed)
    self.c_proj.RESIDUAL_LAYER = True
  
  def forward(self, x, attn_mask):
    B, T, C = x.size() # C is n_head * head_size here
    q, k, v = self.c_attn(x).split(self.n_embed, dim=2) # q, k, v are (B, T, C)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, T, n_head, head_size = C / n_head)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    # x = FA.flex_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), block_mask=attn_mask)
    x = x.transpose(1,2).contiguous().view(B, T, C) # stack head outputs
    x = self.c_proj(x)
    return x

class MLP(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
    self.c_proj.RESIDUAL_LAYER = True

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

class TransformerBlock(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embed)
    self.attn = MultiHeadAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embed)
    self.mlp = MLP(config)

  def forward(self, x, attn_mask):
    x = x + self.attn(self.ln_1(x), attn_mask)
    x = x + self.mlp(self.ln_2(x))
    return x

class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embed),
      wpe = nn.Embedding(config.block_size, config.n_embed),
      h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
      ln_f = nn.LayerNorm(config.n_embed)
    ))
    self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight # share wte and lm_head weights

    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    std = self.config.n_embed ** -0.5
    if isinstance(module, nn.Linear):
      if hasattr(module, 'RESIDUAL_LAYER'):
        std *= (2 * self.config.n_layer) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0., std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0., std=std)

  def forward(self, ids, targets=None, pos_ids=None):
    B, T = ids.size()
    attn_mask = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
    if pos_ids is None:
      pos_ids = torch.arange(T, device=ids.device)
    else:
      bos_mask = (pos_ids == 0)
      doc_ids = bos_mask.cumsum(dim=-1)
      doc_ids = F.pad(doc_ids, (1,0))[...,:-1].view(B,T)
      doc_mask = doc_ids.view(B,T,1) == doc_ids.view(B,1,T)
      attn_mask = attn_mask.expand(B,T,T) & doc_mask
      # attn_mask_mod = lambda b, h, q, k : (q >= k) & doc_ids[q] == doc_ids[k]
      # attn_mask = FA.create_block_mask(attn_mask_mod, B, self.config.n_head, self.config.n_embed, self.config.n_embed, device=ids.device, BLOCK_SIZE=512)
    pos_emb = self.transformer.wpe(pos_ids)
    tok_emb = self.transformer.wte(ids)
    x = pos_emb + tok_emb
    for block in self.transformer.h:
      x = block(x, attn_mask.view(B,1,T,T))
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

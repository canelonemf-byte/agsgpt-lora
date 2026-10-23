"""AGsGPT: red neuronal GPT desde cero, PyTorch puro."""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    def __init__(self):
        self.stoi, self.itos = {}, {}

    def fit(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        return self

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps({"stoi": self.stoi}), encoding="utf-8")

    def load(self, path):
        self.stoi = {k: int(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8"))["stoi"].items()}
        self.itos = {v: k for k, v in self.stoi.items()}
        return self


class CausalSelfAttention(nn.Module):
    def __init__(self, d, h, ctx):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.register_buffer("mask", torch.tril(torch.ones(ctx, ctx)).view(1, 1, ctx, ctx))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d, h, ctx, ff_mult=4, p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h, ctx)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class AGsGPT(nn.Module):
    """GPT propio del AGs. Se entrena con tus datos. No sustituye a las herramientas físicas."""

    def __init__(self, vocab_size, d=256, h=8, n_layers=6, ctx=256):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.Sequential(*[Block(d, h, ctx) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.ln(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=120, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new):
            idx_ctx = idx[:, -self.ctx :]
            logits, _ = self(idx_ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


class LoRALinear(nn.Module):
    """Adaptador LoRA: W·x + (B·A·x)·(α/r). La base queda congelada."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: int = 8):
        super().__init__()
        self.base = linear
        self.rank = rank
        self.A = nn.Parameter(torch.zeros(rank, linear.in_features))
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / rank
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def apply_lora(model: AGsGPT, rank: int = 8, alpha: int = 8) -> int:
    """Envuelve qkv/proj/ff de cada bloque con LoRA. Devuelve nº de params entrenables."""
    if rank <= 0:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    wrapped = []
    for block in model.blocks:
        if not isinstance(block.attn.qkv, LoRALinear):
            block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
            block.attn.proj = LoRALinear(block.attn.proj, rank, alpha)
            block.ff[0] = LoRALinear(block.ff[0], rank, alpha)
            block.ff[2] = LoRALinear(block.ff[2], rank, alpha)
        wrapped += [block.attn.qkv, block.attn.proj, block.ff[0], block.ff[2]]
    for p in model.parameters():
        p.requires_grad_(False)
    for m in wrapped:
        m.A.requires_grad_(True)
        m.B.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapter_state(model: nn.Module) -> dict:
    """Solo los pesos LoRA (A/B)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.rsplit(".", 1)[-1] in ("A", "B")}

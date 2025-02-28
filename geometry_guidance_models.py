#attention implementation is based on 3DShape2Vecset code
#https://github.com/1zb/3DShape2VecSet/blob/master/models_ae.py
import torch
from functools import wraps
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, drop_path_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None,
                 heads=8, dim_head=64, drop_path_rate=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # self.initialize()

    def forward(self, x, context=None, mask=None, return_scores=False):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j k -> (b h) j k', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if return_scores:
            scores = rearrange(sim, '(b h) n k -> b n k h', h=self.heads)

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if return_scores:
            return self.drop_path(self.to_out(out)), scores
        else:
            return self.drop_path(self.to_out(out))

    def initialize(self):
        # Iterate through all layers and set weights and biases to zero
        for layer in self.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm):
                nn.init.uniform_(layer.weight, a=-1e-6, b=1e-6)
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, a=-1e-6, b=1e-6)


class CLIPGeometryGuide(nn.Module):
    def __init__(
            self,
            *,
            depth=4,
            clip_dim=1024,
            pb_dim=768,
            heads=8,
            dim_head=1024,
            weight_tie_layers=False,
            drop_path_rate=0.1
    ):
        super().__init__()

        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.dim = clip_dim
        dim = clip_dim
        get_latent_attn = lambda: PreNorm(dim,
                                          Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=drop_path_rate))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=drop_path_rate))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.cross_attn_layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        self.pb_proj = torch.nn.Linear(pb_dim, clip_dim)
        self.to_outputs = torch.nn.Linear(dim, clip_dim)
        nn.init.uniform_(self.to_outputs.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.to_outputs.bias, a=-1e-3, b=1e-3)

        for i in range(depth):
            self.cross_attn_layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(self, prompt_embs, pb_embs=None):
        x = prompt_embs

        if pb_embs is not None:
            context = self.pb_proj(pb_embs)
        else:
            context = None

        for cross_attn, self_ff in self.cross_attn_layers:
            x = cross_attn(x, context=context) + x
            x = self_ff(x) + x
            # print(x.shape)

        delta_embs = self.to_outputs(x)

        return delta_embs


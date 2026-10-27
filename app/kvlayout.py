"""KV-cache slab layout shared by the prefill and decode workers.

The Hugging Face cache for layer L holds K and V tensors of shape [1, kv_heads, tokens, head_dim].
The slab stores them layer after layer, K then V, each C-contiguous:

    slab = [layer 0: K | V][layer 1: K | V] ... [layer N-1: K | V]

so one layer is one contiguous byte range (what a layer-wise RDMA WRITE pushes) and the whole slab is one
contiguous range (what a bulk RDMA WRITE pushes). The size is analytic:

    bytes = 2 * layers * kv_heads * head_dim * tokens * dtype_bytes

The slab memory itself is allocated by the transport (an RDMA-registered buffer, or a plain buffer for TCP);
this module only gives tensor views over it.
"""
import zlib
from dataclasses import asdict, dataclass

import numpy as np
import torch
from transformers import DynamicCache

DTYPES = {"fp32": (torch.float32, 4), "fp16": (torch.float16, 2)}


@dataclass
class KVSpec:
    layers: int
    kv_heads: int
    head_dim: int
    tokens: int
    dtype: str  # "fp32" | "fp16"

    @property
    def torch_dtype(self):
        return DTYPES[self.dtype][0]

    @property
    def elem_bytes(self):
        return DTYPES[self.dtype][1]

    @property
    def layer_bytes(self):
        return 2 * self.kv_heads * self.tokens * self.head_dim * self.elem_bytes

    @property
    def total_bytes(self):
        return self.layers * self.layer_bytes

    @property
    def bytes_per_token(self):
        return 2 * self.layers * self.kv_heads * self.head_dim * self.elem_bytes

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return KVSpec(**d)


def spec_from_config(cfg, tokens, dtype):
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return KVSpec(cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim, tokens, dtype)


def layer_view(slab, spec, layer):
    """Tensor [2, kv_heads, tokens, head_dim] (index 0 = K, 1 = V) over the slab bytes of one layer."""
    off = layer * spec.layer_bytes
    raw = torch.from_numpy(slab[off:off + spec.layer_bytes])
    return raw.view(spec.torch_dtype).reshape(2, spec.kv_heads, spec.tokens, spec.head_dim)


def pack_layer(slab, spec, layer, k, v):
    """Copy one layer's K and V ([1, kv_heads, tokens, head_dim]) into the slab, casting to the wire dtype."""
    t = layer_view(slab, spec, layer)
    t[0].copy_(k[0])
    t[1].copy_(v[0])


def pack_cache(slab, spec, cache):
    for i in range(spec.layers):
        pack_layer(slab, spec, i, cache.key_cache[i], cache.value_cache[i])


def unpack_cache(slab, spec):
    """Rebuild a DynamicCache (fp32, the model dtype) from the slab."""
    layers = []
    for i in range(spec.layers):
        t = layer_view(slab, spec, i).to(torch.float32)
        layers.append((t[0].unsqueeze(0).contiguous(), t[1].unsqueeze(0).contiguous()))
    return DynamicCache.from_legacy_cache(tuple(layers))


def slab_crc32(slab):
    """Exact byte integrity of the transferred slab."""
    return zlib.crc32(memoryview(slab)) & 0xFFFFFFFF


def slab_checksum(slab, spec):
    """Sum of all KV values in float64, the analogue of the training lab's gradient checksum."""
    return float(torch.from_numpy(slab[:spec.total_bytes]).view(spec.torch_dtype).double().sum())


def weights_checksum(model):
    return float(sum(p.double().sum() for p in model.parameters()))

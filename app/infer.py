"""Disaggregated LLM inference: prefill on one worker, decode on the other, KV cache over RoCEv2.

  --role prefill     process the prompt, build the real KV cache, push it to the decode worker (rank 0, hosts the store)
  --role decode      receive the KV cache into registered memory, rebuild it, generate tokens (rank 1)
  --role colocated   prefill and decode in one process with no network: the reference run

Greedy decoding, fixed seed and thread count, so the colocated and disaggregated runs must produce identical
tokens and an identical KV byte checksum. With --kv-dtype fp16 the colocated run quantizes the cache at the same
point where the disaggregated run does, so the comparison stays exact.

Timeline (wall clock, both workers share the host clock):
  t_req       the request enters prefill
  prefill_ms  forward pass over the prompt (includes layer-wise pushes when --mode layerwise)
  kv_ready    the last KV byte is in decode memory (immediate of the last WRITE received)
  ttft_ms     max(kv_ready, first token published) - t_req: decode emits token 0 once it holds the whole cache
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import timedelta

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvlayout as kv  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/SmolLM2-135M")
PROMPT_FILE = os.environ.get("PROMPT_FILE", "/lab/prompts/source.txt")


def log(msg):
    print(msg, flush=True)


def kvs(**kw):
    return " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items())


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["prefill", "decode", "colocated"], default=os.environ.get("ROLE", "colocated"))
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--kv-dtype", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--mode", choices=["bulk", "layerwise"], default="bulk")
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--transport", choices=["roce", "tcp"], default="roce")
    p.add_argument("--requests", type=int, default=1)
    p.add_argument("--threads", type=int, default=int(os.environ.get("TORCH_THREADS", "4")))
    p.add_argument("--control", choices=["none", "zero-kv"], default="none",
                   help="decode only: ignore the received KV and use zeros, to show the KV carries the state")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
    return tok, model


def prompt_ids(tok, n):
    ids = tok(open(PROMPT_FILE).read(), return_tensors="pt").input_ids
    if ids.shape[1] < n:
        raise SystemExit(f"prompt source has only {ids.shape[1]} tokens, need {n}")
    return ids[:, :n]


def prefill_forward(model, ids, cache, hooks=()):
    handles = [model.model.layers[i].register_forward_hook(h) for i, h in hooks]
    try:
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, num_logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return int(out.logits[0, -1].argmax())


def decode_loop(model, tok, cache, first, n_new):
    """Greedy decode. Token 0 is `first` (already known); returns ids and per-token step times in ms."""
    ids, steps = [first], []
    cur = first
    for _ in range(n_new - 1):
        t = time.perf_counter()
        out = model(input_ids=torch.tensor([[cur]]), past_key_values=cache, use_cache=True)
        cur = int(out.logits[0, -1].argmax())
        steps.append((time.perf_counter() - t) * 1000)
        ids.append(cur)
    return ids, steps


def itl_summary(steps):
    if not steps:
        return {}
    s = sorted(steps)
    return dict(itl_mean_ms=statistics.mean(steps), itl_p50_ms=s[len(s) // 2], itl_max_ms=s[-1])


def run_colocated(args, tok, model):
    for r in range(args.requests):
        ids = prompt_ids(tok, args.prompt_tokens)
        spec = kv.spec_from_config(model.config, ids.shape[1], args.kv_dtype)
        t_req = time.perf_counter()
        cache = kv.DynamicCache()
        first = prefill_forward(model, ids, cache)
        prefill_ms = (time.perf_counter() - t_req) * 1000
        # Same pack/unpack path as the handoff, so fp16 quantization is applied identically.
        slab = torch.zeros(spec.total_bytes, dtype=torch.uint8).numpy()
        t = time.perf_counter()
        kv.pack_cache(slab, spec, cache)
        pack_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        cache = kv.unpack_cache(slab, spec)
        unpack_ms = (time.perf_counter() - t) * 1000
        gen, steps = decode_loop(model, tok, cache, first, args.max_new_tokens)
        log(f"role=colocated req={r} " + kvs(prompt_tokens=spec.tokens, kv_dtype=spec.dtype, kv_bytes=spec.total_bytes,
            prefill_ms=prefill_ms, pack_ms=pack_ms, unpack_ms=unpack_ms, ttft_ms=prefill_ms,
            kv_crc32=f"{kv.slab_crc32(slab):08x}", kv_checksum=f"{kv.slab_checksum(slab, spec):.6f}",
            **itl_summary(steps)))
        log(f"role=colocated req={r} generated_ids={gen}")
        log(f"role=colocated req={r} text={tok.decode(gen)!r}")


def make_transport(args, store):
    if args.transport == "roce":
        from kv_transport import RoceTransport
        return RoceTransport(store, args.role, os.environ["ROCE_DEV"], channels=args.channels, log=log)
    from kv_transport import TcpTransport
    return TcpTransport(store, args.role, log=log)


def run_prefill(args, tok, model, store, tr):
    for r in range(args.requests):
        ids = prompt_ids(tok, args.prompt_tokens)
        spec = kv.spec_from_config(model.config, ids.shape[1], args.kv_dtype)
        t_req = time.time()
        store.set(f"meta/{r}", json.dumps(dict(spec=spec.to_dict(), mode=args.mode, channels=args.channels,
                                               transport=args.transport, t_req=t_req)))
        slab = tr.open_request(r, spec.total_bytes)
        cache = kv.DynamicCache()
        pack_ms = 0.0
        push_t = {}

        def layer_hook(i):
            def hook(module, inputs, output):
                nonlocal pack_ms
                t = time.perf_counter()
                kv.pack_layer(slab, spec, i, cache.key_cache[i], cache.value_cache[i])
                pack_ms += (time.perf_counter() - t) * 1000
                tr.push(r, i * spec.layer_bytes, spec.layer_bytes, i)
                push_t[i] = time.perf_counter()
            return hook

        hooks = [(i, layer_hook(i)) for i in range(spec.layers)] if args.mode == "layerwise" else []
        t_c = time.perf_counter()
        first = prefill_forward(model, ids, cache, hooks)
        prefill_ms = (time.perf_counter() - t_c) * 1000
        store.set(f"first/{r}", str(first))
        if args.mode == "bulk":
            t = time.perf_counter()
            kv.pack_cache(slab, spec, cache)
            pack_ms = (time.perf_counter() - t) * 1000
            tr.push(r, 0, spec.total_bytes, 0)
        t_pushed = time.perf_counter()
        tr.flush()
        push_done_ms = (time.perf_counter() - t_c) * 1000
        crc = kv.slab_crc32(slab)
        log(f"role=prefill req={r} " + kvs(prompt_tokens=spec.tokens, kv_dtype=spec.dtype, kv_bytes=spec.total_bytes,
            bytes_per_token=spec.bytes_per_token, mode=args.mode, channels=args.channels, transport=args.transport,
            prefill_ms=prefill_ms, pack_ms=pack_ms, push_done_ms=push_done_ms,
            exposed_transfer_ms=push_done_ms - prefill_ms - (pack_ms if args.mode == "bulk" else 0.0),
            kv_crc32=f"{crc:08x}", kv_checksum=f"{kv.slab_checksum(slab, spec):.6f}", first_token=first))
        done = json.loads(store.get(f"done/{r}"))
        log(f"role=prefill req={r} " + kvs(**done))


def run_decode(args, tok, model, store, tr):
    for r in range(args.requests):
        meta = json.loads(store.get(f"meta/{r}"))
        spec = kv.KVSpec.from_dict(meta["spec"])
        slab = tr.open_request(r, spec.total_bytes)
        expected = spec.layers * meta["channels"] if meta["mode"] == "layerwise" else meta["channels"]
        if args.control == "zero-kv":
            slab[:] = 0
            first = int(store.get(f"first/{r}"))
            tr.wait_data(expected)  # still drain the transfer so the sender completes, then discard it
            slab[:] = 0
            t_kv = time.time()
            info = {"t_first": t_kv, "messages": expected}
        else:
            info = tr.wait_data(expected)
            t_kv = info["t_last"]
            first = int(store.get(f"first/{r}"))
        t_first = time.time()
        t_ready = max(t_kv, t_first)
        crc = kv.slab_crc32(slab)
        t = time.perf_counter()
        cache = kv.unpack_cache(slab, spec)
        unpack_ms = (time.perf_counter() - t) * 1000
        gen, steps = decode_loop(model, tok, cache, first, args.max_new_tokens)
        res = dict(prompt_tokens=spec.tokens, kv_dtype=spec.dtype, kv_bytes=spec.total_bytes, mode=meta["mode"],
                   channels=meta["channels"], transport=meta["transport"], control=args.control,
                   kv_ready_ms=(t_kv - meta["t_req"]) * 1000,
                   ttft_ms=(t_ready - meta["t_req"]) * 1000, unpack_ms=unpack_ms,
                   writes_received=info["messages"], kv_crc32=f"{crc:08x}",
                   kv_checksum=f"{kv.slab_checksum(slab, spec):.6f}", **itl_summary(steps))
        log(f"role=decode req={r} " + kvs(**res))
        log(f"role=decode req={r} generated_ids={gen}")
        log(f"role=decode req={r} text={tok.decode(gen)!r}")
        store.set(f"done/{r}", json.dumps(dict(decode_ttft_ms=round(res["ttft_ms"], 2), decode_kv_crc32=res["kv_crc32"],
                                               decode_unpack_ms=round(unpack_ms, 2))))


def main():
    args = parse()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    t = time.perf_counter()
    tok, model = load()
    cfg = model.config
    log(f"role={args.role} model={os.path.basename(MODEL_DIR)} layers={cfg.num_hidden_layers} "
        f"kv_heads={cfg.num_key_value_heads} head_dim={cfg.hidden_size // cfg.num_attention_heads} "
        f"load_s={time.perf_counter() - t:.1f} threads={args.threads} "
        f"weights_checksum={kv.weights_checksum(model):.6f}")
    if args.role == "colocated":
        return run_colocated(args, tok, model)

    from torch.distributed import TCPStore
    store = TCPStore(os.environ.get("MASTER_ADDR", "10.10.1.2"), int(os.environ.get("MASTER_PORT", "29500")),
                     world_size=2, is_master=args.role == "prefill", timeout=timedelta(seconds=600),
                     wait_for_workers=True)
    tr = make_transport(args, store)
    (run_prefill if args.role == "prefill" else run_decode)(args, tok, model, store, tr)


if __name__ == "__main__":
    with torch.no_grad():
        main()

"""Train a small Mamba-3 LM on WikiText-2 and compare dense vs low-rank projections.

  python train.py --variant baseline              --steps 3000 --out runs/baseline.json
  python train.py --variant lowrank --rank 64     --steps 3000 --out runs/lowrank_r64.json
  python train.py --compare runs/*.json           # print a comparison table

Smoke test without downloads (random tokens, tiny model, CPU ok):
  python train.py --synthetic --variant lowrank --rank 8 --steps 5 --d_model 64 --n_layer 1 --seq 32 --bs 2
"""
import argparse, importlib.util, json, math, os, sys, time, types
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_block_module(variant):
    """Load mamba3_baseline.py / mamba3_lowrank.py (needs mamba_ssm.utils.tap; stubbed if absent)."""
    try:
        import mamba_ssm.utils.tap  # noqa: F401
    except ImportError:
        for n in ("mamba_ssm", "mamba_ssm.utils"):
            sys.modules.setdefault(n, types.ModuleType(n))
        t = types.ModuleType("mamba_ssm.utils.tap"); t.tap = lambda *a, **k: None
        sys.modules["mamba_ssm.utils.tap"] = t
    f = "mamba3_baseline.py" if variant == "baseline" else "mamba3_lowrank.py"
    spec = importlib.util.spec_from_file_location("mamba3_" + variant, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.eps = eps; self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    def __init__(self, d, d_mlp):
        super().__init__()
        self.w1 = nn.Linear(d, d_mlp, bias=False); self.w3 = nn.Linear(d, d_mlp, bias=False)
        self.w2 = nn.Linear(d_mlp, d, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, mixer, d, d_mlp):
        super().__init__()
        self.n1, self.mixer, self.n2, self.mlp = RMSNorm(d), mixer, RMSNorm(d), SwiGLU(d, d_mlp)
    def forward(self, x):
        x = x + self.mixer(self.n1(x))
        return x + self.mlp(self.n2(x))


class LM(nn.Module):
    def __init__(self, mod, vocab, d_model, n_layer, d_state, d_mlp, rank):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        kw = dict(d_model=d_model, d_state=d_state, expand=2, headdim=64)
        if rank is not None:
            kw["lowrank_r"] = rank
        self.blocks = nn.ModuleList([Block(mod.Mamba3(**kw), d_model, d_mlp) for _ in range(n_layer)])
        self.nf = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight   # tied
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.nf(x))


def get_data(args):
    if args.synthetic:
        g = torch.Generator().manual_seed(0)
        mk = lambda n: torch.randint(0, args.vocab, (n,), generator=g)
        return mk(20000), mk(4000), args.vocab
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)   # Llama 3.1 is gated: accept the licence + `hf auth login`
    texts = load_wikitext2_raw()
    enc = lambda split: torch.tensor(tok(texts[split])["input_ids"])
    return enc("train"), enc("validation"), len(tok)


def load_wikitext2_raw():
    """WikiText-2 raw as plain text. Uses the original zip so that pandas/pyarrow (`datasets`) are not needed."""
    import urllib.request, zipfile
    d = os.path.join(HERE, "data"); os.makedirs(d, exist_ok=True)
    z = os.path.join(d, "wikitext-2-raw-v1.zip")
    if not os.path.exists(z):
        req = urllib.request.Request("https://wikitext.smerity.com/wikitext-2-raw-v1.zip",
                                     headers={"User-Agent": "Mozilla/5.0 (train.py wikitext fetch)"})
        with urllib.request.urlopen(req) as r, open(z, "wb") as out_f:
            out_f.write(r.read())
    out = {}
    with zipfile.ZipFile(z) as f:
        for split, name in (("train", "train"), ("validation", "valid"), ("test", "test")):
            out[split] = f.read(f"wikitext-2-raw/wiki.{name}.raw").decode("utf-8")
    return out


def batches(data, bs, seq, gen):
    ix = torch.randint(0, len(data) - seq - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + seq] for i in ix]); y = torch.stack([data[i + 1:i + seq + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, bs, seq, dev, max_batches=50):
    model.eval(); tot = n = 0
    nseq = (len(data) - 1) // seq
    for b in range(0, min(nseq, max_batches * bs), bs):
        idx = range(b, min(b + bs, nseq))
        x = torch.stack([data[i * seq:(i + 1) * seq] for i in idx]).to(dev)
        y = torch.stack([data[i * seq + 1:(i + 1) * seq + 1] for i in idx]).to(dev)
        loss = F.cross_entropy(model(x).float().view(-1, model.emb.num_embeddings), y.reshape(-1), reduction="sum")
        tot += loss.item(); n += y.numel()
    model.train()
    return math.exp(tot / n)


def compare(files):
    rows = [json.load(open(f)) for f in files]
    print(f"{'run':<16}{'params':>12}{'val ppl':>10}{'vs base':>10}{'s/step':>8}")
    base = next((r for r in rows if r["variant"] == "baseline"), None)
    for r in rows:
        d = "" if base is None else f"{(r['final_val_ppl'] / base['final_val_ppl'] - 1) * 100:+.1f}%"
        print(f"{r['name']:<16}{r['params']:>12,}{r['final_val_ppl']:>10.2f}{d:>10}{r['sec_per_step']:>8.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["baseline", "lowrank"], default="baseline")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=5)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--d_mlp", type=int, default=512)
    p.add_argument("--vocab", type=int, default=1000, help="synthetic mode only")
    p.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--compare", nargs="+", default=None)
    a = p.parse_args()
    if a.compare:
        return compare(a.compare)

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, vocab = get_data(a)
    mod = load_block_module(a.variant)
    model = LM(mod, vocab, a.d_model, a.n_layer, a.d_state, a.d_mlp, a.rank if a.variant == "lowrank" else None).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    name = a.variant if a.variant == "baseline" else f"lowrank_r{a.rank}"
    print(f"{name}: {n_params:,} params, vocab {vocab:,}, {len(train):,} train tokens, device {dev}")

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    sched = lambda s: min(1, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    gen = torch.Generator().manual_seed(a.seed)
    log, t0, t_meas = [], time.time(), None

    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        x, y = batches(train, a.bs, a.seq, gen)
        x, y = x.to(dev), y.to(dev)
        loss = F.cross_entropy(model(x).float().view(-1, vocab), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1:                        # measure real step time after warm-up
            if dev == "cuda": torch.cuda.synchronize()
            t_meas = time.time()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {a.lr * sched(step):.2e}")
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            ppl = evaluate(model, val, a.bs, a.seq, dev)
            log.append({"step": step + 1, "train_loss": loss.item(), "val_ppl": ppl})
            print(f"  >> step {step + 1}  val ppl {ppl:.2f}")

    if dev == "cuda": torch.cuda.synchronize()
    sec = (time.time() - t_meas) / max(1, a.steps - 2) if t_meas else float("nan")
    res = {"name": name, "variant": a.variant, "rank": a.rank if a.variant == "lowrank" else None,
           "params": n_params, "steps": a.steps, "bs": a.bs, "seq": a.seq, "lr": a.lr, "seed": a.seed,
           "sec_per_step": sec, "final_val_ppl": log[-1]["val_ppl"], "log": log}
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None
    print(f"done: val ppl {res['final_val_ppl']:.2f}, {sec:.2f} s/step, peak mem {res['peak_mem_gb']}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

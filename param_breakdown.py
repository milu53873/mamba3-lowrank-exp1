"""Parameter and per-token multiplication breakdown of the small Mamba-3 LM used in train.py.

  python param_breakdown.py                 # GPT-2 vocab (50,257), the vocab used in our runs
  python param_breakdown.py --vocab 128256  # Llama 3.1 vocab

Parameters are counted from the real model (CPU, nothing is trained).
Multiplications per token are hand-derived (not measured): the per-block numbers (603,530 / 341,386)
come from our earlier hand count; LM head = d_model * vocab; SwiGLU = 3 * d_model * d_mlp.
"""
import argparse
from collections import OrderedDict
import train

BLOCK_MULT = {"baseline": 603_530, "lowrank": 341_386}   # one Mamba-3 block, per token (hand count)


def categorize(name):
    if name.startswith("emb"):            return "embedding (tied with LM head)"
    if ".mixer.in_proj" in name:          return "Mamba in_proj"
    if ".mixer.out_proj" in name:         return "Mamba out_proj"
    if ".mixer." in name:                 return "Mamba other (norms, biases, D)"
    if ".mlp." in name:                   return "SwiGLU MLP"
    return "RMSNorm (block / final)"


def breakdown(variant, vocab, rank, d_model, n_layer, d_state, d_mlp):
    mod = train.load_block_module(variant)
    m = train.LM(mod, vocab, d_model, n_layer, d_state, d_mlp, rank if variant == "lowrank" else None)
    cats = OrderedDict((k, 0) for k in ["embedding (tied with LM head)", "Mamba in_proj", "Mamba out_proj",
                                       "Mamba other (norms, biases, D)", "SwiGLU MLP", "RMSNorm (block / final)"])
    for n, p in m.named_parameters():
        cats[categorize(n)] += p.numel()
    total = sum(cats.values())
    mults = OrderedDict()
    mults["LM head (d_model x vocab)"] = d_model * vocab
    mults[f"{n_layer} Mamba-3 blocks"] = n_layer * BLOCK_MULT[variant]
    mults[f"{n_layer} SwiGLU MLPs"] = n_layer * 3 * d_model * d_mlp
    return cats, total, mults


def show(variant, args, rank):
    cats, total, mults = breakdown(variant, args.vocab, rank, args.d_model, args.n_layer, args.d_state, args.d_mlp)
    tag = variant if variant == "baseline" else f"low-rank r={rank}"
    print(f"== {tag}   vocab={args.vocab:,}   d_model={args.d_model}   layers={args.n_layer}")
    print(f"{'parameters':<34}{'count':>12}{'share':>8}")
    for k, v in cats.items():
        print(f"{k:<34}{v:>12,}{v / total * 100:>7.1f}%")
    print(f"{'TOTAL':<34}{total:>12,}")
    mt = sum(mults.values())
    print(f"\n{'multiplications / token (hand)':<34}{'count':>12}{'share':>8}")
    for k, v in mults.items():
        print(f"{k:<34}{v:>12,}{v / mt * 100:>7.1f}%")
    print(f"{'TOTAL':<34}{mt:>12,}\n")
    return total, mt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=5)
    ap.add_argument("--d_state", type=int, default=64)
    ap.add_argument("--d_mlp", type=int, default=512)
    args = ap.parse_args()
    tb, mb = show("baseline", args, None)
    tl, ml = show("lowrank", args, args.rank)
    print(f"low-rank vs baseline: parameters {(tl / tb - 1) * 100:+.1f}%,  multiplications/token {(ml / mb - 1) * 100:+.1f}%")

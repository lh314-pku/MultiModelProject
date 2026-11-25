# -*- coding: utf-8 -*-
"""CPU-only unit tests for the math in ILVAD_repro.ipynb.

The tests pull the functions straight out of the notebook source, so there is
exactly one implementation to keep in sync.

    python tests/test_math.py

Requirements: torch (CPU is fine), numpy. No transformers, no GPU, no model.
"""
import ast
import json
import math
import sys
from pathlib import Path

import torch

NB = Path(__file__).resolve().parents[1] / "ILVAD_repro.ipynb"


def load_functions(names, extra=None):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    srcs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    found = {}
    for src in srcs:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if keep:
            ns = {"torch": torch, "math": math}
            ns.update(extra or {})
            exec(compile(ast.Module(body=keep, type_ignores=[]), "<nb>", "exec"), ns)
            for n in keep:
                found[n.name] = ns[n.name]
    missing = set(names) - set(found)
    assert not missing, f"missing functions: {missing}"
    return found


F = load_functions({"_causal", "_reweight", "build_map"},
                   {"ILVAD": {"row": -1}})
_causal, _reweight, build_map = F["_causal"], F["_reweight"], F["build_map"]

V = load_functions({"visual_span"}, {"IMG_TOK": 32000, "N_VIS": 576})
visual_span = V["visual_span"]

failures = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name, extra)
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- causal mask
m = _causal(3, 3, torch.float32, torch.device("cpu"))
check("causal shape", m.shape == (1, 1, 3, 3), str(tuple(m.shape)))
check("causal row0", bool(torch.isinf(m[0, 0, 0, 1]) and m[0, 0, 0, 0] == 0))
check("causal row2", bool((m[0, 0, 2] == 0).all()))

# ---------------------------------------------------------------- reweight
torch.manual_seed(0)
aw = torch.softmax(torch.randn(2, 4, 10), dim=-1)

out = _reweight(aw, (3, 9), torch.zeros(6), 0.0, "boost")
check("alpha=0 identity", torch.allclose(out, aw, atol=1e-6))
check("rows sum to 1", torch.allclose(out.sum(-1), torch.ones(2, 4), atol=1e-5))

score = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
boosted = _reweight(aw, (3, 9), score, 2.0, "boost")
before, after = aw[..., -1, 3:9].sum(-1), boosted[..., -1, 3:9].sum(-1)
check("boost raises visual mass", bool((after > before).all()),
      f"{float(before.mean()):.3f} -> {float(after.mean()):.3f}")

mask = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
masked = _reweight(aw, (3, 9), mask, 0.0, "mask")
check("mask zeroes the token", float(masked[..., -1, 4].abs().max()) < 1e-6)
check("mask rows sum to 1", torch.allclose(masked.sum(-1), torch.ones(2, 4), atol=1e-5))

# mixed precision: fp16 rows + fp32 score must stay fp16 (else matmul with V fails)
aw16 = torch.softmax(torch.randn(1, 4, 20).half(), dim=-1)
out16 = _reweight(aw16, (5, 13), torch.rand(8), 5.0, "boost")
check("fp16 path dtype", out16.dtype == torch.float16, str(out16.dtype))
check("fp16 rows sum to 1", torch.allclose(out16.sum(-1).float(), torch.ones(1, 4), atol=2e-3))
v = torch.randn(1, 4, 20, 8).half()
try:
    _ = out16 @ v
    check("fp16 matmul", True)
except Exception as e:  # noqa: BLE001
    check("fp16 matmul", False, repr(e))

# ---------------------------------------------------------------- patched forward
# 用一个小号的假 attention 模块跑一遍补丁后的 forward，覆盖 4D / 2D 掩码两条路径。
# 这一组测试就是为了拦住「把 head_dim 当成序列长度」这类形状错误。
class _FakeRotary:
    def __call__(self, x, position_ids=None):
        b, h, s, d = x.shape
        return torch.ones(b, s, d), torch.zeros(b, s, d)


class _FakeAttn(torch.nn.Module):
    def __init__(self, d=8, n_heads=2, n_kv=2):
        super().__init__()
        self.head_dim = d // n_heads
        self.num_heads = n_heads
        self.num_key_value_heads = n_kv
        self.num_key_value_groups = n_heads // n_kv
        self.scaling = self.head_dim ** -0.5
        self.q_proj = torch.nn.Linear(d, d, bias=False)
        self.k_proj = torch.nn.Linear(d, d, bias=False)
        self.v_proj = torch.nn.Linear(d, d, bias=False)
        self.o_proj = torch.nn.Linear(d, d, bias=False)
        self.rotary_emb = _FakeRotary()


P = load_functions(
    {"_causal", "_reweight", "_make_forward"},
    {
        "_rope": lambda q, k, cos, sin: (q, k),
        "repeat_kv": lambda x, n: x,
        "F": torch.nn.functional,
        "ILVAD": {"on": False, "score": None, "alpha": 0.0, "mode": "boost", "span": (0, 0)},
    },
)
attn = _FakeAttn()
fwd = P["_make_forward"](attn)
x = torch.randn(1, 7, 8)
pos = torch.arange(7)[None]
mask4 = torch.triu(torch.full((7, 7), float("-inf")), diagonal=1)[None, None]

out, aw, _ = fwd(attn, x, attention_mask=mask4, position_ids=pos)
check("patch forward shape", tuple(out.shape) == (1, 7, 8), str(tuple(out.shape)))
check("patch rows sum to 1", torch.allclose(aw.sum(-1), torch.ones(1, 2, 7), atol=1e-5))
check("patch causal", float(aw[0, 0, 0, 3].detach()) == 0.0)

out2, aw2, _ = fwd(attn, x, attention_mask=torch.ones(1, 7, dtype=torch.long), position_ids=pos)
check("patch 2D mask", tuple(out2.shape) == (1, 7, 8)
      and torch.allclose(aw2.sum(-1), torch.ones(1, 2, 7), atol=1e-5))

_g = P["_make_forward"].__globals__
_g["ILVAD"].update(on=True, score=torch.ones(3), alpha=2.0, mode="boost", span=(0, 3))
out3, aw3, _ = fwd(attn, x, attention_mask=mask4, position_ids=pos)
check("patch boost path", tuple(out3.shape) == (1, 7, 8)
      and torch.allclose(aw3.sum(-1), torch.ones(1, 2, 7), atol=1e-5))
_g["ILVAD"]["on"] = False

# ---------------------------------------------------------------- next-token shape
# probe_attention / generate 里 ids 是 [1, S]，argmax 必须保留 batch 维成 [1, 1]，
# 否则 torch.cat 会报 "got 2 and 1"。
ids = torch.zeros(1, 5, dtype=torch.long)
logits = torch.randn(1, 5, 10)
nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
check("next token shape", tuple(nxt.shape) == (1, 1), str(tuple(nxt.shape)))
check("cat shape", tuple(torch.cat([ids, nxt], dim=-1).shape) == (1, 6))

# ---------------------------------------------------------------- visual span
# LLaVA 的 processor 可能把 <image> 展开成 576 个占位符，也可能只留 1 个。
# 两种情况都必须定位到同一段 [1, 577)。
single = torch.tensor([[1, 32000, 5, 6]])
check("visual_span single placeholder", visual_span(single) == (1, 577), str(visual_span(single)))

expanded = torch.tensor([[1] + [32000] * 576 + [5]])
check("visual_span pre-expanded", visual_span(expanded) == (1, 577), str(visual_span(expanded)))

# ---------------------------------------------------------------- build_map
class Cfg:
    tau = 2.0
    rho = 0.5


ABAR = [
    [0.0600, 0.0600, 0.1000, 0.0600, 0.6600, 0.0600],
    [0.0650, 0.0650, 0.4000, 0.0650, 0.3375, 0.0675],
    [0.0400, 0.3400, 0.1600, 0.0400, 0.3800, 0.0400],
    [0.0500, 0.0600, 0.4200, 0.0500, 0.3600, 0.0600],
]
rows = []
for _ in range(2):
    per_step = []
    for l in range(4):
        row = torch.tensor(ABAR[l], dtype=torch.float32)
        per_step.append(torch.stack([row, row]))
    rows.append(torch.stack(per_step))

res = build_map(rows, Cfg)
check("build_map S", torch.equal(res["S"], torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0, 0.0])),
      str(res["S"].tolist()))
check("build_map Shat", torch.allclose(res["Shat"], torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0, 0.0])),
      str(res["Shat"].tolist()))
check("build_map B", res["B"].sum(-1).long().tolist() == [1, 2, 2, 2],
      str(res["B"].sum(-1).long().tolist()))

print()
if failures:
    print("FAILED:", ", ".join(failures))
    sys.exit(1)
print("ALL PASS")

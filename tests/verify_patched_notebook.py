"""Verify PVT_Tutelmoe_v10_patched.ipynb by RUNNING its model cells.

    python tests/verify_patched_notebook.py

Not part of `tests/run_all.py`: that suite covers the package, this covers the
self-contained notebook, which duplicates the logic on purpose. Kept separate
so the two never appear to validate each other.

It is not a parse check. It execs the notebook's own class definitions against
a stand-in for tutel (so it runs with no GPU and no CUDA extension) and
asserts the upcycled MoE block reproduces a dense FFN exactly at step 0 — the
bar the original notebook failed, since it discarded the stage-4 dense FFN and
put nothing in its place.
"""
import json, sys, types, torch, torch.nn as nn, torch.nn.functional as F

# ---- stand-in for tutel, matching build_moe_ffn_layer's call signature ----
class _FakeMoELayer(nn.Module):
    def __init__(self, model_dim, experts, result_func=None, **kw):
        super().__init__()
        E, hid = experts['count_per_node'], experts['hidden_size_per_expert']
        self.batched_fc1_w = nn.Parameter(torch.randn(E, hid, model_dim) * 0.02)
        self.batched_fc2_w = nn.Parameter(torch.randn(E, hid, model_dim) * 0.02)
        self.batched_fc1_bias = nn.Parameter(torch.zeros(E, hid))
        self.batched_fc2_bias = nn.Parameter(torch.zeros(E, model_dim))
        self.gate_wg = nn.Parameter(torch.randn(E, model_dim) * 0.02)
        self.act = experts['activation_fn']
        self.result_func = result_func
    def forward(self, x):
        h = torch.addmm(self.batched_fc1_bias[0], x, self.batched_fc1_w[0].t())
        out = torch.addmm(self.batched_fc2_bias[0], self.act(h), self.batched_fc2_w[0])
        aux = (x @ self.gate_wg.t()).softmax(-1).mean(0).pow(2).sum()
        out.l_aux = aux
        return self.result_func(out) if self.result_func else out

fake = types.ModuleType("tutel_moe"); fake.moe_layer = _FakeMoELayer

# the notebook's own import cell carries %pip magics, so seed its globals
import math, os
from functools import partial
from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
import pytorch_lightning as pl
g = {"__name__": "nb", "tutel_moe": fake, "torch": torch, "nn": nn, "F": F,
     "math": math, "os": os, "partial": partial, "rearrange": rearrange,
     "DropPath": DropPath, "to_2tuple": to_2tuple, "trunc_normal_": trunc_normal_,
     "pl": pl, "RMSNorm": nn.RMSNorm}

nb = json.load(open("/home/user/Thesis/PVT_Tutelmoe_v10_patched.ipynb"))
WANT = ("class OverlapPatchEmbed", "class DWConv", "class RMSNorm", "v10 PATCH: helpers",
        "class Mlp", "def _init_t_xy", "class GQAttention", "class Block",
        "class PyramidVisionTransformerV2")
ran = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    s = "".join(c["source"])
    if not any(w in s for w in WANT): continue
    if any(l.lstrip().startswith(("%", "!")) for l in s.splitlines()): continue
    try:
        exec(compile(s, f"cell{i}", "exec"), g); ran.append(i)
    except Exception as e:
        print(f"  cell {i} FAILED: {type(e).__name__}: {e}"); raise
print(f"executed model cells: {ran}")

Mlp, PVT = g["Mlp"], g["PyramidVisionTransformerV2"]
DIM, HID = 64, 128

# 1. the MoE'd Mlp reproduces a dense FFN once upcycled
torch.manual_seed(0)
moe = Mlp(in_features=DIM, hidden_features=HID, use_moe=True, num_experts=4,
          moe_shared_expert=True, moe_block_dwconv=True)
dense = Mlp(in_features=DIM, hidden_features=HID, use_moe=False)
n = moe.load_from_dense_ffn(dense.state_dict())
moe.zero_routed_experts()
moe.eval(); dense.eval()
x = torch.randn(2, 49, DIM)
with torch.no_grad():
    got, aux = moe(x, 7, 7)
    want = dense(x, 7, 7)
err = (got - want).abs().max().item()
print(f"\nupcycled MoE Mlp vs dense FFN: loaded {n} tensors, max|delta| = {err:.3e}")
assert err < 1e-5, "NOT function-preserving"
assert aux is not None

# 2. routed experts are not frozen
moe.train()
opt = torch.optim.AdamW(moe.parameters(), lr=1e-2)
fc2 = moe.moe_layer.batched_fc2_w
assert fc2.abs().sum().item() == 0.0
for _ in range(3):
    out, a = moe(torch.randn(2, 49, DIM), 7, 7)
    loss = out.pow(2).mean() + 0.01 * a
    opt.zero_grad(); loss.backward(); opt.step()
print(f"routed fc2 after 3 steps: |W| = {fc2.abs().sum().item():.4f} (was 0)")
assert fc2.abs().sum().item() > 0, "zero-init froze the routed experts"

# 3. moe_block_dwconv scoping + the plain variant
plain = Mlp(in_features=DIM, hidden_features=HID, use_moe=True, num_experts=4,
            moe_shared_expert=True, moe_block_dwconv=False)
assert plain.shared_expert.dwconv is None
assert moe.shared_expert.dwconv is not None
print("moe_block_dwconv toggles the shared branch's conv: OK")

# 4. a whole model builds and runs with MoE in the last stage
torch.manual_seed(0)
model = PVT(img_size=64, patch_size=4, in_chans=3, num_classes=10,
            embed_dims=[16,32,48,64], num_heads=[1,2,4,4], num_kv_heads=[1,1,2,2],
            mlp_ratios=[2,2,2,2], qkv_bias=True, qk_scale=None, drop_rate=0.,
            attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
            depths=[1,1,1,2], sr_ratios=[8,4,2,1], num_stages=4, linear=False,
            stage4_lr_multiplier=1.0, use_moe=True, num_experts=4,
            moe_last_n_stages=1, use_rope=True, rope_last_n_stages=1, rope_theta=50.0,
            moe_shared_expert=True, moe_block_dwconv=True)
model.train()
out = model(torch.randn(2, 3, 64, 64))
logits, aux = out if isinstance(out, tuple) else (out, None)
print(f"full model forward: logits {tuple(logits.shape)}, aux {'present' if aux is not None else 'MISSING'}")
assert logits.shape == (2, 10) and aux is not None

# non-MoE blocks keep their DWConv (scoping)
assert model.block4[1].mlp.use_moe and model.block4[1].mlp.shared_expert is not None
assert not model.block1[0].mlp.use_moe and model.block1[0].mlp.dwconv is not None
print("scoping: stage-4 last block is MoE; stage-1 keeps its dense CFFN: OK")
print("\nALL NOTEBOOK MODEL CHECKS PASSED")

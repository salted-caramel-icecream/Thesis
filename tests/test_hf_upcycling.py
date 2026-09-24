"""The HF warm start (mode hf_pretrained) end to end, on a REAL state dict.

Until now the suite never ran ``load_hf_pretrained`` with weights in it:
``test_shared_expert.py`` calls the seeding helpers directly and
``test_variants.py`` stubs ``transformers`` with an EMPTY state dict. That
gap let ``mode: hf_pretrained`` under the scratch recipe resolve
``model.moe.upcycle_init`` to "none" — shared expert seeded, routed experts
replicated, nothing zeroed, the block emitting ~2x the pretrained FFN — and
``tools/verify_upcycling.py`` reported it as an error of ~1e-1 on the real
Tutel backend while the warm-start path passed at 0.0.

Here a tiny dense model's state dict is renamed to HF PvtV2 naming (the
exact inverse of ``_remap_hf_key``, key/value split back apart), served by a
stub ``transformers`` module, and loaded into a MoE model. The bar is the
same as for the warm_start path: same input, same output, within 1e-4.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import pathlib
import re
import sys
import types

import torch

from helpers import NATIVE_MOE, install_fake_tutel_backend, tiny_config

from pvt_moe.config import default_config, merge_config, validate_config
from pvt_moe.models import build_model
from pvt_moe.models.pretrained import _remap_hf_key, load_hf_pretrained


def _backend(name: str) -> dict:
    """The moe block for one backend (native cannot take the Tutel-only router)."""
    return dict(NATIVE_MOE) if name == "native" else {"backend": name}


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "verify_upcycling.py"

# RoPE off in BOTH models: it adds parameters the HF checkpoint never had, so
# leaving it on would measure RoPE, not the upcycling (as in test_shared_expert).
_NO_ROPE = {"use_rope": False, "rope_placement": [[], [], [], []]}
_MOE_LAST_S4 = {"use_moe": True, "moe_placement": [[], [], [], [-1]]}


# --- the inverse of _remap_hf_key --------------------------------------------

_BLOCK_RENAMES = (
    ("norm1.", "layer_norm_1."),
    ("norm2.", "layer_norm_2."),
    ("attn.q.", "attention.query."),
    ("attn.proj.", "attention.proj."),
    ("attn.sr.", "attention.spatial_reduction."),
    ("attn.norm.", "attention.layer_norm."),
    ("mlp.fc1.", "mlp.dense1."),
    ("mlp.fc2.", "mlp.dense2."),
    ("mlp.dwconv.dwconv.", "mlp.dwconv.dwconv."),
)


def _to_hf_state(dense_state: dict) -> dict:
    """Rename EVERY key of this package's state dict to HF PvtV2 naming."""
    hf = {}
    for key, value in dense_state.items():
        m = re.match(r"block(\d+)\.(\d+)\.(.*)", key)
        if m:
            n, blk, rest = int(m.group(1)) - 1, m.group(2), m.group(3)
            prefix = f"pvt_v2.encoder.layers.{n}.blocks.{blk}."
            if rest.startswith("attn.kv."):
                # our fused kv = cat([key, value], dim=0) -> split it back
                suffix = rest[len("attn.kv."):]
                k, v = value.chunk(2, dim=0)
                hf[f"{prefix}attention.key.{suffix}"] = k.clone()
                hf[f"{prefix}attention.value.{suffix}"] = v.clone()
                continue
            for ours, theirs in _BLOCK_RENAMES:
                if rest.startswith(ours):
                    hf[prefix + theirs + rest[len(ours):]] = value
                    break
            else:
                raise AssertionError(f"no HF spelling for {key}")
            continue
        m = re.match(r"patch_embed(\d+)\.(proj|norm)\.(.*)", key)
        if m:
            n = int(m.group(1)) - 1
            part = {"proj": "projection", "norm": "layer_norm"}[m.group(2)]
            hf[f"pvt_v2.encoder.layers.{n}.patch_embedding.{part}.{m.group(3)}"] = value
            continue
        m = re.match(r"norm(\d+)\.(.*)", key)
        if m:
            hf[f"pvt_v2.encoder.layers.{int(m.group(1)) - 1}.layer_norm.{m.group(2)}"] = value
            continue
        m = re.match(r"head\.(.*)", key)
        if m:
            hf[f"classifier.{m.group(1)}"] = value
            continue
        raise AssertionError(f"no HF spelling for {key}")
    return hf


def _stub_transformers(depths, hidden_sizes, state):
    """``transformers`` with an AutoModelForImageClassification serving ``state``."""
    mod = types.ModuleType("transformers")

    class _Auto:
        @staticmethod
        def from_pretrained(hf_id):
            return types.SimpleNamespace(
                config=types.SimpleNamespace(depths=depths, hidden_sizes=hidden_sizes),
                state_dict=lambda: dict(state))

    mod.AutoModelForImageClassification = _Auto
    return mod


@contextlib.contextmanager
def _serving(depths, hidden_sizes, state):
    had = sys.modules.get("transformers")
    sys.modules["transformers"] = _stub_transformers(depths, hidden_sizes, state)
    try:
        yield
    finally:
        if had is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = had


def _dense_model():
    cfg = tiny_config(model={"ablation": {**_NO_ROPE, "use_moe": False}})
    torch.manual_seed(1234)
    return build_model(cfg), cfg


def _moe_model(cfg):
    undo = install_fake_tutel_backend()
    try:
        torch.manual_seed(1234)
        return build_model(cfg)
    finally:
        undo()


def _max_err(dense, moe):
    dense.eval()
    moe.eval()
    torch.manual_seed(7)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        want, _ = dense(x)
        got, _ = moe(x)
    return (got - want).abs().max().item()


# --- 1. permanent functional test --------------------------------------------

def test_hf_loader_maps_every_dense_tensor_and_preserves_the_function():
    for backend in ("tutel", "native"):
        dense, dense_cfg = _dense_model()
        dense_state = {k: v.clone() for k, v in dense.state_dict().items()}
        hf_state = _to_hf_state(dense_state)
        # every key round-trips through the loader's own remap (kv aside)
        for hf_key in hf_state:
            ours = _remap_hf_key(hf_key)
            if ".attention.key." in hf_key or ".attention.value." in hf_key:
                assert ours is None, hf_key
            else:
                assert ours in dense_state, (hf_key, ours)

        moe_cfg = tiny_config(model={"ablation": {**_NO_ROPE, **_MOE_LAST_S4},
                                     "moe": {**_backend(backend), "shared_expert": True,
                                             "upcycle_init": "routed_zero"}})
        moe = _moe_model(moe_cfg)
        with _serving(dense_cfg["model"]["depths"], dense_cfg["model"]["embed_dims"], hf_state):
            stats = load_hf_pretrained(moe, "stub/tiny", seed_moe_experts=True,
                                       upcycle_init="routed_zero", verbose=False)

        n_blocks = sum(dense_cfg["model"]["depths"])
        n_kv = sum(1 for k in dense_state if ".attn.kv." in k)          # weight + bias per block
        n_moe_mlp = sum(1 for k in dense_state if k.startswith("block4.1.mlp."))
        assert n_moe_mlp == 6, n_moe_mlp                                  # fc1/fc2/dwconv x (w, b)
        assert stats["unmapped"] == [], stats["unmapped"]
        assert stats["dropped_no_target"] == 0, stats
        assert stats["skipped_shape"] == 0, stats
        assert stats["kv_skipped"] == 0, stats
        assert stats["kv_fused"] == n_kv == 2 * n_blocks, (stats["kv_fused"], n_kv, n_blocks)
        assert stats["skipped_moe_mlp"] == n_moe_mlp, stats
        assert stats["seeded_moe_blocks"] == 1, stats
        assert stats["seeded_shared_experts"] == 1, stats
        assert stats["zeroed_routed_fc2"] >= 1, stats
        # every dense tensor lands: a fused kv pair (2 HF tensors) counts once,
        # i.e. as the single dense kv tensor it becomes.
        assert stats["loaded"] == len(dense_state) - n_moe_mlp, (stats["loaded"], len(dense_state), n_moe_mlp)
        assert all(m.startswith("block4.1.mlp.") for m in stats["missing"]), stats["missing"]
        assert stats["unexpected"] == [], stats["unexpected"]

        err = _max_err(dense, moe)
        assert err < 1e-4, f"[{backend}] upcycled model differs from the dense one by {err:.3e}"


# --- 2. regression: the fill rule under the scratch recipe -------------------

def test_hf_upcycling_reproduces_the_dense_model_under_the_scratch_recipe():
    """mode hf_pretrained with the recipe left at its default (scratch) must
    resolve upcycle_init to routed_zero and reproduce the dense model."""
    for backend in ("tutel", "native"):
        dense, dense_cfg = _dense_model()
        hf_state = _to_hf_state(dense.state_dict())
        # exactly what tiny_config does — validate_config(merge_config(default_config(), ...))
        # — with mode hf_pretrained and NO recipe set (default: scratch)
        moe_cfg = tiny_config(mode="hf_pretrained",
                              model={"ablation": {**_NO_ROPE, **_MOE_LAST_S4},
                                     "moe": {**_backend(backend), "shared_expert": True}})
        assert moe_cfg["recipe"] == "scratch", moe_cfg["recipe"]
        assert moe_cfg["model"]["moe"]["upcycle_init"] == "routed_zero", \
            f"mode hf_pretrained under the scratch recipe resolved upcycle_init " \
            f"to {moe_cfg['model']['moe']['upcycle_init']!r}"
        moe = _moe_model(moe_cfg)
        with _serving(dense_cfg["model"]["depths"], dense_cfg["model"]["embed_dims"], hf_state):
            stats = load_hf_pretrained(moe, "stub/tiny", seed_moe_experts=True,
                                       upcycle_init=moe_cfg["model"]["moe"]["upcycle_init"],
                                       verbose=False)
        assert stats["seeded_moe_blocks"] == 1 and stats["zeroed_routed_fc2"] >= 1, stats
        err = _max_err(dense, moe)
        assert err < 1e-4, f"[{backend}] upcycled model differs from the dense one by {err:.3e}"


# --- 3. regression: both warm-start modes fill routed_zero -------------------

def _resolved(mode, **moe_over):
    over = {"mode": mode, "model": {"moe": {"shared_expert": True, **moe_over},
                                    "ablation": _MOE_LAST_S4}}
    if mode == "warm_start":
        over["ckpt_path"] = "/nonexistent/backbone.pt"     # only validated at load time
    return validate_config(merge_config(default_config(), over))


def test_config_fills_routed_zero_for_both_warm_start_modes():
    for mode in ("warm_start", "hf_pretrained"):
        cfg = _resolved(mode)
        assert cfg["recipe"] == "scratch"
        assert cfg["model"]["moe"]["upcycle_init"] == "routed_zero", \
            (mode, cfg["model"]["moe"]["upcycle_init"])
    assert _resolved("scratch")["model"]["moe"]["upcycle_init"] == "none"


# --- 4. regression: explicit "none" refused for hf_pretrained ----------------

def test_explicit_none_is_refused_for_hf_pretrained_with_a_shared_expert():
    for mode in ("hf_pretrained", "warm_start"):
        try:
            _resolved(mode, upcycle_init="none")
        except ValueError as e:
            assert mode in str(e) and "upcycle_init" in str(e), (mode, str(e))
        else:
            raise AssertionError(f"mode {mode} + shared expert + upcycle_init none must raise")
    # said on purpose: no seeding, or no shared expert to zero against
    over = {"mode": "hf_pretrained",
            "model": {"seed_moe_from_dense": False, "ablation": _MOE_LAST_S4,
                      "moe": {"shared_expert": True, "upcycle_init": "none"}}}
    cfg = validate_config(merge_config(default_config(), over))
    assert cfg["model"]["moe"]["upcycle_init"] == "none"
    cfg = _resolved("hf_pretrained", shared_expert=False, upcycle_init="none")
    assert cfg["model"]["moe"]["upcycle_init"] == "none"


# --- 5./6. tools/verify_upcycling.py -----------------------------------------

def _load_tool():
    """Import the script by path (it is a tool, not a package module)."""
    spec = importlib.util.spec_from_file_location("verify_upcycling", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_tool_refuses_an_hf_path_that_did_not_zero():
    tool = _load_tool()
    calls = []

    def fake_load(model, hf_id, seed_moe_experts=True, upcycle_init="none", verbose=True):
        calls.append((hf_id, seed_moe_experts, upcycle_init))
        return {"loaded": 999, "kv_fused": 8, "kv_skipped": 0, "skipped_moe_mlp": 6,
                "skipped_shape": 0, "dropped_no_target": 0, "unmapped": [],
                "seeded_moe_blocks": 1, "seeded_shared_experts": 1,
                "zeroed_routed_fc2": 0, "zeroed_shared_fc2": 0,
                "missing": [], "unexpected": []}

    args = argparse.Namespace(variant="b0", backend="native", experts=4, seed=0,
                              recipe="pretrained")
    real = tool.load_hf_pretrained
    tool.load_hf_pretrained = fake_load
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                tool.run_hf_path(args, torch.device("cpu"))
            except AssertionError as e:
                assert "zeroed_routed_fc2" in str(e), str(e)
            else:
                raise AssertionError("run_hf_path must refuse a load that zeroed nothing")
    finally:
        tool.load_hf_pretrained = real
    assert len(calls) == 2 and calls[1][1] is True and calls[1][2] != "none", calls
    assert "[config] recipe=pretrained mode=hf_pretrained upcycle_init=routed_zero" in buf.getvalue()


def test_verify_tool_requires_an_explicit_recipe():
    tool = _load_tool()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            tool.build_parser().parse_args(["--variant", "b1"])
    except SystemExit as e:
        assert e.code == 2, e.code
    else:
        raise AssertionError("--recipe must be required")
    args = tool.build_parser().parse_args(["--variant", "b1", "--recipe", "scratch"])
    assert args.recipe == "scratch" and args.variant == "b1"
    assert tool.build_parser().parse_args(["--recipe", "pretrained"]).recipe == "pretrained"

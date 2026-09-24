"""The sv2 router (Swin-MoE's settings): config checks and the Tutel wiring.

CI has no Tutel, and ``install_fake_tutel_backend`` replaces ``_build_tutel``
wholesale, so nothing else here ever runs the code that hands the router
policy to Tutel. These tests put a recording ``tutel.moe`` module in
``sys.modules`` instead and let the REAL ``_build_tutel`` call it. What they
cannot check is Tutel itself: ``tools/verify_upcycling.py`` on the training
machine builds the real layer, and Tutel raises on an unknown keyword.
"""

from __future__ import annotations

import sys
import types

from helpers import NATIVE_MOE, FakeTutelMoELayer, tiny_config

#: ``tutel/impls/moe_layer.py`` (microsoft/tutel main), verbatim:
#:   def __init__(self, gate_type, model_dim: int, experts=None,
#:       scan_expert_func=None, result_func=None, group=None, seeds=None,
#:       a2a_ffn_overlap_degree=1, is_postscore=True,
#:       batch_prioritized_routing=False, normalize_gate=True,
#:       is_gshard_loss=True, parallel_type='adaptive:1', use_2dh=False,
#:       **kwargs)
#: followed by ``for k in kwargs: raise Exception('Unrecognized argument ...')``.
TUTEL_MOE_LAYER_KWARGS = {
    "gate_type", "model_dim", "experts", "scan_expert_func", "result_func",
    "group", "seeds", "a2a_ffn_overlap_degree", "is_postscore",
    "batch_prioritized_routing", "normalize_gate", "is_gshard_loss",
    "parallel_type", "use_2dh",
}

_MOE_ON = {"ablation": {"use_moe": True, "moe_placement": [[], [], [], [1]]}}


def _build_with_recording_tutel(moe_cfg: dict) -> dict:
    """Run the real ``MoEMlp._build_tutel`` against a stand-in ``tutel.moe``;
    return the keyword arguments it passed to ``moe_layer``."""
    from pvt_moe.models.ffn import MoEMlp

    seen = {}

    def moe_layer(**kw):
        seen.update(kw)
        return FakeTutelMoELayer(kw["model_dim"], kw["experts"]["hidden_size_per_expert"],
                                 kw["experts"]["count_per_node"])

    fake_moe = types.ModuleType("tutel.moe")
    fake_moe.moe_layer = moe_layer
    fake_pkg = types.ModuleType("tutel")
    fake_pkg.moe = fake_moe
    saved = {k: sys.modules.get(k) for k in ("tutel", "tutel.moe")}
    sys.modules["tutel"], sys.modules["tutel.moe"] = fake_pkg, fake_moe
    try:
        MoEMlp(16, 32, moe_cfg=moe_cfg)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return seen


def test_defaults_reach_tutel_as_moe_layer_keywords():
    kw = _build_with_recording_tutel(tiny_config(model=_MOE_ON)["model"]["moe"])
    unknown = set(kw) - TUTEL_MOE_LAYER_KWARGS
    assert not unknown, f"Tutel's moe_layer raises on {unknown}"
    assert kw["batch_prioritized_routing"] is True
    assert kw["is_gshard_loss"] is False                  # load_importance
    assert kw["gate_type"]["capacity_factor"] == 1.25
    assert kw["gate_type"]["gate_noise"] == 1.0
    assert kw["gate_type"]["type"] == "top" and kw["gate_type"]["k"] == 1
    # the router policy is NOT a gate_type key (the gate raises on those)
    assert set(kw["gate_type"]) == {"type", "k", "capacity_factor", "gate_noise"}


def test_gshard_and_legacy_configs_reach_tutel_as_its_own_defaults():
    moe = tiny_config(model={**_MOE_ON, "moe": {"balance_loss": "gshard",
                                                "batch_prioritized_routing": False}})
    kw = _build_with_recording_tutel(moe["model"]["moe"])
    assert kw["is_gshard_loss"] is True and kw["batch_prioritized_routing"] is False
    # an sv1 checkpoint's saved config has neither key: it must rebuild the
    # router it trained with -- Tutel's defaults, gshard and token order.
    legacy = dict(moe["model"]["moe"])
    del legacy["balance_loss"], legacy["batch_prioritized_routing"]
    kw = _build_with_recording_tutel(legacy)
    assert kw["is_gshard_loss"] is True and kw["batch_prioritized_routing"] is False


def _refused(msg_part: str, **model):
    try:
        tiny_config(model=model)
    except ValueError as e:
        assert msg_part in str(e), str(e)
        return str(e)
    raise AssertionError(f"accepted: {model}")


def test_load_importance_refuses_zero_gate_noise_at_config_time():
    """Tutel asserts gate_noise > 0 inside load_importance_loss -- at the first
    forward, on the GPU. The validator says it at --dry-run instead."""
    msg = _refused("gate_noise > 0", **_MOE_ON, moe={"gate_noise": 0.0})
    assert "balance_loss=gshard" in msg
    # the escape hatch works, and a dense arm (no router built) is never refused
    tiny_config(model={**_MOE_ON, "moe": {"gate_noise": 0.0, "balance_loss": "gshard"}})
    tiny_config(model={"moe": {"gate_noise": 0.0}})


def test_native_backend_refuses_the_tutel_only_router():
    msg = _refused("native MoE backend", **_MOE_ON, moe={"backend": "native"})
    assert "balance_loss=gshard" in msg and "batch_prioritized_routing=false" in msg
    _refused("native MoE backend", **_MOE_ON,
             moe={**NATIVE_MOE, "batch_prioritized_routing": True})
    tiny_config(model={**_MOE_ON, "moe": dict(NATIVE_MOE)})     # the documented fix

    # and a hand-built moe_cfg that never went through validate_config
    from pvt_moe.models.ffn import MoEMlp

    cfg = {"backend": "native", "num_experts": 4, "top_k": 1, "capacity_factor": 1.0,
           "gate_noise": 0.0, "balance_loss": "load_importance"}
    try:
        MoEMlp(16, 32, moe_cfg=cfg)
    except ValueError as e:
        assert "native MoE backend" in str(e), e
    else:
        raise AssertionError("_build_native built an unsupported router")
    MoEMlp(16, 32, moe_cfg={**cfg, "balance_loss": "gshard"})   # today's native layer


def test_router_keys_are_type_checked():
    _refused("balance_loss", **_MOE_ON, moe={"balance_loss": "switch"})
    _refused("batch_prioritized_routing", **_MOE_ON, moe={"batch_prioritized_routing": 1})


def test_the_launch_banner_names_the_router():
    from pvt_moe.cli import build_config, build_parser, describe

    args = build_parser().parse_args(["--recipe", "scratch", "--ladder", "4"])
    text = describe(build_config(args, verbose=False))
    assert "cap 1.25 | noise 1.0 | bpr on | loss load_importance" in text, text

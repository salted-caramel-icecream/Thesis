"""``ablation.dwconv_off_placement``: strip the DWConv from NAMED dense blocks.

The dense control of a MoE arm. A no-shared-expert MoE block has no depthwise
conv, so the block-for-block dense mirror of ``moe-s4b2 + rope-s4b2`` is a
dense model with RoPE at s4b2 AND no conv at s4b2 -- something the global
``model.dense_dwconv`` (all 16 blocks or none) could not express. Everything
here is CPU, tiny configs, no Tutel: the arm is dense.
"""

from __future__ import annotations

import contextlib
import io

import torch

from helpers import NATIVE_MOE, tiny_config

from pvt_moe.config import default_config, merge_config, validate_config
from pvt_moe.models import build_model

_ROPE_S4 = {"use_rope": True, "rope_placement": [[], [], [], [-1]]}
_OFF_S4 = {"dwconv_off_placement": [[], [], [], [-1]]}


def _quiet_build(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_model(cfg)


def _dwconv_blocks(model):
    """The (stage, block) pairs whose dense Mlp carries a DWConv."""
    return {tuple(int(t) for t in k.split(".")[0][5:].split("_") if t) or k.split(".")[0]
            for k in model.state_dict() if ".mlp.dwconv." in k}


def test_the_named_block_loses_its_conv_and_its_neighbours_keep_theirs():
    torch.manual_seed(0)
    cfg = tiny_config(model={"ablation": {**_ROPE_S4, **_OFF_S4}})
    assert cfg["model"]["ablation"]["dwconv_off_placement"] == [[], [], [], [1]]  # resolved, depth 2
    model = _quiet_build(cfg)
    assert model.dwconv_off_placement == [[], [], [], [1]]
    keys = set(model.state_dict())
    conv_prefixes = {k.split(".mlp.dwconv.")[0] for k in keys if ".mlp.dwconv." in k}
    # every dense block has its conv except block4.1
    all_blocks = {f"block{s + 1}.{j}" for s, d in enumerate(cfg["model"]["depths"]) for j in range(d)}
    assert conv_prefixes == all_blocks - {"block4.1"}, sorted(all_blocks - conv_prefixes)
    assert "block4.1.attn.rope.freqs" in keys                  # and RoPE is there
    # the stripped block still has its dense FFN
    assert "block4.1.mlp.fc1.weight" in keys and "block4.1.mlp.fc2.weight" in keys
    # forward works: the Mlp must not ask for H, W it no longer needs
    logits, aux = model(torch.randn(2, 3, 64, 64))
    assert logits.shape[0] == 2 and aux is None


def test_unset_is_byte_identical_to_today_and_the_global_switch_still_strips_all():
    """The empty list is the off state: the same seed must give the same
    tensors, key for key. And model.dense_dwconv false keeps stripping every
    block (ladder rows 2 and 6 move nothing)."""
    torch.manual_seed(0)
    plain = _quiet_build(tiny_config())
    torch.manual_seed(0)
    explicit = _quiet_build(tiny_config(model={"ablation": {"dwconv_off_placement": [[], [], [], []]}}))
    a, b = plain.state_dict(), explicit.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    stripped = _quiet_build(tiny_config(model={"dense_dwconv": False}))
    assert not any(".mlp.dwconv." in k for k in stripped.state_dict())


def test_a_config_from_before_the_key_resolves_to_the_same_model_and_name():
    """evaluate.py re-validates a checkpoint's SAVED config with no defaults
    merge: an sv1/sv2 config has no dwconv_off_placement at all."""
    base = merge_config(default_config(), {"model": {"ablation": {"use_moe": False, "use_rope": False}}})
    fresh = validate_config(merge_config(default_config(), {"model": {"ablation": {"use_moe": False, "use_rope": False}}}))
    del base["model"]["ablation"]["dwconv_off_placement"]
    old = validate_config(base)
    assert old["run_name"] == fresh["run_name"] == "sv2_b1_in1k_r224_dense_norope_scratch90"
    assert old["model"]["ablation"]["dwconv_off_placement"] == [[], [], [], []]


def test_run_names_of_the_control_and_all_its_neighbours_are_distinct():
    """Plain dense + RoPE, global no-DWConv + RoPE, the per-block control, its
    stages-3+4 sibling, and the MoE arm it mirrors: five directories."""
    names = {
        "dense+rope": tiny_config(model={"ablation": _ROPE_S4})["run_name"],
        "global-nodw+rope": tiny_config(model={"dense_dwconv": False, "ablation": _ROPE_S4})["run_name"],
        "control-s4": tiny_config(model={"ablation": {**_ROPE_S4, **_OFF_S4}})["run_name"],
        "control-s34": tiny_config(model={"ablation": {"use_rope": True,
                                                       "rope_placement": [[], [], [-1], [-1]],
                                                       "dwconv_off_placement": [[], [], [-1], [-1]]}})["run_name"],
        "moe-s4": tiny_config(model={"moe": {**NATIVE_MOE, "shared_expert": False},
                                     "ablation": {**_ROPE_S4, "use_moe": True,
                                                  "moe_placement": [[], [], [], [-1]]}})["run_name"],
    }
    assert len(set(names.values())) == len(names), names
    assert names["control-s4"].endswith("_dense_rope-s4b1_nodw-s4b1_scratch4"), names["control-s4"]
    # tiny depths are [1, 1, 1, 2]: stage 3 has one block, so -1 there is the full stage "s3"
    assert names["control-s34"].endswith("_dense_rope-s3+s4b1_nodw-s3+s4b1_scratch4"), names["control-s34"]
    assert names["global-nodw+rope"].endswith("_dense_rope-s4b1_nodw_scratch4"), names["global-nodw+rope"]
    # a child warm-started from the control is still a DENSE parent (parent_tag
    # reads the "dense" fragment, which the token must not disturb)
    from pvt_moe.config import run_name_parts
    assert run_name_parts(tiny_config(model={"ablation": {**_ROPE_S4, **_OFF_S4}}))["moe"] == "dense"


def _refused(msg_part, **model):
    try:
        tiny_config(model=model)
    except ValueError as e:
        assert msg_part in str(e), str(e)
        return
    raise AssertionError(f"accepted: {model}")


def test_refusals_and_the_every_block_normalisation():
    # (a) stacking the list on the global switch: a no-op with a misleading name
    _refused("dense_dwconv is false", dense_dwconv=False, ablation=_OFF_S4)
    # (b) naming a routed block (only while use_moe: a dense arm keeps the
    #     default moe_placement around unused, and that must NOT refuse)
    _refused("also in moe_placement", moe=dict(NATIVE_MOE),
             ablation={"use_moe": True, "moe_placement": [[], [], [], [-1]], **_OFF_S4})
    ok = tiny_config(model={"ablation": {"use_moe": False, "moe_placement": [[], [], [], [-1]], **_OFF_S4}})
    assert ok["model"]["ablation"]["dwconv_off_placement"] == [[], [], [], [1]]
    # (c) every block named == the global arm: normalised to it, one directory
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        every = tiny_config(model={"ablation": {"dwconv_off_placement": [[0], [0], [0], [0, 1]]}})  # depths [1,1,1,2]
    assert every["model"]["dense_dwconv"] is False
    assert every["model"]["ablation"]["dwconv_off_placement"] == [[], [], [], []]
    assert every["run_name"] == tiny_config(model={"dense_dwconv": False})["run_name"]
    assert "names every block" in buf.getvalue()
    # (d) out of range is the placement resolver's own error
    _refused("out of range", ablation={"dwconv_off_placement": [[], [], [], [2]]})


def test_the_cli_flag_round_trips_and_names_the_s4b2_control_under_b2():
    from pvt_moe.cli import build_config, build_parser, describe

    def _cli(*argv):
        return build_config(build_parser().parse_args(list(argv)), verbose=False)

    c = _cli("--no-moe", "--dwconv-off-placement", "[[],[],[],[-1]]")
    assert c["model"]["ablation"]["dwconv_off_placement"] == [[], [], [], [1]]     # b1: resolved
    ctl = _cli("--recipe", "scratch", "--ladder", "1", "--variant", "b2", "--rope",
               "--rope-placement", "[[],[],[],[-1]]", "--dwconv-off-placement", "[[],[],[],[-1]]")
    assert ctl["run_name"] == "sv2_b2_in1k_r224_dense_rope-s4b2_nodw-s4b2_scratch90"
    assert ctl["model"]["ablation"]["dwconv_off_placement"] == [[], [], [], [2]]
    assert ctl["model"]["dense_dwconv"] is True and not ctl["model"]["ablation"]["use_moe"]
    assert "(off at [[], [], [], [2]])" in describe(ctl)
    # bad JSON is a clean CLI error naming the flag, like the other placement flags
    try:
        _cli("--dwconv-off-placement", "[[],[],[],[-1]")
    except SystemExit as e:
        assert "--dwconv-off-placement must be JSON" in str(e.code), e.code
    else:
        raise AssertionError("malformed JSON must be a SystemExit")


def test_results_identity_records_it():
    from pvt_moe.engine.results import run_identity

    assert run_identity(tiny_config(model={"ablation": {**_ROPE_S4, **_OFF_S4}}))["dwconv_off_placement"] == [[], [], [], [1]]
    assert run_identity(tiny_config())["dwconv_off_placement"] == [[], [], [], []]


def test_warm_start_check_sees_a_conv_mismatch_in_both_directions():
    """Before this key the architecture check compared RoPE and MoE placement
    and nothing conv-related: a conv-everywhere parent loaded into a stripped
    child silently, with the child's block4.1 conv 'dropped_no_target'."""
    from pvt_moe.models.pretrained import check_backbone_architecture

    parent = tiny_config(model={"ablation": _ROPE_S4})
    child = tiny_config(model={"ablation": {**_ROPE_S4, **_OFF_S4}})
    keys = list(_quiet_build(parent).state_dict())
    down = check_backbone_architecture(parent, child, keys, "parent.ckpt")
    up = check_backbone_architecture(child, parent, list(_quiet_build(child).state_dict()), "child.ckpt")
    assert any("without DWConv" in p for p in down), down
    assert any("without DWConv" in p for p in up), up
    assert check_backbone_architecture(child, child, keys, "same.ckpt") == []
    # the global arm reads as "every block", so it differs from the control too
    glob = tiny_config(model={"dense_dwconv": False, "ablation": _ROPE_S4})
    assert any("without DWConv" in p for p in check_backbone_architecture(glob, child, keys, "g.ckpt"))
    # and a parent config from before the key (no dense_dwconv either) is not compared
    legacy = {"model": {k: v for k, v in parent["model"].items() if k != "dense_dwconv"}}
    legacy["model"]["ablation"] = {k: v for k, v in parent["model"]["ablation"].items()
                                   if k != "dwconv_off_placement"}
    assert not any("without DWConv" in p for p in check_backbone_architecture(legacy, child, keys, "l.ckpt"))

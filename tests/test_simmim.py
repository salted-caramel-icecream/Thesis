"""SimMIM pretraining (pvt_moe.ssl.simmim) against the reference semantics.

- mask generator: ceil(N * 0.6) 32-px patches per image on the (img/32)^2 grid,
  expanded x8 onto the stride-4 token grid (microsoft/SimMIM data_simmim.py);
- token space leaks a 3-px band through PVT v2's overlapping 7x7/stride-4
  embed and pixel space does not — measured, not assumed;
- loss = masked-only L1 / (mask.sum() + 1e-5) / in_chans on a full-resolution
  reconstruction from a 1x1 conv + PixelShuffle(32) head;
- optimiser groups / schedule; MoE pretraining adds the aux loss and the
  routing diagnostic counts the masked positions;
- the backbone round trip: path 3 (dense -> upcycled) and path 2 (MoE ->
  straight load) both carry the chain; run names keep the paths apart;
- a 1-epoch fit through build_ssl_trainer leaves results.json + backbone.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import pathlib
import tempfile
import types

import numpy as np
import torch
from PIL import Image as PILImage

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.config import default_config, merge_config, parent_tag, validate_config
from pvt_moe.engine.classifier import LitClassifier
from pvt_moe.models.ffn import MoEMlp
from pvt_moe.models.pretrained import load_backbone_checkpoint
from pvt_moe.models.pvt import build_model
from pvt_moe.ssl import LitSimMIM, SimMIMMaskGenerator, backbone_filename, build_ssl_module, mask_token_routing

ROPE = {"use_rope": True, "rope_mode": "mixed", "rope_last_n_stages": 1}
MOE = {"use_moe": True, "moe_placement": [[], [], [], [-1]]}


def _cfg(img=64, **over):
    base = {"task": "ssl", "dataset": {"name": "pass", "img_size": img},
            "model": {"pretrained_hf_id": None}}
    return tiny_config(**merge_config(base, over))


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# --- mask generator ------------------------------------------------------------

def test_mask_generator_matches_the_reference_counts():
    gen = SimMIMMaskGenerator(224, 32, 4, 0.6)
    assert (gen.rand_size, gen.scale, gen.token_count, gen.mask_count) == (7, 8, 49, 30)
    assert gen.mask_count == math.ceil(49 * 0.6)
    g = torch.Generator().manual_seed(0)
    patch, token = gen(4, generator=g)
    assert patch.shape == (4, 7, 7) and token.shape == (4, 56 * 56)
    assert patch.dtype == torch.bool and token.dtype == torch.bool
    assert (patch.sum(dim=(1, 2)) == 30).all()                       # exactly 30 per image
    assert torch.allclose(token.float().mean(dim=1), torch.full((4,), 30 / 49))
    up = patch.repeat_interleave(8, 1).repeat_interleave(8, 2).reshape(4, -1)
    assert torch.equal(up, token)                                    # x8 expansion, no drift
    assert not torch.equal(patch[0], patch[1])                       # independent per image
    patch2, _ = gen(4, generator=torch.Generator().manual_seed(0))
    assert torch.equal(patch, patch2)                                # seeded => reproducible
    pm = gen.pixel_mask(patch)
    assert pm.shape == (4, 1, 224, 224) and math.isclose(pm.mean().item(), 30 / 49, rel_tol=1e-6)
    for bad in ((100, 32), (224, 33)):
        try:
            SimMIMMaskGenerator(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} must be rejected")


# --- token vs pixel mask space: the leak, measured --------------------------------

def test_token_space_leaks_a_3px_band_and_pixel_space_does_not():
    """One masked 32-px patch in the middle of a 96-px image (3x3 patch grid).
    Stage-1 token o covers input pixels [4o-3, 4o+3]; the patch covers
    [32, 63], so of the VISIBLE tokens only row/col 16 (covering 61..67)
    sees masked pixels: 17 tokens, all on the bottom / right edge."""
    outs = {}
    for space in ("token", "pixel"):
        lit = _quiet(LitSimMIM, _cfg(img=96, ssl={"mask_space": space}))
        lit.eval()
        patch = torch.zeros(1, 3, 3, dtype=torch.bool)
        patch[0, 1, 1] = True
        x = torch.randn(1, 3, 96, 96)
        y = x.clone()
        pm = lit.mask_gen.pixel_mask(patch).bool()
        y[pm.expand_as(y)] = torch.randn(int(pm.sum()) * 3)          # differs ONLY inside the patch
        with torch.no_grad():
            ex, _, _ = lit.encoder.patch_embed1(lit.encoder_input(x, patch))
            ey, _, _ = lit.encoder.patch_embed1(lit.encoder_input(y, patch))
        diff = (ex - ey).abs().amax(-1).view(24, 24)                 # per token, 96/4 grid
        token_masked = torch.zeros(24, 24, dtype=torch.bool)
        token_masked[8:16, 8:16] = True
        visible_changed = (diff > 1e-6) & ~token_masked
        outs[space] = visible_changed
    band = torch.zeros(24, 24, dtype=torch.bool)
    band[16, 8:17] = True
    band[8:16, 16] = True
    assert torch.equal(outs["token"], band), outs["token"].nonzero().tolist()
    assert int(outs["token"].sum()) == 17
    assert not outs["pixel"].any(), outs["pixel"].nonzero().tolist()
    # in pixel space the embed input is literally zero under the patch
    lit_px = _quiet(LitSimMIM, _cfg(img=96, ssl={"mask_space": "pixel"}))
    xin = lit_px.encoder_input(x, patch)
    assert (xin[pm.expand_as(xin)] == 0).all() and torch.equal(xin[~pm.expand_as(xin)], x[~pm.expand_as(x)])


# --- loss / head ------------------------------------------------------------------

def test_loss_is_masked_only_l1_over_channels_on_a_full_resolution_reconstruction():
    lit = _quiet(LitSimMIM, _cfg(img=64))
    lit.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = lit.masked_forward(x, generator=torch.Generator().manual_seed(1))
    x_rec, patch = out["x_rec"], out["patch_mask"]
    assert x_rec.shape == x.shape
    m = lit.mask_gen.pixel_mask(patch)
    ref = ((x - x_rec).abs() * m).sum() / (m.sum() + 1e-5) / 3
    assert torch.allclose(out["recon"], ref)
    assert torch.equal(out["loss"], out["recon"]) and out["aux"].item() == 0.0   # dense: no aux
    # visible pixels never enter the loss
    x_rec2 = x_rec + 10.0 * (1 - m)
    ref2 = ((x - x_rec2).abs() * m).sum() / (m.sum() + 1e-5) / 3
    assert torch.allclose(ref2, ref)
    conv, shuffle = lit.head
    assert isinstance(conv, torch.nn.Conv2d) and conv.kernel_size == (1, 1)
    assert conv.out_channels == 32 * 32 * 3 and lit.encoder_stride == 32
    assert isinstance(shuffle, torch.nn.PixelShuffle) and shuffle.upscale_factor == 32
    # gradients reach the encoder, the mask token and the head
    lit.train()
    loss = lit.masked_forward(x)["loss"]
    loss.backward()
    assert lit.mask_token.grad is not None and lit.mask_token.grad.abs().sum() > 0
    assert lit.encoder.patch_embed1.proj.weight.grad is not None
    assert lit.head[0].weight.grad is not None


def test_sanity_step_reports_floats_and_drop_path_is_zero_for_pretraining():
    # tiny_config pins drop_path 0.1 for the supervised tests; the DEFAULT for
    # an SSL run (drop_path_rate None) is SimMIM's pretraining value 0.0.
    ssl_default = validate_config(merge_config(default_config(), {"task": "ssl", "dataset": {"name": "pass"},
                                                                  "use_wandb": False}))
    assert ssl_default["model"]["drop_path_rate"] == 0.0
    sup = validate_config(merge_config(default_config(), {"use_wandb": False}))
    assert sup["model"]["drop_path_rate"] == 0.1                     # supervised rule untouched
    cfg = _cfg(img=64)
    lit = _quiet(LitSimMIM, cfg)
    out = lit.sanity_step(torch.randn(2, 3, 64, 64))
    assert set(out) >= {"loss", "recon", "aux", "mask_ratio", "x_rec_shape"}
    assert out["x_rec_shape"] == (2, 3, 64, 64) and math.isfinite(out["loss"])


# --- optimiser / schedule ---------------------------------------------------------

def test_optimizer_groups_and_per_step_schedule():
    cfg = _cfg(img=64, batch_size=4, effective_batch_size=8, model={"ablation": ROPE},
               ssl={"epochs": 10, "warmup_epochs": 2})
    lit = _quiet(LitSimMIM, cfg)
    lit.trainer = types.SimpleNamespace(estimated_stepping_batches=100)
    out = _quiet(lit.configure_optimizers)
    opt = out["optimizer"]
    groups = {g["name"]: g for g in opt.param_groups}
    assert set(groups) == {"decay", "no_decay"}
    names = {id(p): n for n, p in lit.named_parameters()}
    nd = {names[id(p)] for p in groups["no_decay"]["params"]}
    assert "mask_token" in nd and any(n.endswith("rope.freqs") for n in nd)
    assert all(p.ndim <= 1 or names[id(p)].endswith(".bias") or names[id(p)] in lit.no_weight_decay()
               for p in groups["no_decay"]["params"])
    assert all(p.ndim > 1 for p in groups["decay"]["params"])
    assert groups["decay"]["weight_decay"] == 0.05 and groups["no_decay"]["weight_decay"] == 0.0
    assert opt.defaults["betas"] == (0.9, 0.999)
    s = cfg["ssl"]
    assert s["lr"] == 2e-4 * 8 / 512 and s["warmup_lr"] == 1e-6 * 8 / 512 and s["final_lr"] == 1e-5 * 8 / 512
    sched = out["lr_scheduler"]["scheduler"]
    lam = sched.lr_lambdas[0]
    warm = 20                                                        # 100 steps * 2 / 10
    assert math.isclose(lam(0) * s["lr"], s["warmup_lr"])
    assert math.isclose(lam(warm) * s["lr"], s["lr"])
    assert math.isclose(lam(100) * s["lr"], s["final_lr"], rel_tol=1e-6)
    assert lam(warm) > lam(60) > lam(100)                            # cosine, monotone


# --- MoE pretraining (path 2) -----------------------------------------------------

def test_moe_pretraining_adds_aux_loss_and_routing_counts_masked_positions():
    undo = install_fake_tutel_backend()
    try:
        lit = _quiet(LitSimMIM, _cfg(img=64, model={"ablation": {**ROPE, **MOE}}))
        assert sum(isinstance(m, MoEMlp) for m in lit.encoder.modules()) == 1
        x = torch.randn(3, 3, 64, 64)
        out = lit.sanity_step(x)
        assert out["aux"] > 0 and math.isclose(out["loss"], out["recon"] + 0.01 * out["aux"], rel_tol=1e-5)
        stats = _quiet(mask_token_routing, lit, x, torch.Generator().manual_seed(0))
        (name, s), = stats.items()
        assert name == "block4.1.mlp" and s["stage"] == 4 and s["num_experts"] == 4
        assert s["tokens_total"] == 3 * 4                             # 2x2 stage-4 grid x 3 images
        assert s["tokens_masked"] + s["tokens_visible"] == s["tokens_total"]
        assert s["tokens_masked"] == 3 * 3                            # 3 of 4 patches masked
        assert s["aux_counts_mask_tokens"] is True
        assert math.isclose(sum(s["masked_share"]), 1.0, abs_tol=1e-3)
        assert 0 <= s["share_gap"] <= 1 and s["masked_entropy"] <= s["max_entropy"] + 1e-6
        extra = lit.results_extra()
        assert extra["ssl"]["method"] == "simmim" and "EXPECTED to be low" in extra["ssl"]["note"]
    finally:
        undo()


# --- backbone round trip, chain, run names ------------------------------------------

def _labelled_cfg(ckpt, **model):
    over = {"pretrained_hf_id": None, "ablation": {**ROPE, **MOE}}
    over.update(model)
    return tiny_config(recipe="ssl_finetune", mode="ssl_init", ckpt_path=ckpt, epochs=2,
                       dataset={"img_size": 64}, model=over)


def test_backbone_round_trip_paths_2_and_3_carry_the_chain():
    undo = install_fake_tutel_backend()
    try:
        with tempfile.TemporaryDirectory() as d:
            # path 3: dense SimMIM encoder -> MoE fine-tune upcycles it
            dense = _quiet(LitSimMIM, _cfg(img=64, model={"ablation": ROPE}))
            p3 = os.path.join(d, "sv1_custom_pass_r64_dense_rope-s4_simmim200", "simmim_backbone.pt")
            os.makedirs(os.path.dirname(p3))
            # A real SSL run leaves results.json here too (build_ssl_trainer
            # installs ResultsWriter); parent_tag reads the lineage from it.
            _write_results(os.path.dirname(p3), "dense", "simmim200")
            _quiet(dense.save_backbone, p3)
            ck = torch.load(p3, map_location="cpu", weights_only=False)
            assert ck["method"] == "simmim" and ck["cfg"]["chain"] == ["simmim_pretrain@pass_r64"]
            assert not any(".moe_layer." in k for k in ck["state_dict"])
            cfg = _labelled_cfg(p3)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                lit = LitClassifier(cfg)
            assert lit.cfg["chain"] == ["simmim_pretrain@pass_r64", "ssl_finetune+moe@imagenet-1k_r64"]
            assert lit.hparams["cfg"]["chain"] == lit.cfg["chain"]      # saved AFTER the warm start
            assert "seeded_moe_blocks=1" in buf.getvalue() and "zeroed_routed_fc2=2" in buf.getvalue()
            assert cfg["run_name"].endswith("_sslft2_from-dense-simmim200")
            # path 2: MoE SimMIM encoder -> MoE fine-tune loads it as trained
            moe = _quiet(LitSimMIM, _cfg(img=64, model={"ablation": {**ROPE, **MOE}}))
            p2 = os.path.join(d, "sv1_custom_pass_r64_moe-s4b1-e4k1+sh_rope-s4_simmim200",
                              "simmim_backbone.pt")
            os.makedirs(os.path.dirname(p2))
            _write_results(os.path.dirname(p2), "moe", "simmim200")
            _quiet(moe.save_backbone, p2)
            ck = torch.load(p2, map_location="cpu", weights_only=False)
            assert ck["cfg"]["chain"] == ["simmim_pretrain+moe@pass_r64"]
            cfg2 = _labelled_cfg(p2)
            model = build_model(cfg2)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats = load_backbone_checkpoint(model, p2, expected_cfg=cfg2)
            assert stats["seeded_moe_blocks"] == 0 and stats["dense_mlp_for_seeding"] == 0
            assert stats["parent_chain"] == ["simmim_pretrain+moe@pass_r64"] and stats["parent_method"] == "simmim"
            assert "nothing to upcycle" in buf.getvalue()
            src = moe.encoder.state_dict()
            for k, v in model.state_dict().items():
                if ".mlp." in k and "block4.1" in k:
                    assert torch.equal(v, src[k]), k
            assert cfg2["run_name"].endswith("_sslft2_from-moe-simmim200")
            assert cfg["run_name"] != cfg2["run_name"]
    finally:
        undo()


def _write_results(dirpath, moe, budget):
    """A minimal parent results.json: only the two fragments parent_tag reads."""
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"identity": {"name_moe": moe, "name_budget": budget}}, fh)
    return os.path.join(dirpath, "last.ckpt")


def test_parent_tag_reads_the_lineage_from_the_parents_results_json():
    """parent_tag reads tokens the parent RECORDED, never its directory name.

    The tag used to be recovered by splitting the parent directory on "_" and
    hunting for the norm segment, which meant any change to the run-name
    format silently broke lineage. It now reads identity.name_moe /
    identity.name_budget out of results.json, which build_run_tag wrote.
    """
    with tempfile.TemporaryDirectory() as d:
        ckpt = _write_results(os.path.join(d, "a_directory_name_nobody_parses"),
                              "dense", "simmim200")
        assert parent_tag(ckpt) == "from-dense-simmim200"

        ckpt = _write_results(os.path.join(d, "whatever"), "moe",
                              "sslft100-from-dense-simmim200")
        assert parent_tag(ckpt) == "from-moe-sslft100-from-dense-simmim200"

        # No results.json, an unreadable one, and no checkpoint at all: None,
        # which validate_config turns into a "pass --run-name" warning.
        assert parent_tag(os.path.join(d, "empty", "last.ckpt")) is None
        assert parent_tag(None) is None
        bad = os.path.join(d, "corrupt")
        os.makedirs(bad)
        pathlib.Path(bad, "results.json").write_text("{not json")
        assert parent_tag(os.path.join(bad, "last.ckpt")) is None
        # Present but missing the fields (a results.json from before this change)
        partial = os.path.join(d, "partial")
        os.makedirs(partial)
        pathlib.Path(partial, "results.json").write_text('{"identity": {"seed": 42}}')
        assert parent_tag(os.path.join(partial, "last.ckpt")) is None


def test_a_warm_start_without_a_parent_tag_warns_and_keeps_the_bare_budget():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c = validate_config(merge_config(default_config(), {"recipe": "ssl_finetune", "ckpt_path": "/x/b.pt",
                                                            "use_wandb": False}))
    assert "carries no parent tag" in buf.getvalue() and c["run_name"].endswith("_sslft100")
    # resume keeps the original name: no parent suffix
    with tempfile.TemporaryDirectory() as d:
        ckpt = _write_results(os.path.join(d, "parent"), "dense", "simmim200")
        c = validate_config(merge_config(default_config(), {"mode": "resume", "use_wandb": False,
                                                            "ckpt_path": ckpt}))
    assert "from-" not in c["run_name"]


# --- a real fit ------------------------------------------------------------------

def _pass_snapshot(root, n=8, size=64):
    from datasets import Dataset, DatasetDict, Features, Image

    rng = np.random.default_rng(0)
    imgs = [PILImage.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)) for _ in range(n)]
    DatasetDict({"train": Dataset.from_dict({"image": imgs}, features=Features({"image": Image()}))}).save_to_disk(root)
    return root


def test_one_epoch_simmim_fit_leaves_results_backbone_and_rope_snapshots():
    from pvt_moe.data import build_dataloaders
    from pvt_moe.engine.callbacks import build_ssl_trainer
    from pvt_moe.engine.results import read_results

    with tempfile.TemporaryDirectory() as d:
        root = _pass_snapshot(os.path.join(d, "pass_arrow"))
        cfg = _cfg(img=64, dataset={"arrow_dirs": {"pass": root}}, model={"ablation": ROPE},
                   num_workers=0, checkpoint_root=d, log_root=d, use_wandb=False, use_tensorboard=False,
                   batch_size=4, effective_batch_size=4, ssl={"epochs": 1, "warmup_epochs": 1})
        with contextlib.redirect_stdout(io.StringIO()):
            train_loader, val_loader = build_dataloaders(cfg)
            assert val_loader is None
            module = build_ssl_module(cfg)
            assert isinstance(module, LitSimMIM)
            trainer = build_ssl_trainer(cfg)
            trainer.fit(module, train_loader)
            run_dir = os.path.join(d, cfg["run_name"])
            module.save_backbone(os.path.join(run_dir, backbone_filename("simmim")))
        files = set(os.listdir(run_dir))
        assert {"last.ckpt", "results.json", "results.md", "simmim_backbone.pt",
                "rope_freqs_init.pt", "rope_freqs_final.pt"} <= files, files
        rec = read_results(run_dir)
        assert rec["identity"]["chain"] == ["simmim_pretrain@pass_r64"]
        assert rec["identity"]["ssl"]["mask_space"] == "token" and rec["status"]["finished"] is True
        assert rec["ssl"]["ssl_loss"] is not None and "EXPECTED to be low" in rec["ssl"]["note"]
        assert rec["history"][0]["epoch"] == 1 and rec["efficiency"]["images_per_second"] > 0
        assert rec["environment"]["torch"] == torch.__version__
        md = open(os.path.join(run_dir, "results.md")).read()
        assert "simmim_pretrain@pass_r64" in md and "## SSL pretraining" in md

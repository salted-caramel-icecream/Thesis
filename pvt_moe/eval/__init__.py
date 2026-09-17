"""Evaluation of trained backbones: validation, k-NN, linear probe, low-shot.

Entry point: ``evaluate.py`` at the repo root (a thin front end over
``pvt_moe.eval.runner``); ``python -m pvt_moe.eval.lowshot`` writes the
seeded low-shot subsets. Submodules are imported explicitly
(``pvt_moe.eval.probe``, ``.knn``, ``.features``, ``.lowshot``) so that the
torch-free ones stay torch-free for download_data.py.
"""

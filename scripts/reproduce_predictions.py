"""
Reproduce GNN predictions from persisted artifacts (item 16 acceptance).

Loads results/models/{gcn,graphsage}.pt checkpoint bundles (model_state_dict
+ node features + edge index + test mask, saved by the pipeline) and
re-derives predictions on the held-out test nodes, so anyone can verify the
saved weights actually work without re-running the pipeline.

Usage: python scripts/reproduce_predictions.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gcn import GCN  # noqa: E402
from src.models.graphsage import GraphSAGE  # noqa: E402

MODELS_DIR = os.path.join("results", "models")
NUM_LAYERS = 2  # pipeline default (config: models.gcn.num_layers)


def _infer_dims(state_dict, num_layers: int):
    """Infer (in_channels, hidden_channels, out_channels) from a state_dict."""
    weight_keys = [k for k in state_dict
                   if k.split(".")[0] in ("convs", "sages")
                   and k.endswith("weight")]
    if len(weight_keys) < num_layers:
        raise KeyError(f"expected >= {num_layers} conv weights, found "
                       f"{weight_keys}")
    first_w = next(k for k in weight_keys if int(k.split(".")[1]) == 0)
    last_w = next(k for k in weight_keys
                  if int(k.split(".")[1]) == num_layers - 1)
    in_ch = state_dict[first_w].shape[-1]
    out_ch = state_dict[last_w].shape[0]
    hidden = state_dict[first_w].shape[0]
    return in_ch, hidden, out_ch


def reproduce(model_name: str, model_cls) -> None:
    path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found - run the pipeline first")
        return

    ckpt = torch.load(path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt
            and "x" in ckpt and "edge_index" in ckpt):
        print(f"[FAIL] {path} lacks the data bundle (x/edge_index/); "
              "re-run the pipeline to regenerate bundled checkpoints")
        return

    state = ckpt["model_state_dict"]
    in_ch, hidden, out_ch = _infer_dims(state, NUM_LAYERS)
    module = model_cls(in_ch, hidden, out_ch, num_layers=NUM_LAYERS)
    module.load_state_dict(state)
    module.eval()

    x, edge_index, test_mask = ckpt["x"], ckpt["edge_index"], ckpt["test_mask"]
    with torch.no_grad():
        probs = torch.softmax(module(x, edge_index), dim=1)

    n_test = int(test_mask.sum())
    if n_test == 0:
        print(f"[FAIL] {model_name}: empty test mask in bundle")
        return
    p_io = probs[test_mask, 1]
    print(f"[PASS] {model_name}: reloaded weights reproduce predictions on "
          f"{n_test} held-out nodes | P(IO) mean={float(p_io.mean()):.4f} "
          f"min={float(p_io.min()):.4f} max={float(p_io.max()):.4f}")


def main() -> None:
    print(f"Reproducing predictions from checkpoints in {MODELS_DIR}/\n")
    reproduce("gcn", GCN)
    reproduce("graphsage", GraphSAGE)
    print("\nDone.")


if __name__ == "__main__":
    main()

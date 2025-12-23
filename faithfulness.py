

"""Faithfulness tests (Deletion/Insertion) for co-attention explanations.

This module implements:
- Deletion tests: progressively mask top-k residues/atoms and measure prediction degradation.
- Insertion tests: start from fully masked baseline and progressively add top-k residues/atoms.

Designed to be model-agnostic:
- You provide callables that run your model and optionally return attention matrices.
- Masking is done on embeddings + masks (recommended, avoids tokenizer/chemistry confounds).

Typical usage pattern:

    scorer = FaithfulnessScorer(
        forward_fn=my_forward_fn,                 # required
        forward_with_attn_fn=my_forward_attn_fn,   # required if you want attention-based ranking
    )

    result = scorer.evaluate_dataset(
        samples=test_samples,
        mode="protein",  # or "drug"
        test_type="deletion",  # or "insertion"
        fractions=[0, .05, .1, .2, .3, .4, .5],
        n_random=5,
    )

You can then plot result.mean_ours / mean_rand vs fractions and report AUDC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import copy
import math

import numpy as np

import torch
from torch import Tensor
from tqdm import tqdm


Mode = Literal["protein", "drug"]
TestType = Literal["deletion", "insertion"]
RankSource = Literal["coattn", "random", "provided"]


@dataclass
class Sample:
    """Container for a single DTI example.

    You can extend/replace this in your code; FaithfulnessScorer only relies on the attributes.

    Required:
        protein_emb: (Lp, Dp) float
        drug_emb: (Ld, Dd) float
        protein_mask: (Lp,) 1=keep, 0=masked
        drug_mask: (Ld,) 1=keep, 0=masked

    Optional:
        meta: any extra info (ids, names)
    """

    protein_emb: "Tensor"
    drug_emb: "Tensor"
    protein_mask: "Tensor"
    drug_mask: "Tensor"
    meta: Optional[Dict] = None


@dataclass
class CurveResult:
    fractions: np.ndarray
    mean_ours: np.ndarray
    std_ours: np.ndarray
    mean_rand: np.ndarray
    std_rand: np.ndarray
    audc_ours: float
    audc_rand: float


def _to_numpy(x: "Tensor | np.ndarray | float") -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _trapz_auc(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal area under curve."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size != x.size:
        raise ValueError(f"AUC expects y and x same size; got {y.size} vs {x.size}")
    return float(np.trapz(y, x))


def _argsort_desc(score: np.ndarray) -> np.ndarray:
    return np.argsort(-score)


def coattention_inter_map(attn_p: np.ndarray, attn_d: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute symmetric co-attention interaction map.

    Assumes:
        attn_p: (Lp, Ld) protein→drug
        attn_d: (Ld, Lp) drug→protein

    Returns:
        inter_map: (Lp, Ld)

    Default is geometric mean of mutual attentions, which is common and numerically stable.
    """
    ap = np.asarray(attn_p, dtype=float)
    ad = np.asarray(attn_d, dtype=float)
    if ap.ndim != 2 or ad.ndim != 2:
        raise ValueError(f"attn_p/attn_d must be 2D; got {ap.shape} and {ad.shape}")
    if ap.shape[0] != ad.shape[1] or ap.shape[1] != ad.shape[0]:
        raise ValueError(
            "attn shapes inconsistent: expected attn_p (Lp,Ld) and attn_d (Ld,Lp); "
            f"got {ap.shape} and {ad.shape}"
        )
    inter = np.sqrt(np.maximum(ap * ad.T, 0.0) + eps)
    return inter


def rank_from_attention(attn_p: np.ndarray, attn_d: np.ndarray, mode: Mode) -> np.ndarray:
    """Return ranked indices for protein residues or drug atoms based on co-attention."""
    inter = coattention_inter_map(attn_p, attn_d)
    if mode == "protein":
        score = inter.sum(axis=1)  # (Lp,)
    elif mode == "drug":
        score = inter.sum(axis=0)  # (Ld,)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return _argsort_desc(score)


def _clone_sample(sample: Sample) -> Sample:
    # Deepcopy works, but can be expensive; do a light clone for tensors.
    return Sample(
        protein_emb=sample.protein_emb.clone(),
        drug_emb=sample.drug_emb.clone(),
        protein_mask=sample.protein_mask.clone(),
        drug_mask=sample.drug_mask.clone(),
        meta=copy.deepcopy(sample.meta),
    )


def _mask_indices(
    emb: "Tensor",
    mask: "Tensor",
    indices: Sequence[int],
    mask_embedding_value: float = 0.0,
) -> Tuple["Tensor", "Tensor"]:
    """Mask a set of positions by zeroing embeddings and setting mask to 0."""
    if len(indices) == 0:
        return emb, mask
    idx = torch.as_tensor(indices, device=emb.device, dtype=torch.long)
    mask[idx] = 0
    emb[idx, :] = mask_embedding_value
    return emb, mask


def _insert_indices(
    emb_base: "Tensor",
    mask_base: "Tensor",
    emb_full: "Tensor",
    indices: Sequence[int],
) -> Tuple["Tensor", "Tensor"]:
    """Insert (unmask) a set of positions from full embeddings into baseline."""
    if len(indices) == 0:
        return emb_base, mask_base
    idx = torch.as_tensor(indices, device=emb_base.device, dtype=torch.long)
    mask_base[idx] = 1
    emb_base[idx, :] = emb_full[idx, :]
    return emb_base, mask_base


class FaithfulnessScorer:
    """Runs faithfulness curves for a model.

    You must provide:
        forward_fn(sample) -> scalar score (preferably a logit for positive class)

    If you want attention-based ranking:
        forward_with_attn_fn(sample) -> (score, attn_p, attn_d)
            where attn_p is (Lp,Ld) and attn_d is (Ld,Lp)

    Notes:
    - We recommend scoring with a *logit* rather than probability for smoother curves.
    - Masking is done on embedding tensors and masks.
    """

    def __init__(
        self,
        forward_fn: Callable[[Sample], "Tensor"],
        forward_with_attn_fn: Optional[Callable[[Sample], Tuple["Tensor", np.ndarray, np.ndarray]]] = None,
        device: Optional[str] = None,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for faithfulness.py")
        self.forward_fn = forward_fn
        self.forward_with_attn_fn = forward_with_attn_fn
        self.device = device

    def _ensure_device(self, sample: Sample) -> Sample:
        if self.device is None:
            return sample
        dev = torch.device(self.device)
        return Sample(
            protein_emb=sample.protein_emb.to(dev),
            drug_emb=sample.drug_emb.to(dev),
            protein_mask=sample.protein_mask.to(dev),
            drug_mask=sample.drug_mask.to(dev),
            meta=sample.meta,
        )

    @torch.no_grad()
    def _score(self, sample: Sample) -> float:
        s = self.forward_fn(sample)
        if isinstance(s, torch.Tensor):
            s = s.squeeze()
            return float(s.detach().cpu().item())
        return float(s)

    @torch.no_grad()
    def _score_and_attn(self, sample: Sample) -> Tuple[float, np.ndarray, np.ndarray]:
        if self.forward_with_attn_fn is None:
            raise ValueError("forward_with_attn_fn is required for attention-based ranking")
        s, ap, ad = self.forward_with_attn_fn(sample)
        if isinstance(s, torch.Tensor):
            s = s.squeeze().detach().cpu().item()
        return float(s), np.asarray(ap), np.asarray(ad)

    def _get_ranked_indices(
        self,
        sample: Sample,
        mode: Mode,
        rank_source: RankSource,
        provided_rank: Optional[Sequence[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        L = int(sample.protein_mask.numel() if mode == "protein" else sample.drug_mask.numel())

        if rank_source == "provided":
            if provided_rank is None:
                raise ValueError("provided_rank must be set when rank_source='provided'")
            ranked = np.asarray(list(provided_rank), dtype=int)
            return ranked[:L]

        if rank_source == "random":
            if rng is None:
                rng = np.random.default_rng()
            ranked = np.arange(L, dtype=int)
            rng.shuffle(ranked)
            return ranked

        # co-attention ranking
        score0, ap, ad = self._score_and_attn(sample)
        ranked = rank_from_attention(ap, ad, mode=mode)
        return ranked

    @torch.no_grad()
    def curve_single(
        self,
        sample: Sample,
        mode: Mode,
        test_type: TestType,
        fractions: Sequence[float],
        rank_source: RankSource = "coattn",
        provided_rank: Optional[Sequence[int]] = None,
        rng: Optional[np.random.Generator] = None,
        mask_embedding_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a single curve for one sample.

        Returns:
            fractions_np: (T,)
            curve_scores: (T,) scores (logits/probs depending on your forward_fn)
        """
        sample = self._ensure_device(sample)

        fracs = np.asarray(list(fractions), dtype=float)
        if fracs.ndim != 1:
            raise ValueError("fractions must be 1D")
        fracs = np.clip(fracs, 0.0, 1.0)

        ranked = self._get_ranked_indices(
            sample=sample,
            mode=mode,
            rank_source=rank_source,
            provided_rank=provided_rank,
            rng=rng,
        )

        # Determine length
        L = int(sample.protein_mask.numel() if mode == "protein" else sample.drug_mask.numel())

        # Baseline for insertion: fully masked
        if test_type == "insertion":
            base = _clone_sample(sample)
            base.protein_emb.zero_(); base.protein_mask.zero_()
            base.drug_emb.zero_(); base.drug_mask.zero_()
        else:
            base = None

        scores: List[float] = []

        for f in fracs:
            k = int(math.floor(float(f) * L))
            idx = ranked[:k].tolist() if k > 0 else []

            if test_type == "deletion":
                s_mod = _clone_sample(sample)
                if mode == "protein":
                    s_mod.protein_emb, s_mod.protein_mask = _mask_indices(
                        s_mod.protein_emb, s_mod.protein_mask, idx, mask_embedding_value
                    )
                else:
                    s_mod.drug_emb, s_mod.drug_mask = _mask_indices(
                        s_mod.drug_emb, s_mod.drug_mask, idx, mask_embedding_value
                    )
                scores.append(self._score(s_mod))

            elif test_type == "insertion":
                s_mod = _clone_sample(base)  # type: ignore[arg-type]
                if mode == "protein":
                    s_mod.protein_emb, s_mod.protein_mask = _insert_indices(
                        s_mod.protein_emb, s_mod.protein_mask, sample.protein_emb, idx
                    )
                else:
                    s_mod.drug_emb, s_mod.drug_mask = _insert_indices(
                        s_mod.drug_emb, s_mod.drug_mask, sample.drug_emb, idx
                    )
                scores.append(self._score(s_mod))
            else:
                raise ValueError(f"Unknown test_type: {test_type}")

        return fracs, np.asarray(scores, dtype=float)

    def evaluate_dataset(
        self,
        samples: Sequence[Sample],
        mode: Mode,
        test_type: TestType,
        fractions: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
        n_random: int = 5,
        seed: int = 0,
        mask_embedding_value: float = 0.0,
        normalize_to_baseline: bool = True,
    ) -> CurveResult:
        """Evaluate mean/std curves over a dataset with random baselines.

        normalize_to_baseline:
            - For deletion: subtract score at fraction=0.
            - For insertion: subtract score at fraction=0 (fully masked baseline).
            This makes curves comparable across samples.
        """

        rng = np.random.default_rng(seed)
        fracs_np = np.asarray(list(fractions), dtype=float)

        curves_ours: List[np.ndarray] = []
        curves_rand: List[np.ndarray] = []

        for s in tqdm(samples):
            x, y_ours = self.curve_single(
                sample=s,
                mode=mode,
                test_type=test_type,
                fractions=fracs_np,
                rank_source="coattn",
                rng=rng,
                mask_embedding_value=mask_embedding_value,
            )

            # Random baselines: average over n_random shuffles
            y_rs: List[np.ndarray] = []
            for _ in range(max(1, int(n_random))):
                _, y_r = self.curve_single(
                    sample=s,
                    mode=mode,
                    test_type=test_type,
                    fractions=fracs_np,
                    rank_source="random",
                    rng=rng,
                    mask_embedding_value=mask_embedding_value,
                )
                y_rs.append(y_r)
            y_rand = np.mean(np.stack(y_rs, axis=0), axis=0)

            if normalize_to_baseline:
                y_ours = y_ours - y_ours[0]
                y_rand = y_rand - y_rand[0]

            curves_ours.append(y_ours)
            curves_rand.append(y_rand)

        Y_ours = np.stack(curves_ours, axis=0)
        Y_rand = np.stack(curves_rand, axis=0)

        mean_ours = Y_ours.mean(axis=0)
        std_ours = Y_ours.std(axis=0)
        mean_rand = Y_rand.mean(axis=0)
        std_rand = Y_rand.std(axis=0)

        audc_ours = _trapz_auc(mean_ours, fracs_np)
        audc_rand = _trapz_auc(mean_rand, fracs_np)

        return CurveResult(
            fractions=fracs_np,
            mean_ours=mean_ours,
            std_ours=std_ours,
            mean_rand=mean_rand,
            std_rand=std_rand,
            audc_ours=audc_ours,
            audc_rand=audc_rand,
        )


# --------------------------
# Convenience plotting (optional)
# --------------------------

def plot_curve(result: CurveResult, title: str = "Faithfulness curve") -> None:
    """Quick matplotlib plot; keeps matplotlib optional."""
    import matplotlib.pyplot as plt

    x = result.fractions

    plt.figure()
    plt.plot(x, result.mean_ours, label="ours")
    plt.fill_between(x, result.mean_ours - result.std_ours, result.mean_ours + result.std_ours, alpha=0.2)

    plt.plot(x, result.mean_rand, label="random")
    plt.fill_between(x, result.mean_rand - result.std_rand, result.mean_rand + result.std_rand, alpha=0.2)

    plt.xlabel("fraction masked/inserted")
    plt.ylabel("Δ score (normalized)")
    plt.title(f"{title}\nAUDC ours={result.audc_ours:.4f}, rand={result.audc_rand:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

#
# --------------------------
# Runnable example (mirrors attentiontest.py)
# --------------------------

from dataset import MyDataset, collate_fn
from model import Model
from config.cfg import get_cfg_defaults


def _build_samples_from_loader(dl, device: str, max_samples: int = 64) -> List[Sample]:
    samples: List[Sample] = []
    for i, batch in enumerate(dl):
        if i >= max_samples:
            break
        # DataLoader returns batch-first tensors (B, ...). Faithfulness code expects per-sample tensors
        # without the leading batch dimension.
        p_emb = batch["protein_emb"].to(device)
        d_emb = batch["drug_emb"].to(device)
        p_mask = batch["protein_mask"].to(device)
        d_mask = batch["drug_mask"].to(device)

        # If batch_size=1, strip the batch dimension: (1,L,D)->(L,D), (1,L)->(L,)
        if p_emb.ndim == 3 and p_emb.shape[0] == 1:
            p_emb = p_emb[0]
        if d_emb.ndim == 3 and d_emb.shape[0] == 1:
            d_emb = d_emb[0]
        if p_mask.ndim == 2 and p_mask.shape[0] == 1:
            p_mask = p_mask[0]
        if d_mask.ndim == 2 and d_mask.shape[0] == 1:
            d_mask = d_mask[0]

        samples.append(
            Sample(
                protein_emb=p_emb,
                drug_emb=d_emb,
                protein_mask=p_mask,
                drug_mask=d_mask,
                meta={
                    "drug_id": batch.get("drug_id"),
                    "uniprot_id": batch.get("uniprot_id"),
                    "label": int(batch["label"].detach().cpu().view(-1)[0].item()) if ("label" in batch and batch["label"] is not None) else None,
                },
            )
        )
    return samples


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to saved model checkpoint .pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--n_random", type=int, default=5)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="margin",
        choices=["pos", "neg", "margin", "prob", "entropy", "ce_loss"],
        help=(
            "Scalar score used for faithfulness curves. "
            "pos/neg use class logits, margin=pos-neg, prob=softmax(pos), entropy=H(softmax), "
            "ce_loss=CrossEntropyLoss(logits,target) (label-dependent)."
        ),
    )
    args = parser.parse_args()
    def _scalar_from_logits(logits: Tensor, score_mode: str, sample: Optional[Sample] = None) -> Tensor:
        """Convert 2-class logits to a scalar confidence/uncertainty score."""
        # Ensure shape (..., 2)
        z = logits
        if isinstance(z, (tuple, list)):
            z = z[0]
        z = z.squeeze()
        if z.ndim == 1 and z.numel() == 2:
            z2 = z
        elif z.ndim >= 2 and z.shape[-1] == 2:
            z2 = z[..., -2:].squeeze()
            if z2.ndim > 1:
                z2 = z2.view(-1, 2)[0]
        else:
            raise ValueError(f"Expected 2-class logits; got shape {tuple(z.shape)}")

        z_neg, z_pos = z2[0], z2[1]

        if score_mode == "pos":
            return z_pos
        if score_mode == "neg":
            return z_neg
        if score_mode == "margin":
            return z_pos - z_neg
        if score_mode == "prob":
            p = torch.softmax(z2, dim=-1)
            return p[1]
        if score_mode == "entropy":
            p = torch.softmax(z2, dim=-1)
            return -(p * torch.clamp(p, min=1e-12).log()).sum()
        if score_mode == "ce_loss":
            if sample is None or sample.meta is None or sample.meta.get("label", None) is None:
                raise ValueError("ce_loss score_mode requires sample.meta['label'] to be 0/1")
            y = torch.tensor([int(sample.meta["label"])], device=z2.device, dtype=torch.long)
            # CrossEntropyLoss expects (N,C)
            z_batch = z2.view(1, 2)
            return torch.nn.functional.cross_entropy(z_batch, y, reduction="none")[0]

        raise ValueError(f"Unknown score_mode: {score_mode}")

    DEVICE = args.device

    # ---- load checkpoint + cfg ----
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = get_cfg_defaults()

    model = Model(cfg=cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # ---- dataset/dataloader (same style as attentiontest.py) ----
    test_ds = MyDataset(cfg.DATA.TEST_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)

    samples = _build_samples_from_loader(test_dl, device=DEVICE, max_samples=args.max_samples)
    if len(samples) == 0:
        raise RuntimeError("No samples loaded. Check cfg.DATA.TEST_CSV_PATH and directories.")

    # ---- Split by label (positives vs negatives) ----
    def _label_of(s: Sample):
        if s.meta is None:
            return None
        return s.meta.get("label", None)

    samples_pos = [s for s in samples if _label_of(s) == 1]
    samples_neg = [s for s in samples if _label_of(s) == 0]

    print(f"Loaded {len(samples)} samples: pos={len(samples_pos)}, neg={len(samples_neg)}")

    # --------------------------
    # Model adapter functions (matches attentiontest.py)
    # --------------------------

    @torch.no_grad()
    def forward_fn(sample: Sample) -> Tensor:
        """Return a scalar score (preferably a logit for positive class).

        Your model returns `predictions` when called with return_attention=False.
        If predictions is shape (B, 1) or (1,), we squeeze to scalar.
        """
        # model.py expects batch-first tensors: (B, L, D) and (B, L)
        p_emb = sample.protein_emb
        d_emb = sample.drug_emb
        p_mask = sample.protein_mask
        d_mask = sample.drug_mask

        if p_emb.ndim == 2:
            p_emb = p_emb.unsqueeze(0)
        if d_emb.ndim == 2:
            d_emb = d_emb.unsqueeze(0)
        if p_mask.ndim == 1:
            p_mask = p_mask.unsqueeze(0)
        if d_mask.ndim == 1:
            d_mask = d_mask.unsqueeze(0)

        preds = model(
            p_emb,
            d_emb,
            protein_mask=p_mask,
            drug_mask=d_mask,
            return_attention=False,
        )

        # Some implementations return just preds; others return (preds, ...)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        preds = preds.squeeze()

        # Convert 2-class logits to scalar score
        return _scalar_from_logits(preds, args.score_mode, sample=sample)

    @torch.no_grad()
    def forward_with_attn_fn(sample: Sample) -> Tuple[Tensor, np.ndarray, np.ndarray]:
        """Return (score, attn_p, attn_d) where:
        - attn_p is (Lp, Ld) protein→drug
        - attn_d is (Ld, Lp) drug→protein

        This mirrors attentiontest.py where the model returns:
            predictions, attentionp, attentiond
        with attentionp: (B, Lp, Ld), attentiond: (B, Ld, Lp)
        """
        # model.py expects batch-first tensors: (B, L, D) and (B, L)
        p_emb = sample.protein_emb
        d_emb = sample.drug_emb
        p_mask = sample.protein_mask
        d_mask = sample.drug_mask

        if p_emb.ndim == 2:
            p_emb = p_emb.unsqueeze(0)
        if d_emb.ndim == 2:
            d_emb = d_emb.unsqueeze(0)
        if p_mask.ndim == 1:
            p_mask = p_mask.unsqueeze(0)
        if d_mask.ndim == 1:
            d_mask = d_mask.unsqueeze(0)

        predictions, attentionp, attentiond = model(
            p_emb,
            d_emb,
            protein_mask=p_mask,
            drug_mask=d_mask,
            return_attention=True,
        )

        score = _scalar_from_logits(predictions, args.score_mode, sample=sample)

        # Take batch 0; model uses average_attn_weights=True so there is no head dim.
        attn_p = attentionp[0].detach().cpu().numpy()  # (Lp, Ld)
        attn_d = attentiond[0].detach().cpu().numpy()  # (Ld, Lp)

        return score.squeeze(), attn_p, attn_d

    scorer = FaithfulnessScorer(
        forward_fn=forward_fn,
        forward_with_attn_fn=forward_with_attn_fn,
        device=DEVICE,
    )

    fractions = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)

    # ---- Protein-side deletion (ALL / POS / NEG) ----
    # res_p_del_all = scorer.evaluate_dataset(
    #     samples=samples,
    #     mode="protein",
    #     test_type="deletion",
    #     fractions=fractions,
    #     n_random=args.n_random,
    #     seed=0,
    #     normalize_to_baseline=True,
    # )
    # plot_curve(res_p_del_all, title=f"Faithfulness: Protein deletion (ALL, Δ{args.score_mode})")

    # if len(samples_pos) >= 2:
    #     res_p_del_pos = scorer.evaluate_dataset(
    #         samples=samples_pos,
    #         mode="protein",
    #         test_type="deletion",
    #         fractions=fractions,
    #         n_random=args.n_random,
    #         seed=0,
    #         normalize_to_baseline=True,
    #     )
    #     plot_curve(res_p_del_pos, title="Faithfulness: Protein deletion (POS, Δlogit)")

    if len(samples_neg) >= 2:
        res_p_del_neg = scorer.evaluate_dataset(
            samples=samples_neg,
            mode="protein",
            test_type="deletion",
            fractions=fractions,
            n_random=args.n_random,
            seed=0,
            normalize_to_baseline=True,
        )
        plot_curve(res_p_del_neg, title="Faithfulness: Protein deletion (NEG, Δlogit)")

    # ---- Drug-side deletion (ALL / POS / NEG) ----
    # res_d_del_all = scorer.evaluate_dataset(
    #     samples=samples,
    #     mode="drug",
    #     test_type="deletion",
    #     fractions=fractions,
    #     n_random=args.n_random,
    #     seed=0,
    #     normalize_to_baseline=True,
    # )
    # plot_curve(res_d_del_all, title=f"Faithfulness: Drug deletion (ALL, Δ{args.score_mode})")

    if len(samples_pos) >= 2:
        res_d_del_pos = scorer.evaluate_dataset(
            samples=samples_pos,
            mode="drug",
            test_type="deletion",
            fractions=fractions,
            n_random=args.n_random,
            seed=0,
            normalize_to_baseline=True,
        )
        plot_curve(res_d_del_pos, title="Faithfulness: Drug deletion (POS, Δlogit)")

    if len(samples_neg) >= 2:
        res_d_del_neg = scorer.evaluate_dataset(
            samples=samples_neg,
            mode="drug",
            test_type="deletion",
            fractions=fractions,
            n_random=args.n_random,
            seed=0,
            normalize_to_baseline=True,
        )
        plot_curve(res_d_del_neg, title="Faithfulness: Drug deletion (NEG, Δlogit)")

    # Optional: insertion tests (uncomment if desired)
    # res_p_ins = scorer.evaluate_dataset(samples=samples, mode="protein", test_type="insertion", fractions=fractions, n_random=args.n_random)
    # plot_curve(res_p_ins, title="Faithfulness: Protein insertion (Δlogit)")
    # res_d_ins = scorer.evaluate_dataset(samples=samples, mode="drug", test_type="insertion", fractions=fractions, n_random=args.n_random)
    # plot_curve(res_d_ins, title="Faithfulness: Drug insertion (Δlogit)")

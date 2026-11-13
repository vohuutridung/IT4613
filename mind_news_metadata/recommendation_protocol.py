"""Sampling, masking, schedule, and impression metrics for stage 03/04."""
import hashlib
import math

import numpy as np
from mind_config import TRAINING_CONFIG


def stable_impression_seed(base_seed, split, impression_id):
    payload = "{}\0{}\0{}".format(int(base_seed), split, impression_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def partition_for_impression(labels, positions, impression_id, partition,
                             partitions=None, base_seed=None, split="train"):
    """Keep all positives; assign every negative to exactly one stable partition.

    For a given impression, one MT19937 permutation maps raw candidate positions
    to modulo partitions. This is a deterministic reimplementation of the
    upstream per-impression shuffle, not a claim of matching its process-global
    NumPy RNG sequence.
    """
    if partitions is None:
        partitions = TRAINING_CONFIG["partition_count"]
    if base_seed is None:
        base_seed = TRAINING_CONFIG["partition_base_seed"]
    if split != "train":
        raise ValueError("negative partition sampling is train-only")
    labels = np.asarray(labels, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    if labels.shape != positions.shape or len(set(positions.tolist())) != len(positions):
        raise ValueError("labels and unique candidate positions must align")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("partition sampling requires binary labeled candidates")
    if not 0 <= int(partition) < int(partitions):
        raise ValueError("partition outside configured range")
    n = int(positions.max()) + 1 if len(positions) else 0
    if len(positions) != n or set(positions.tolist()) != set(range(n)):
        raise ValueError("partition function requires the complete impression candidate set")
    permuted = np.random.RandomState(stable_impression_seed(base_seed, split, impression_id)).permutation(n)
    chosen = (labels == 1) | ((permuted[positions] % int(partitions)) == int(partition))
    return chosen


def learning_rate_at(update, total_updates,
                     peak=TRAINING_CONFIG["learning_rate_peak"],
                     minimum=TRAINING_CONFIG["learning_rate_min"],
                     warmup_fraction=TRAINING_CONFIG["warmup_fraction"]):
    """Warm to peak then linearly decay to minimum over one epoch."""
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    warmup = max(1, int(math.ceil(total_updates * float(warmup_fraction))))
    update = int(update)
    if update < 0 or update >= total_updates:
        raise ValueError("update index outside schedule")
    if update < warmup:
        return float(peak) * float(update + 1) / warmup
    decay_steps = max(1, total_updates - warmup - 1)
    progress = min(1., float(update - warmup) / decay_steps)
    return float(peak) + (float(minimum) - float(peak)) * progress


def _dcg(labels, order, k):
    total = 0.0
    for rank, idx in enumerate(order[:k], 1):
        total += (2.0 ** int(labels[idx]) - 1.0) / math.log2(rank + 1)
    return total


def impression_metrics(labels, scores, positions, k_values=None):
    """One impression metrics; ties break by candidate_position.

    MRR is the mean reciprocal rank over all positive candidates, matching the
    locked upstream helper rather than first-positive reciprocal rank.
    """
    if k_values is None:
        k_values = TRAINING_CONFIG["ranking_metrics_k"]
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    p = np.asarray(positions, dtype=np.int64)
    if not (y.ndim == s.ndim == p.ndim == 1 and len(y) == len(s) == len(p)):
        raise ValueError("labels, scores, positions must be aligned vectors")
    if len(set(p.tolist())) != len(p) or not np.isfinite(s).all():
        raise ValueError("positions must be unique and scores finite")
    order = sorted(range(len(y)), key=lambda i: (-s[i], p[i]))
    positives = int(y.sum())
    negatives = len(y) - positives
    auc = None
    if positives and negatives:
        # Mann-Whitney with average ranks for score ties.
        ranks = np.empty(len(y), dtype=np.float64)
        sorted_idx = sorted(range(len(y)), key=lambda i: (s[i], p[i]))
        i = 0
        while i < len(sorted_idx):
            j = i + 1
            while j < len(sorted_idx) and s[sorted_idx[j]] == s[sorted_idx[i]]:
                j += 1
            avg_rank = ((i + 1) + j) / 2.0
            for q in sorted_idx[i:j]:
                ranks[q] = avg_rank
            i = j
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) /
                    (positives * negatives))
    rr = (sum(1.0 / (rank + 1) for rank, idx in enumerate(order) if y[idx] == 1) / positives
          if positives else None)
    ndcg = {}
    ideal_labels = sorted(y.tolist(), reverse=True)
    for k in k_values:
        ideal = _dcg(np.asarray(ideal_labels, np.int32), list(range(len(ideal_labels))), k)
        ndcg[int(k)] = _dcg(y, order, k) / ideal if ideal else None
    return {"auc": auc, "mrr_all_positives": rr, "ndcg": ndcg,
            "positive_count": positives, "negative_count": negatives}


def aggregate_impression_metrics(groups):
    """groups yields (impression_id, labels, scores, positions)."""
    sums = {"auc": 0., "mrr_all_positives": 0., "ndcg@5": 0., "ndcg@10": 0.}
    source_legacy = dict(sums)
    valid_auc_count = 0
    valid_ranking_count = 0
    group_count = 0
    excluded_auc = 0
    for _, labels, scores, positions in groups:
        m = impression_metrics(labels, scores, positions)
        group_count += 1
        if m["auc"] is None:
            excluded_auc += 1
        else:
            sums["auc"] += m["auc"]
            source_legacy["auc"] += m["auc"]
            valid_auc_count += 1
        for name, val in (("mrr_all_positives", m["mrr_all_positives"]),
                          ("ndcg@5", m["ndcg"][5]), ("ndcg@10", m["ndcg"][10])):
            if val is not None:
                sums[name] += val
                if m["positive_count"] > 0:
                    valid_ranking_count += 1 if name == "mrr_all_positives" else 0
                if m["auc"] is not None:  # legacy helper accumulated only mixed-label groups
                    source_legacy[name] += val
        # nDCG denominators are also all groups containing at least one positive.
    return {
        "valid_groups_only": {
            "auc": sums["auc"] / valid_auc_count if valid_auc_count else None,
            "mrr_all_positives": sums["mrr_all_positives"] / valid_ranking_count if valid_ranking_count else None,
            "ndcg@5": sums["ndcg@5"] / valid_ranking_count if valid_ranking_count else None,
            "ndcg@10": sums["ndcg@10"] / valid_ranking_count if valid_ranking_count else None,
        },
        "source_legacy": {k: (v / group_count if group_count else None) for k, v in source_legacy.items()},
        "group_count": group_count, "valid_mixed_label_groups": valid_auc_count,
        "valid_positive_groups": valid_ranking_count,
        "excluded_auc_groups": excluded_auc,
    }

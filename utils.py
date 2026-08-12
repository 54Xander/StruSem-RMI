import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import InMemoryDataset


def set_seed(seed):
    """Set random seeds and deterministic backend options."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass


def rna2D_from_dot(seq, dot_bracket):
    """Convert an RNA sequence and dot-bracket structure to an RGCN graph."""
    base_dict = {
        "A": 0,
        "U": 1,
        "G": 2,
        "C": 3,
        "N": 4,
    }
    indices = torch.tensor(
        [
            base_dict.get(base.upper(), 4)
            for base in seq
        ],
        dtype=torch.long,
    )
    x = F.one_hot(
        indices,
        num_classes=5,
    ).float()

    stack = []
    pair_map = {}

    for idx, symbol in enumerate(dot_bracket):
        if symbol == "(":
            stack.append(idx)
        elif symbol == ")" and stack:
            paired_idx = stack.pop()
            pair_map[idx] = paired_idx
            pair_map[paired_idx] = idx

    edge_type_map = {
        "link": 0,
        ("C", "G"): 1,
        ("A", "U"): 2,
        ("G", "U"): 3,
        ("A", "G"): 4,
        ("U", "U"): 5,
        ("C", "C"): 6,
        ("A", "A"): 7,
        "unknown": 8,
    }

    edge_index = []
    edge_type = []

    for idx in range(len(seq) - 1):
        edge_index.extend(
            [
                [idx, idx + 1],
                [idx + 1, idx],
            ]
        )
        edge_type.extend([0, 0])

    for idx, paired_idx in pair_map.items():
        if idx >= paired_idx:
            continue

        sorted_bases = tuple(
            sorted(
                [
                    seq[idx].upper(),
                    seq[paired_idx].upper(),
                ]
            )
        )
        pair_type = edge_type_map.get(
            sorted_bases,
            8,
        )

        edge_index.extend(
            [
                [idx, paired_idx],
                [paired_idx, idx],
            ]
        )
        edge_type.extend(
            [pair_type, pair_type]
        )

    if not edge_index:
        edge_index = torch.zeros(
            (2, 0),
            dtype=torch.long,
        )
        edge_type = torch.zeros(
            0,
            dtype=torch.long,
        )
    else:
        edge_index = torch.tensor(
            edge_index,
            dtype=torch.long,
        ).t().contiguous()
        edge_type = torch.tensor(
            edge_type,
            dtype=torch.long,
        )

    return x, edge_index, edge_type


def get_metrics(y_true, y_pred):
    """Return AUC, AUPR, F1, accuracy, recall, specificity, and precision."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    try:
        auc_score = roc_auc_score(
            y_true,
            y_pred,
        )
    except ValueError:
        auc_score = 0.5

    aupr_score = average_precision_score(
        y_true,
        y_pred,
    )
    pred_label = (
        y_pred >= 0.5
    ).astype(int)

    true_negative = (
        (pred_label == 0)
        & (y_true == 0)
    ).sum()
    false_positive = (
        (pred_label == 1)
        & (y_true == 0)
    ).sum()
    specificity = true_negative / (
        true_negative
        + false_positive
        + 1e-7
    )

    return [
        round(float(auc_score), 4),
        round(float(aupr_score), 4),
        round(
            float(
                f1_score(
                    y_true,
                    pred_label,
                    zero_division=0,
                )
            ),
            4,
        ),
        round(
            float(
                accuracy_score(
                    y_true,
                    pred_label,
                )
            ),
            4,
        ),
        round(
            float(
                recall_score(
                    y_true,
                    pred_label,
                    zero_division=0,
                )
            ),
            4,
        ),
        round(float(specificity), 4),
        round(
            float(
                precision_score(
                    y_true,
                    pred_label,
                    zero_division=0,
                )
            ),
            4,
        ),
    ]


class TestbedDataset(InMemoryDataset):
    def __init__(
        self,
        root="/tmp",
        dataset="davis",
        xd=None,
        xt=None,
        y=None,
        transform=None,
        pre_transform=None,
        smile_graph=None,
        k_mer_features=None,
    ):
        super().__init__(
            root,
            transform,
            pre_transform,
        )
        self.dataset = dataset

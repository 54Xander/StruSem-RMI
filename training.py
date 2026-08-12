import argparse
import gc
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from model import InteractionContrastiveLoss, MultiModalModel
from utils import TestbedDataset, get_metrics, set_seed


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

try:
    import create_data
except ImportError:
    create_data = None


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_data_exists(args):
    """Generate offline features automatically when they are missing."""
    dataset_name = args["dataset"]
    check_path = (
        f"data/processed/{dataset_name}_fold1_tra.pt"
    )

    if os.path.exists(check_path):
        print("[Auto-Check] Offline data detected.")
        return

    print(
        f"[Auto-Check] Data file not found: {check_path}"
    )
    print("[Auto-Check] Running create_data.py...")

    if create_data is None:
        raise FileNotFoundError(
            "create_data.py is required to generate offline data."
        )

    create_data.run_generation(args)


def build_dataset(data_list):
    dataset = TestbedDataset(root="data", dataset="tmp")
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset


def load_single_fold(args, fold):
    """Load one fold and create train/validation/test loaders."""
    dataset_name = args["dataset"]
    print(f"[Load] Loading offline data for Fold {fold}...")

    train_list = torch.load(
        f"data/processed/{dataset_name}_fold{fold}_tra.pt",
        weights_only=False,
    )
    val_list = torch.load(
        f"data/processed/{dataset_name}_fold{fold}_val.pt",
        weights_only=False,
    )
    test_list = torch.load(
        f"data/processed/{dataset_name}_fold{fold}_tes.pt",
        weights_only=False,
    )

    train_dataset = build_dataset(train_list)
    val_dataset = build_dataset(val_list)
    test_dataset = build_dataset(test_list)

    generator = torch.Generator()
    generator.manual_seed(args["seed"])

    common_loader_args = {
        "batch_size": args["batch_size"],
        "num_workers": args["num_workers"],
        "pin_memory": bool(args["pin_memory"]),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_args,
    )

    return train_loader, val_loader, test_loader


def compute_pos_weight(loader):
    labels = np.array(
        [data.y.item() for data in loader.dataset]
    )
    num_neg = (labels == 0).sum()
    num_pos = (labels == 1).sum()
    return torch.tensor(
        [(num_neg + 1e-7) / (num_pos + 1e-7)],
        dtype=torch.float32,
    )


def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    args,
):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    contrastive_loss_fn = InteractionContrastiveLoss(
        temperature=args["contrastive_temp"]
    )

    for data in loader:
        data = data.to(device)
        target = data.y.view(-1, 1).float()

        optimizer.zero_grad()
        output = model(data)
        logits = output["out"]

        bce_loss = loss_fn(logits, target)
        interaction_loss = contrastive_loss_fn(
            output["rna_feat"],
            output["drug_feat"],
            data.y.view(-1),
        )
        loss = (
            bce_loss
            + args["aux_weight_interaction"] * interaction_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        y_pred.extend(
            torch.sigmoid(logits)
            .detach()
            .cpu()
            .view(-1)
            .tolist()
        )
        y_true.extend(data.y.detach().cpu().view(-1).tolist())

    mean_loss = total_loss / max(len(y_true), 1)
    return round(mean_loss, 5), get_metrics(y_true, y_pred)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target = data.y.view(-1, 1).float()
            output = model(data)
            logits = output["out"]

            loss = loss_fn(logits, target)
            total_loss += loss.item() * target.size(0)

            y_pred.extend(
                torch.sigmoid(logits)
                .cpu()
                .view(-1)
                .tolist()
            )
            y_true.extend(
                data.y.cpu().view(-1).tolist()
            )

    mean_loss = total_loss / max(len(y_true), 1)
    return round(mean_loss, 5), get_metrics(y_true, y_pred)


def run_single_fold(fold, args, device):
    """Train and evaluate one cross-validation fold."""
    train_loader, val_loader, test_loader = load_single_fold(
        args,
        fold,
    )

    sample_data = train_loader.dataset[0]
    args["mol_input_dim"] = sample_data.mol_chem_feat.shape[-1]
    args["rna_input_dim"] = sample_data.rna_bert_feat.shape[-1]

    if sample_data.mol_sem_feat.shape[-1] > 0:
        args["bio_input_dim"] = sample_data.mol_sem_feat.shape[-1]
    elif sample_data.rna_sem_feat.shape[-1] > 0:
        args["bio_input_dim"] = sample_data.rna_sem_feat.shape[-1]
    else:
        args["bio_input_dim"] = 768

    print(
        f"\n===== Fold {fold} / {args['n_splits']} ====="
    )

    set_seed(args["seed"])
    model = MultiModalModel(args).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lr"],
        weight_decay=args["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args["scheduler_patience"],
        factor=args["scheduler_factor"],
    )

    pos_weight = compute_pos_weight(train_loader).to(device)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    ).to(device)

    dataset_name = args["dataset"]
    best_auc = -np.inf
    best_epoch = -1
    best_test_metrics = None
    result_metrics = np.zeros(
        (args["epochs"], 1 + 7 * 3)
    )
    early_stop_counter = 0

    os.makedirs("model", exist_ok=True)
    os.makedirs("result", exist_ok=True)

    for epoch in range(args["epochs"]):
        start_time = time.time()

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            args,
        )
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )
        test_loss, test_metrics = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
        )

        scheduler.step(val_loss)

        elapsed_minutes = round(
            (time.time() - start_time) / 60,
            2,
        )
        print(
            f"Ep {epoch:03d} | Time {elapsed_minutes}m"
        )
        print(
            f"Tra Loss: {train_loss:.5f} | "
            f"Metrics: {train_metrics}"
        )
        print(
            f"Val Loss: {val_loss:.5f} | "
            f"Metrics: {val_metrics}"
        )
        print(
            f"Tes Loss: {test_loss:.5f} | "
            f"Metrics: {test_metrics}",
            flush=True,
        )

        result_metrics[epoch, 0] = epoch
        for metric_idx in range(7):
            base_col = 3 * metric_idx + 1
            result_metrics[epoch, base_col] = (
                train_metrics[metric_idx]
            )
            result_metrics[epoch, base_col + 1] = (
                val_metrics[metric_idx]
            )
            result_metrics[epoch, base_col + 2] = (
                test_metrics[metric_idx]
            )

        if val_metrics[0] > best_auc:
            best_auc = val_metrics[0]
            best_epoch = epoch
            best_test_metrics = test_metrics
            early_stop_counter = 0

            model_path = (
                f"model/{dataset_name}_fold{fold}.pt"
            )
            torch.save(model.state_dict(), model_path)
            print(
                f">>> Best validation AUC updated at "
                f"epoch {epoch:03d}; "
                f"test AUC={test_metrics[0]}"
            )
        else:
            early_stop_counter += 1
            print(
                f"No validation improvement for "
                f"{early_stop_counter}/"
                f"{args['early_stop_patience']} epochs"
            )

        if (
            early_stop_counter
            >= args["early_stop_patience"]
        ):
            print("Early stopping triggered.")
            break

    print(
        f"\nFold {fold} Best Epoch: {best_epoch:03d} | "
        f"Best Test Metrics: {best_test_metrics}"
    )

    actual_epochs = epoch + 1
    result_metrics = result_metrics[:actual_epochs, :]

    columns = ["Epoch"] + [
        f"{phase}_{metric}"
        for metric in [
            "AUC",
            "AUPR",
            "F1",
            "Acc",
            "Rec",
            "Spec",
            "Prec",
        ]
        for phase in ["Tra", "Val", "Tes"]
    ]

    result_df = pd.DataFrame(
        result_metrics,
        columns=columns,
    )
    time_str = time.strftime(
        "%Y-%m-%d-%H_%M_%S",
        time.localtime(),
    )
    result_path = (
        f"result/result_{dataset_name}_"
        f"fold{fold}_{time_str}.csv"
    )
    result_df.to_csv(result_path, index=False)


def train_cross_validation(args):
    device = args["device"]

    try:
        check_data_exists(args)
    except Exception as exc:
        raise RuntimeError(
            f"Data preparation failed: {exc}"
        ) from exc

    for fold in range(1, args["n_splits"] + 1):
        run_single_fold(fold, args, device)

        print(
            f"Cleaning memory after Fold {fold}..."
        )
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"Fold {fold} cleanup completed.\n")


def get_args():
    parser = argparse.ArgumentParser(description="Train the multimodal interaction model")

    parser.add_argument("--dataset", default="RSID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--n_splits", type=int, default=5)

    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--n_output", type=int, default=1)

    parser.add_argument("--contrastive_temp", type=float, default=0.3)
    parser.add_argument("--aux_weight_interaction", type=float, default=0.3)

    parser.add_argument("--scheduler_patience", type=int, default=7)
    parser.add_argument("--scheduler_factor", type=float, default=0.7)
    parser.add_argument("--early_stop_patience", type=int, default=20)

    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--mol_chem_path", type=str, default="./pretrained_models/ChemBERTa-77M-MTR")
    parser.add_argument("--mol_sem_path", type=str, default="./pretrained_models/biobert-base-cased-v1.2")
    parser.add_argument("--rnabert_path", type=str, default="./pretrained_models/rnabert")
    parser.add_argument("--max_seq_len", type=int, default=267)
    parser.add_argument("--max_mol_len", type=int, default=241)
    parser.add_argument("--val_size", type=float, default=0.1)

    args = vars(parser.parse_args())
    args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


if __name__ == "__main__":
    config = get_args()
    set_seed(config["seed"])
    train_cross_validation(config)

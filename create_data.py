import argparse
import os

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer

from utils import rna2D_from_dot


RDLogger.DisableLog("rdApp.*")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class SimpleRnaTokenizer:
    """Minimal tokenizer compatible with the local RNABERT vocabulary."""

    def __init__(self, model_path):
        vocab_path = os.path.join(model_path, "vocab.txt")
        self.vocab = {}

        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as file:
                for idx, line in enumerate(file):
                    self.vocab[line.strip()] = idx
        else:
            self.vocab = {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "A": 4,
                "U": 5,
                "G": 6,
                "C": 7,
                "N": 8,
            }

        self.pad_token_id = self.vocab.get("[PAD]", 0)
        self.cls_token_id = self.vocab.get("[CLS]", 2)
        self.sep_token_id = self.vocab.get("[SEP]", 3)
        self.unk_token_id = self.vocab.get("[UNK]", 1)

    def __call__(
        self,
        text,
        max_length=220,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ):
        text = str(text) if text is not None else ""
        tokens = list(text.upper())
        input_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]

        if truncation and len(input_ids) > max_length - 2:
            input_ids = input_ids[: max_length - 2]

        input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)

        if padding == "max_length" and len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.tensor([attention_mask]),
            }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    @classmethod
    def from_pretrained(cls, path):
        return cls(path)


def extract_feature(model, inputs, device):
    """Extract one pooled feature vector from a pretrained encoder."""
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if (
            hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
        ):
            feature = outputs.pooler_output
        else:
            feature = outputs.last_hidden_state[:, 0, :]

    return feature.cpu()


def read_raw_data(dataset_path, n_splits=5, seed=42, val_size=0.1):
    """Create stratified random train/validation/test folds."""
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    molecule_path = os.path.join(dataset_path, "Molecule.xlsx")
    rna_path = os.path.join(dataset_path, "RNA.xlsx")
    interaction_path = os.path.join(dataset_path, "RNA-Molecule.xlsx")

    df_molecules = pd.read_excel(molecule_path)
    molecules_map = {
        row["Small molecule_ID"]: row["SMILES"]
        for _, row in df_molecules.iterrows()
    }
    mol_info_map = {
        row["Small molecule_ID"]: (
            str(row["Small molecule information"])
            if pd.notna(row["Small molecule information"])
            else ""
        )
        for _, row in df_molecules.iterrows()
    }

    df_rnas = pd.read_excel(rna_path).set_index("RNA_ID")
    if "RNA information" not in df_rnas.columns:
        df_rnas["RNA information"] = ""

    df_labels = pd.read_excel(interaction_path)

    mol_id_col = "Small molecule_ID"
    label_col = "label"

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    splits = list(splitter.split(df_labels, df_labels[label_col]))

    all_folds_data = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        df_train_full = df_labels.iloc[train_idx]
        df_test = df_labels.iloc[test_idx]

        df_train, df_val = train_test_split(
            df_train_full,
            test_size=val_size,
            stratify=df_train_full[label_col],
            random_state=seed,
        )

        processed_dfs = []
        for df_subset, split_name in zip(
            [df_train, df_val, df_test],
            ["tra", "val", "tes"],
        ):
            df_subset = df_subset.merge(
                df_rnas,
                left_on="RNA_ID",
                right_index=True,
            )
            df_subset["SMILES"] = df_subset[mol_id_col].map(molecules_map)
            df_subset["Small_molecule_information"] = df_subset[
                mol_id_col
            ].map(mol_info_map)
            df_subset = df_subset.dropna(subset=["SMILES"])

            csv_path = os.path.join(
                processed_dir,
                f"{dataset_name}_fold{fold_idx}_{split_name}.csv",
            )
            df_subset.to_csv(csv_path, index=False)
            processed_dfs.append(df_subset)

        all_folds_data.append(tuple(processed_dfs))
        print(
            f"  [Fold {fold_idx}] Split complete. "
            f"Train: {len(processed_dfs[0])}, "
            f"Val: {len(processed_dfs[1])}, "
            f"Test: {len(processed_dfs[2])}"
        )

    return all_folds_data


def trans_multimodal(
    dataset_path,
    df_data,
    split_name,
    fold,
    args,
    models_dict,
    device,
):
    """Generate and cache offline multimodal features."""
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    pt_path = f"data/processed/{dataset_name}_fold{fold}_{split_name}.pt"

    if os.path.exists(pt_path):
        print(f"[Cache] Loading offline feature data: {pt_path}")
        return torch.load(pt_path, weights_only=False)

    print(
        f"[Process] Generating offline features for "
        f"{split_name} (Fold {fold})..."
    )

    def get_arg(key):
        return args[key] if isinstance(args, dict) else getattr(args, key)

    tok_chem = AutoTokenizer.from_pretrained(
        get_arg("mol_chem_path"),
        local_files_only=True,
    )
    tok_bio = AutoTokenizer.from_pretrained(
        get_arg("mol_sem_path"),
        local_files_only=True,
    )
    tok_rna = SimpleRnaTokenizer.from_pretrained(get_arg("rnabert_path"))

    max_seq_len = get_arg("max_seq_len")
    max_mol_len = get_arg("max_mol_len")

    model_chem = models_dict["mol_chem"]
    model_bio = models_dict["mol_sem"]
    model_rna = models_dict["rnabert"]

    data_list = []

    for idx, (_, row) in enumerate(df_data.iterrows()):
        if idx % 100 == 0:
            percent = (idx / max(len(df_data), 1)) * 100
            print(
                f"  Processing {idx}/{len(df_data)} ({percent:.0f}%)...",
                end="\r",
            )

        smiles = str(row["SMILES"]) if pd.notna(row["SMILES"]) else ""
        mol_text = (
            str(row["Small_molecule_information"])
            if pd.notna(row["Small_molecule_information"])
            else ""
        )
        rna_seq = (
            str(row["1D Sequence"])
            if pd.notna(row["1D Sequence"])
            else ""
        )
        rna_text = (
            str(row["RNA information"])
            if pd.notna(row["RNA information"])
            else ""
        )
        dot_bracket = (
            str(row["Dot bracket"])
            if pd.notna(row["Dot bracket"])
            else ""
        )

        chem_tokens = tok_chem(
            smiles,
            max_length=max_mol_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        mol_chem_feat = extract_feature(
            model_chem,
            chem_tokens,
            device,
        )

        mol_sem_tokens = tok_bio(
            f"{smiles} [SEP] {mol_text}",
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        mol_sem_feat = extract_feature(
            model_bio,
            mol_sem_tokens,
            device,
        )

        rna_tokens = tok_rna(
            rna_seq,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rna_bert_feat = extract_feature(
            model_rna,
            rna_tokens,
            device,
        )

        rna_sem_tokens = tok_bio(
            f"{rna_seq} [SEP] {rna_text}",
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rna_sem_feat = extract_feature(
            model_bio,
            rna_sem_tokens,
            device,
        )

        is_valid_2d = int(
            bool(dot_bracket) and "(" in dot_bracket
        )
        rna_2d_x = torch.zeros(
            (max_seq_len, 5),
            dtype=torch.float,
        )
        rna_2d_edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )
        rna_2d_edge_type = torch.empty(
            (0,),
            dtype=torch.long,
        )

        if is_valid_2d:
            try:
                result_2d = rna2D_from_dot(
                    rna_seq[:max_seq_len].ljust(max_seq_len, "N"),
                    dot_bracket[:max_seq_len].ljust(max_seq_len, "."),
                )
                (
                    rna_2d_x,
                    rna_2d_edge_index,
                    rna_2d_edge_type,
                ) = result_2d
            except Exception:
                is_valid_2d = 0
                rna_2d_x = torch.zeros(
                    (max_seq_len, 5),
                    dtype=torch.float,
                )
                rna_2d_edge_index = torch.empty(
                    (2, 0),
                    dtype=torch.long,
                )
                rna_2d_edge_type = torch.empty(
                    (0,),
                    dtype=torch.long,
                )

        mol_mask = torch.tensor(
            [
                int(len(smiles) > 5),
                int(len(mol_text) > 5),
            ],
            dtype=torch.float,
        )
        rna_mask = torch.tensor(
            [
                int(len(rna_seq) > 0),
                int(len(rna_text) > 5),
                is_valid_2d,
            ],
            dtype=torch.float,
        )

        data = Data(
            y=torch.tensor([row["label"]], dtype=torch.float),
            mol_chem_feat=mol_chem_feat.squeeze(0),
            mol_sem_feat=mol_sem_feat.squeeze(0),
            rna_bert_feat=rna_bert_feat.squeeze(0),
            rna_sem_feat=rna_sem_feat.squeeze(0),
            rna_2d_x=rna_2d_x.float().contiguous(),
            rna_2d_edge_index=(
                rna_2d_edge_index.long().contiguous()
            ),
            rna_2d_edge_type=(
                rna_2d_edge_type.long().contiguous()
            ),
            mol_mask=mol_mask,
            rna_mask=rna_mask,
            num_nodes=rna_2d_x.size(0),
        )
        data_list.append(data)

    print()
    os.makedirs("data/processed", exist_ok=True)
    torch.save(data_list, pt_path)
    return data_list


def run_generation(args):
    """Generate all folds and offline features."""
    args_dict = vars(args) if not isinstance(args, dict) else args

    print("=" * 50)
    print("[Auto-Gen] Starting data generation workflow")
    print(f"  Dataset: {args_dict.get('dataset', 'RSID')}")
    print("=" * 50)

    if "dataset_path" not in args_dict:
        args_dict["dataset_path"] = os.path.join(
            "data",
            args_dict.get("dataset", "RSID"),
        )

    if not os.path.exists(args_dict["dataset_path"]):
        raise FileNotFoundError(
            f"Dataset path does not exist: "
            f"{args_dict['dataset_path']}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Loading pretrained models on {device}...")

    def load_model(path):
        print(f"  Loading model from {path}...")
        return AutoModel.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=True,
        ).to(device).eval()

    models_dict = {
        "mol_chem": load_model(args_dict["mol_chem_path"]),
        "mol_sem": load_model(args_dict["mol_sem_path"]),
        "rnabert": load_model(args_dict["rnabert_path"]),
    }

    n_splits = args_dict.get("n_splits", 5)
    seed = args_dict.get("seed", 42)
    val_size = args_dict.get("val_size", 0.1)

    all_folds = read_raw_data(
        args_dict["dataset_path"],
        n_splits=n_splits,
        seed=seed,
        val_size=val_size,
    )

    for fold, fold_data in enumerate(all_folds, start=1):
        for df_subset, split_name in zip(
            fold_data,
            ["tra", "val", "tes"],
        ):
            trans_multimodal(
                args_dict["dataset_path"],
                df_subset,
                split_name,
                fold,
                args_dict,
                models_dict,
                device,
            )

    print("[Done] Data generation completed.")


def get_args():
    parser = argparse.ArgumentParser(description="Offline multimodal feature extraction")

    parser.add_argument("--dataset_path", type=str, default="data/RSID")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=267)
    parser.add_argument("--max_mol_len", type=int, default=241)

    parser.add_argument("--mol_chem_path", type=str, default="./pretrained_models/ChemBERTa-77M-MTR")
    parser.add_argument("--mol_sem_path", type=str, default="./pretrained_models/biobert-base-cased-v1.2")
    parser.add_argument("--rnabert_path", type=str, default="./pretrained_models/rnabert")

    return parser.parse_args()


if __name__ == "__main__":
    run_generation(get_args())

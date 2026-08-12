import os

import pandas as pd
import torch
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer

from create_data import SimpleRnaTokenizer, extract_feature
from model import MultiModalModel
from utils import rna2D_from_dot


args = {
    "embed_dim": 512,
    "nhead": 8,
    "dropout": 0.3,
    "n_output": 1,
    "max_seq_len": 267,
    "max_mol_len": 241,
    "mol_chem_path": "./pretrained_models/ChemBERTa-77M-MTR",
    "mol_sem_path": "./pretrained_models/biobert-base-cased-v1.2",
    "rnabert_path": "./pretrained_models/rnabert",
    "model_weight": "model/RSID_fold1.pt",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_feature_extractors():
    print("[Init] Loading feature extraction models...")

    tok_chem = AutoTokenizer.from_pretrained(
        args["mol_chem_path"], local_files_only=True
    )
    tok_bio = AutoTokenizer.from_pretrained(
        args["mol_sem_path"], local_files_only=True
    )
    tok_rna = SimpleRnaTokenizer.from_pretrained(args["rnabert_path"])

    model_chem = AutoModel.from_pretrained(
        args["mol_chem_path"], local_files_only=True
    ).to(device).eval()
    model_bio = AutoModel.from_pretrained(
        args["mol_sem_path"], local_files_only=True
    ).to(device).eval()
    model_rna = AutoModel.from_pretrained(
        args["rnabert_path"],
        local_files_only=True,
        trust_remote_code=True,
    ).to(device).eval()

    tokenizers = (tok_chem, tok_bio, tok_rna)
    models = (model_chem, model_bio, model_rna)
    return tokenizers, models


def process_row(row, tokenizers, models):
    """Convert one prediction row into a PyG Data object."""
    tok_chem, tok_bio, tok_rna = tokenizers
    model_chem, model_bio, model_rna = models

    smiles = str(row["SMILES"]) if pd.notna(row["SMILES"]) else ""
    mol_text = (
        str(row["Small molecule information"])
        if pd.notna(row["Small molecule information"])
        else ""
    )
    rna_seq = str(row["1D Sequence"]) if pd.notna(row["1D Sequence"]) else ""
    rna_text = (
        str(row["RNA information"])
        if pd.notna(row["RNA information"])
        else ""
    )
    dot_bracket = (
        str(row["Dot bracket"]) if pd.notna(row["Dot bracket"]) else ""
    )

    with torch.no_grad():
        chem_inputs = tok_chem(
            smiles,
            max_length=args["max_mol_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        mol_chem_feat = extract_feature(model_chem, chem_inputs, device)

        mol_sem_inputs = tok_bio(
            f"{smiles} [SEP] {mol_text}",
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        mol_sem_feat = extract_feature(model_bio, mol_sem_inputs, device)

        rna_inputs = tok_rna(
            rna_seq,
            max_length=args["max_seq_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rna_bert_feat = extract_feature(model_rna, rna_inputs, device)

        rna_sem_inputs = tok_bio(
            f"{rna_seq} [SEP] {rna_text}",
            max_length=args["max_seq_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rna_sem_feat = extract_feature(model_bio, rna_sem_inputs, device)

    max_len = args["max_seq_len"]
    is_valid_2d = int(bool(dot_bracket) and "(" in dot_bracket)
    rna_2d_x = torch.zeros((max_len, 5), dtype=torch.float)
    rna_2d_edge_index = torch.empty((2, 0), dtype=torch.long)
    rna_2d_edge_type = torch.empty((0,), dtype=torch.long)

    if is_valid_2d:
        try:
            result_2d = rna2D_from_dot(
                rna_seq[:max_len].ljust(max_len, "N"),
                dot_bracket[:max_len].ljust(max_len, "."),
            )
            rna_2d_x, rna_2d_edge_index, rna_2d_edge_type = result_2d
        except Exception:
            is_valid_2d = 0
            rna_2d_x = torch.zeros((max_len, 5), dtype=torch.float)
            rna_2d_edge_index = torch.empty((2, 0), dtype=torch.long)
            rna_2d_edge_type = torch.empty((0,), dtype=torch.long)

    mol_mask = torch.tensor(
        [int(len(smiles) > 5), int(len(mol_text) > 5)],
        dtype=torch.float,
    )
    rna_mask = torch.tensor(
        [int(len(rna_seq) > 0), int(len(rna_text) > 5), is_valid_2d],
        dtype=torch.float,
    )

    return Data(
        mol_chem_feat=mol_chem_feat.squeeze(0),
        mol_sem_feat=mol_sem_feat.squeeze(0),
        rna_bert_feat=rna_bert_feat.squeeze(0),
        rna_sem_feat=rna_sem_feat.squeeze(0),
        rna_2d_x=rna_2d_x.float().contiguous(),
        rna_2d_edge_index=rna_2d_edge_index.long().contiguous(),
        rna_2d_edge_type=rna_2d_edge_type.long().contiguous(),
        mol_mask=mol_mask,
        rna_mask=rna_mask,
        y=torch.tensor([0.0], dtype=torch.float),
        num_nodes=rna_2d_x.size(0),
    )


def main():
    input_path = "prediction/example.xlsx"
    output_path = "prediction/prediction_results.xlsx"
    os.makedirs("prediction", exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Prediction input not found: {input_path}")
    if not os.path.exists(args["model_weight"]):
        raise FileNotFoundError(
            f"Model weight not found: {args['model_weight']}. "
            "Train the model first or update args['model_weight']."
        )

    df = pd.read_excel(input_path)
    print(f"[Data] Loaded {len(df)} prediction rows.")

    required_columns = {
        "SMILES",
        "Small molecule information",
        "1D Sequence",
        "RNA information",
        "Dot bracket",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Missing required prediction columns: "
            + ", ".join(sorted(missing_columns))
        )

    tokenizers, extractors = load_feature_extractors()

    args["mol_input_dim"] = 384
    args["rna_input_dim"] = 120
    args["bio_input_dim"] = 768

    model = MultiModalModel(args).to(device)
    state_dict = torch.load(args["model_weight"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    predictions = []
    print("[Predict] Running inference...")

    for idx, row in df.iterrows():
        data_obj = process_row(row, tokenizers, extractors).to(device)
        data_obj.batch = torch.zeros(
            data_obj.rna_2d_x.size(0),
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            output = model(data_obj)
            probability = torch.sigmoid(output["out"]).item()
            predictions.append(int(probability >= 0.5))

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
            print(f"  Completed: {idx + 1}/{len(df)}")

    df["Prediction"] = predictions
    df.to_excel(output_path, index=False)
    print(f"[Done] Prediction results saved to: {output_path}")


if __name__ == "__main__":
    main()

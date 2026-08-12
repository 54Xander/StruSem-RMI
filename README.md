# StruSem-RMI

**StruSem-RMI** is a knowledge-enhanced multimodal framework for RNA-small molecule interaction prediction built on the **RSID** benchmark. The model integrates RNA sequence, RNA secondary structure, RNA semantic information, small-molecule structure, and small-molecule semantic information within a dual-branch architecture.

The accompanying RSID benchmark contains **1,665 RNA-small molecule interaction records**, covering **211 RNAs**, **857 small molecules**, and **105 peer-reviewed publications**.

![StruSem-RMI framework](figures/framework.png)

## Overview

StruSem-RMI combines five input modalities:

- **RNA sequence** encoded by RNABERT
- **RNA secondary structure** represented as a relation-aware graph and encoded by RGCN
- **RNA semantic information** encoded by BioBERT
- **Small-molecule structure** encoded from SMILES by ChemBERTa
- **Small-molecule semantic information** encoded by BioBERT

The modality-specific representations are projected into a shared latent space, fused with mask-aware adaptive weighting, transformed through bidirectional cross-entity multi-head attention, and optimized using weighted binary cross-entropy with an auxiliary interaction contrastive loss.

In the five-fold warm-start evaluation reported in the accompanying manuscript, StruSem-RMI achieved:

| Metric | Performance |
|---|---:|
| AUC | 88.65 ± 0.50% |
| AUPR | 92.79 ± 0.59% |

## Repository Structure

```text
.
├── training.py
├── create_data.py
├── model.py
├── predict.py
├── utils.py
├── requirements.txt
├── environment.yml
├── figures/
│   └── framework.png
├── data/
│   └── RSID/
│       ├── Molecule.xlsx
│       ├── RNA.xlsx
│       └── RNA-Molecule.xlsx
├── pretrained_models/
│   ├── ChemBERTa-77M-MTR/
│   ├── biobert-base-cased-v1.2/
│   └── rnabert/
├── model/
├── result/
└── prediction/
    └── example.xlsx
```

## Requirements

The development environment uses Python 3.11 with PyTorch, PyTorch Geometric, Transformers, RDKit, scikit-learn, pandas, NumPy, and OpenPyXL.

Key package versions in the provided environment include:

- Python 3.11
- PyTorch 2.5.1
- PyTorch Geometric 2.6.1
- Transformers 4.46.3
- RDKit 2024.09
- scikit-learn 1.5.1
- pandas 2.0.3
- NumPy 1.26.4
- OpenPyXL 3.1.5

You can reproduce the development environment using the provided dependency files:

```bash
pip install -r requirements.txt
```

or

```bash
conda env create -f environment.yml
conda activate StruSem-RMI
```

> **Note:** The supplied environment files were exported from the original development environment and contain platform- and CUDA-specific packages. Some package paths or CUDA dependencies may need to be adjusted for your system.

## Pretrained Models

Download the following pretrained models and place them under `pretrained_models/`:

- [ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR)
- [BioBERT base cased v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
- [RNABERT](https://huggingface.co/yangheng/rnabert)

The expected directory layout is:

```text
pretrained_models/
├── ChemBERTa-77M-MTR/
├── biobert-base-cased-v1.2/
└── rnabert/
```

The scripts load pretrained models from local directories, so these files must be downloaded before feature generation, training, or prediction.

## Dataset Preparation

Place the RSID input files in:

```text
data/RSID/
├── Molecule.xlsx
├── RNA.xlsx
└── RNA-Molecule.xlsx
```

The training pipeline expects the following core fields.

### `Molecule.xlsx`

| Column | Description |
|---|---|
| `Small molecule_ID` | Unique small-molecule identifier |
| `SMILES` | Standardized molecular representation |
| `Small molecule information` | Small-molecule semantic description |

### `RNA.xlsx`

| Column | Description |
|---|---|
| `RNA_ID` | Unique RNA identifier |
| `1D Sequence` | RNA nucleotide sequence |
| `Dot bracket` | Sequence-aligned secondary structure |
| `RNA information` | RNA semantic description |

### `RNA-Molecule.xlsx`

| Column | Description |
|---|---|
| `RNA_ID` | RNA identifier |
| `Small molecule_ID` | Small-molecule identifier |
| `label` | Binary interaction label (`0` or `1`) |

## Training

Run:

```bash
python training.py
```

The training script automatically:

1. performs stratified five-fold splitting,
2. extracts and caches pretrained multimodal features when necessary,
3. trains the full StruSem-RMI model,
4. selects checkpoints according to validation AUC,
5. applies early stopping, and
6. saves fold-level evaluation results.

Offline features are cached under:

```text
data/processed/
```

Trained checkpoints are saved as:

```text
model/RSID_fold1.pt
model/RSID_fold2.pt
...
model/RSID_fold5.pt
```

Training metrics are written to:

```text
result/
```

Common training options can be changed from the command line, for example:

```bash
python training.py --epochs 100 --batch_size 32 --lr 5e-4 --n_splits 5
```

## Prediction

By default, prediction reads:

```text
prediction/example.xlsx
```

The input file should contain the following columns:

| Column | Description |
|---|---|
| `SMILES` | Small-molecule SMILES |
| `Small molecule information` | Small-molecule semantic description |
| `1D Sequence` | RNA sequence |
| `RNA information` | RNA semantic description |
| `Dot bracket` | RNA secondary structure |

Before prediction, make sure the checkpoint path in `predict.py` points to the desired trained model. The default is:

```text
model/RSID_fold1.pt
```

Run:

```bash
python predict.py
```

Predictions are saved to:

```text
prediction/prediction_results.xlsx
```

The output includes a `Prediction` column:

- `0`: predicted non-interaction
- `1`: predicted interaction

## RSID

RSID is a manually curated, literature-traceable benchmark for RNA-small molecule interaction prediction. It integrates RNA sequences, sequence-aligned secondary structures, standardized molecular representations, quantitative interaction evidence, and entity-indexed semantic annotations.

RSID web platform: [http://rsid.hzau.edu.cn](http://rsid.hzau.edu.cn)

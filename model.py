import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation, RGCNConv


class PretrainedBranch(nn.Module):
    """Project an offline pretrained feature into the shared model space."""

    def __init__(self, input_dim=768, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class RNA_RGCN_Advanced(nn.Module):
    """Encode RNA secondary-structure graphs with an RGCN."""

    def __init__(self, args):
        super().__init__()
        self.dim = args["embed_dim"]
        self.num_relations = 9

        self.conv1 = RGCNConv(
            5,
            self.dim,
            num_relations=self.num_relations,
        )
        self.conv2 = RGCNConv(
            self.dim,
            self.dim,
            num_relations=self.num_relations,
        )

        self.gate_nn = nn.Linear(self.dim, 1)
        self.pool = AttentionalAggregation(
            gate_nn=self.gate_nn
        )
        self.fc = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        x,
        edge_index,
        edge_type,
        batch,
    ):
        x = x.float().contiguous()
        edge_index = edge_index.long()
        edge_type = edge_type.long()

        x = self.conv1(
            x,
            edge_index,
            edge_type,
        )
        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index,
            edge_type,
        )
        x = F.relu(x)

        pooled = self.pool(x, batch)
        return self.fc(pooled)


class DynamicWeightNet(nn.Module):
    """Learn sample-specific weights for available modalities."""

    def __init__(self, input_dim, num_modalities):
        super().__init__()
        self.fc = nn.Linear(
            input_dim,
            num_modalities,
        )

    def forward(self, x, mask=None):
        logits = self.fc(x)

        if mask is not None:
            mask = mask.view(logits.size(0), -1)
            logits = logits.masked_fill(
                mask == 0,
                -1e9,
            )

        return F.softmax(logits, dim=1)


class MultiModalModel(nn.Module):
    """Full multimodal RNA-small-molecule interaction model."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args["embed_dim"]

        chem_dim = args.get("mol_input_dim", 384)
        bio_dim = args.get("bio_input_dim", 768)
        rna_dim = args.get("rna_input_dim", 120)

        print(
            "[Model] Adaptive initialization: "
            f"Chem={chem_dim}, "
            f"Bio={bio_dim}, "
            f"RNA={rna_dim}"
        )

        self.mol_chem = PretrainedBranch(
            input_dim=chem_dim,
            embed_dim=self.embed_dim,
        )
        self.mol_bio = PretrainedBranch(
            input_dim=bio_dim,
            embed_dim=self.embed_dim,
        )
        self.mol_weight_net = DynamicWeightNet(
            input_dim=2 * self.embed_dim,
            num_modalities=2,
        )

        self.rnabert = PretrainedBranch(
            input_dim=rna_dim,
            embed_dim=self.embed_dim,
        )
        self.rna_bio = PretrainedBranch(
            input_dim=bio_dim,
            embed_dim=self.embed_dim,
        )
        self.rna_rgcn = RNA_RGCN_Advanced(args)
        self.rna_weight_net = DynamicWeightNet(
            input_dim=3 * self.embed_dim,
            num_modalities=3,
        )

        self.MHA_m_from_r = nn.MultiheadAttention(
            self.embed_dim,
            args["nhead"],
            batch_first=True,
        )
        self.MHA_r_from_m = nn.MultiheadAttention(
            self.embed_dim,
            args["nhead"],
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.embed_dim * 2,
                self.embed_dim,
            ),
            nn.ReLU(),
            nn.Dropout(args["dropout"]),
            nn.Linear(
                self.embed_dim,
                args["n_output"],
            ),
        )

    def forward(self, data):
        batch_size = data.y.size(0)

        mol_features = [
            self.mol_chem(
                data.mol_chem_feat.view(
                    batch_size,
                    -1,
                )
            ),
            self.mol_bio(
                data.mol_sem_feat.view(
                    batch_size,
                    -1,
                )
            ),
        ]

        mol_mask = getattr(data, "mol_mask", None)
        if mol_mask is not None:
            mol_mask = mol_mask.view(
                batch_size,
                2,
            )

        mol_weights = self.mol_weight_net(
            torch.cat(mol_features, dim=1),
            mask=mol_mask,
        ).unsqueeze(2)
        mol_feat = torch.sum(
            torch.stack(mol_features, dim=1)
            * mol_weights,
            dim=1,
        )

        rna_features = [
            self.rnabert(
                data.rna_bert_feat.view(
                    batch_size,
                    -1,
                )
            ),
            self.rna_bio(
                data.rna_sem_feat.view(
                    batch_size,
                    -1,
                )
            ),
            self.rna_rgcn(
                data.rna_2d_x,
                data.rna_2d_edge_index,
                data.rna_2d_edge_type,
                data.batch,
            ),
        ]

        rna_mask = getattr(data, "rna_mask", None)
        if rna_mask is not None:
            rna_mask = rna_mask.view(
                batch_size,
                3,
            )

        rna_weights = self.rna_weight_net(
            torch.cat(rna_features, dim=1),
            mask=rna_mask,
        ).unsqueeze(2)
        rna_feat = torch.sum(
            torch.stack(rna_features, dim=1)
            * rna_weights,
            dim=1,
        )

        mol_context = self.MHA_m_from_r(
            mol_feat.unsqueeze(1),
            rna_feat.unsqueeze(1),
            rna_feat.unsqueeze(1),
        )[0].squeeze(1)

        rna_context = self.MHA_r_from_m(
            rna_feat.unsqueeze(1),
            mol_feat.unsqueeze(1),
            mol_feat.unsqueeze(1),
        )[0].squeeze(1)

        output = self.classifier(
            torch.cat(
                [mol_context, rna_context],
                dim=1,
            )
        )

        return {
            "out": output,
            "drug_feat": mol_context,
            "rna_feat": rna_context,
            "mol_weights": mol_weights.squeeze(2),
            "rna_weights": rna_weights.squeeze(2),
        }


class InteractionContrastiveLoss(nn.Module):
    """Contrast paired RNA and molecule embeddings for positive samples."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        rna_emb,
        drug_emb,
        labels,
    ):
        rna_emb = F.normalize(
            rna_emb,
            p=2,
            dim=1,
        )
        drug_emb = F.normalize(
            drug_emb,
            p=2,
            dim=1,
        )

        logits = (
            torch.matmul(rna_emb, drug_emb.T)
            / self.temperature
        )
        positive_mask = (labels == 1).float()

        if positive_mask.sum() == 0:
            return torch.tensor(
                0.0,
                device=rna_emb.device,
                requires_grad=True,
            )

        batch_size = rna_emb.size(0)
        target = torch.arange(
            batch_size,
            device=rna_emb.device,
        )

        loss_rna = F.cross_entropy(
            logits,
            target,
            reduction="none",
        )
        loss_drug = F.cross_entropy(
            logits.T,
            target,
            reduction="none",
        )

        interaction_loss = (
            (loss_rna + loss_drug) / 2
        ) * positive_mask

        return interaction_loss.sum() / (
            positive_mask.sum() + 1e-7
        )

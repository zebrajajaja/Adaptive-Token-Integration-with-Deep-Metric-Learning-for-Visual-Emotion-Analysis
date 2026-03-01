import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, CLIPVisionModel


class FusionBlock(nn.Module):


    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_cross = nn.LayerNorm(hidden_dim)
        # Self Attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_self = nn.LayerNorm(hidden_dim)

    def forward(self, visual_feats, text_feats, text_padding_mask=None):
        batch_size = visual_feats.size(0)

        # 1. Construct KV: Concat[Visual_CLS, Text]
        visual_cls = visual_feats[:, 0:1, :]
        kv_feats = torch.cat([visual_cls, text_feats], dim=1)

        # Mask handling
        cls_mask = torch.zeros((batch_size, 1), device=visual_feats.device, dtype=torch.bool)
        if text_padding_mask is not None:
            kv_padding_mask = torch.cat([cls_mask, text_padding_mask], dim=1)
        else:
            kv_padding_mask = None

        # 2. Cross Attention
        attn_out, _ = self.cross_attn(
            query=visual_feats,
            key=kv_feats,
            value=kv_feats,
            key_padding_mask=kv_padding_mask
        )
        v_enriched = self.ln_cross(visual_feats + attn_out)

        # 3. Self Attention
        attn_out, _ = self.self_attn(
            query=v_enriched,
            key=v_enriched,
            value=v_enriched
        )
        v_final = self.ln_self(v_enriched + attn_out)

        return v_final


class APSEModel(nn.Module):
    def __init__(self,
                 num_classes,
                 text_model_name='roberta-base',
                 visual_model_name='openai/clip-vit-base-patch32',
                 fusion_layers=1):
        super().__init__()
        # Load Backbones
        self.text_encoder = RobertaModel.from_pretrained(text_model_name)
        self.visual_encoder = CLIPVisionModel.from_pretrained(visual_model_name)

        self.hidden_dim = 768

        # Fusion Layers
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(hidden_dim=self.hidden_dim) for _ in range(fusion_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)


        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, images, desc_input_ids, desc_mask):
        # 1. Extract Features
        v_out = self.visual_encoder(images).last_hidden_state
        t_out = self.text_encoder(input_ids=desc_input_ids, attention_mask=desc_mask).last_hidden_state
        padding_mask = ~desc_mask.bool()

        # 2. Fusion
        x = v_out
        for block in self.fusion_blocks:
            x = block(visual_feats=x, text_feats=t_out, text_padding_mask=padding_mask)

        features = self.pool(x.transpose(1, 2)).squeeze(2)

        # 4. Classification
        # [Batch, Dim] -> [Batch, Num_Classes]
        logits = self.classifier(features)

        return logits
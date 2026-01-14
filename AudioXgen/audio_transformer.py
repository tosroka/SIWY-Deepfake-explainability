import torch
import torch.nn as nn


class AudioTransformer(nn.Module):
    def __init__(
        self,
        feature_size=128,
        seq_length=375,
        num_classes=2,
        model_dim=256,
        nhead=8,
        num_layers=3,
        dropout=0.1,
    ):
        super(AudioTransformer, self).__init__()
        self.embedding = torch.nn.Linear(feature_size, model_dim)
        self.pos_encoder = torch.nn.Parameter(torch.randn(1, seq_length, model_dim))

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_layer = torch.nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x += self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        return x

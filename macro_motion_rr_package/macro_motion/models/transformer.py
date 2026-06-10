import torch
import torch.nn as nn

class SharedMacroTransformer(nn.Module):
    DOMAIN_RR = 0
    DOMAIN_TEMP = 1
    COND_EXPOSED = 0
    COND_BLANKET = 1

    def __init__(self, d_model=512, n_heads=8, depth=6, dropout=0.1, ff_mult=4, max_len=700):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult*d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.domain_embed = nn.Embedding(2, d_model)
        self.cond_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

    def make_seq(self, z, domain_id: int, cond_id):
        B,T,D = z.shape
        cls = self.cls.expand(B, 1, D)
        dom_ids = torch.full((B,1), int(domain_id), device=z.device, dtype=torch.long)
        dom = self.domain_embed(dom_ids)
        cond = self.cond_embed(cond_id.view(B,1).to(z.device))
        seq = torch.cat([cls, dom, cond, z], dim=1)
        return seq + self.pos_embed[:, :T+3, :]

    def forward_once(self, z, domain_id: int, cond_id):
        out = self.encoder(self.make_seq(z, domain_id, cond_id))
        return out[:, 0, :]

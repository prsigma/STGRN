import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from einops import rearrange

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, attention_dropout):
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        if attn_mask is None:
            attn_mask = TriangularCausalMask(B, L, device=queries.device)
        attn_mask = attn_mask.to(dtype=torch.bool)
        scores.masked_fill_(attn_mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return (V.contiguous(), A)
        
class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class Segment_embedding(nn.Module):
    def __init__(self, seg_len, d_model,dropout):
        super(Segment_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x_segment = rearrange(x, 'b (seg_num seg_len) gene -> b gene seg_num seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        
        return self.dropout(x_embed)

class TimeLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(TimeLayer, self).__init__()
        self.time_attention = AttentionLayer(d_model, n_heads,dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        
    def forward(self, x):
        batch = x.shape[0]
        time_in = rearrange(x, 'b gene_num seg_num d_model -> (b gene_num) seg_num d_model')
        
        #mask, the previous time couldn't notice the time after.
        seg_num = time_in.shape[1]
        time_mask = torch.ones((seg_num,seg_num))
        time_mask = torch.triu(time_mask, diagonal=1).to(device=time_in.device)

        time_enc, _ = self.time_attention(
            time_in, time_in, time_in, time_mask
        )

        out = time_in + self.dropout(time_enc)
        out = self.norm1(out)
        out = out + self.dropout(self.MLP(out))
        out = self.norm2(out)

        final_out = rearrange(out, '(b gene_num) seg_num d_model -> b seg_num gene_num d_model', b=batch)

        return final_out

class SpatialLayer(nn.Module):
    def __init__(self,d_model, n_heads, d_ff, dropout) -> None:
        super().__init__()
        self.heads = n_heads
        self.spatial_attention = AttentionLayer(d_model, n_heads,dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
    
    def forward(self,x,prior_mask):
        batch = x.shape[0]

        spatial_in = rearrange(x, 'b seg_num gene_num d_model -> (b seg_num) gene_num d_model')
        
        spatial_enc, attention = self.spatial_attention(
            spatial_in, spatial_in, spatial_in, prior_mask
        )

        attn_scores = attention.mean(1)

        out = spatial_in + self.dropout(spatial_enc)
        out = self.norm1(out)
        out = out + self.dropout(self.MLP(out))
        out = self.norm2(out)

        final_out = rearrange(out, '(b seg_num) gene_num d_model -> b seg_num gene_num d_model',b = batch)

        return final_out, attn_scores
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.time_encoder = TimeLayer(d_model, heads, d_ff, dropout)
        self.spatial_encoder = SpatialLayer(d_model, heads, d_ff, dropout)

    def forward(self, x, prior_mask):
        time_out = self.time_encoder(x)
        spatial_out,attention = self.spatial_encoder(time_out,prior_mask)
        
        return spatial_out, attention

class Encoder(nn.Module):
    def __init__(self, encoder_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x, prior_mask):
        attns = []
        for encoder_layer in self.encoder_layers:
            x, attn = encoder_layer(x, prior_mask)
            x = rearrange(x, 'b seg_num gene_num d_model -> b gene_num seg_num d_model')
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class STGRN(nn.Module):
    def __init__(self, gene_num, in_len, out_len, seg_len, d_model, d_ff, n_heads, dropout, e_layers, device):
        super(STGRN, self).__init__()
        self.gene_num = gene_num
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.heads = n_heads
        self.dropout = dropout
        self.device = device
        self.e_layers = e_layers
        self.input_seg_num = int(self.in_len / self.seg_len)
        self.output_seg_num = int(self.out_len / self.seg_len)

        # Embedding
        self.enc_value_embedding = Segment_embedding(self.seg_len, self.d_model,self.dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.gene_num, self.input_seg_num, self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    self.d_model, self.heads, self.d_ff, self.dropout
                ) for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projector = nn.Linear(self.d_model, self.seg_len, bias=True)
    
    def forward(self, x, prior_mask):
        x = self.enc_value_embedding(x)     #output: batch gene seg_num d_model
        x += self.pos_embedding
        x = self.pre_norm(x)     
        
        prior_mask = torch.from_numpy(prior_mask).to(device=x.device)
        x, spatial_attention = self.encoder(x,prior_mask)   #output: b  gene_num seg_num d_model

        out = self.projector(x) #b gene_num seg_num seg_len
        out = rearrange(out, 'b gene_num seg_num output_len -> b (seg_num output_len) gene_num')

        return out, spatial_attention
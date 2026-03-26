from typing import *
from torch import Tensor, LongTensor

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


class Attention(nn.Module):
    def __init__(self,
        query_dim: int, context_dim: Optional[int]=None,
        n_heads=8, hidden_dim=512, dropout=0.
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"Hidden dimension ({hidden_dim}) must be divisible by number of heads ({n_heads})"
        head_dim = hidden_dim // n_heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = head_dim ** -0.5
        self.n_heads = n_heads

        self.to_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self,
        x: Tensor,
        context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None
    ):
        h = self.n_heads

        q = self.to_q(x)
        if mask is not None:
            q = q * mask.unsqueeze(-1)

        if context is None:
            context = x
            context_mask = mask

        k = self.to_k(context)
        v = self.to_v(context)
        if context_mask is not None:
            k = k * context_mask.unsqueeze(-1)
            v = v * context_mask.unsqueeze(-1)

        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> (b h) n d", h=h), (q, k, v))



        
        attn_mask = None
        if context_mask is not None:
            attn_mask = ~context_mask.bool().reshape(b, 1, 1, -1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self.to_out(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out


class GraphAttention(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int, global_dim: Optional[int]=None,
        n_heads=8, hidden_dim=512, dropout=0.
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"Hidden dimension ({hidden_dim}) must be divisible by number of heads ({n_heads})"
        head_dim = hidden_dim // n_heads

        self.scale = head_dim ** -0.5
        self.n_heads = n_heads

        self.to_q = nn.Linear(node_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(node_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(node_dim, hidden_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, node_dim),
            nn.Dropout(dropout)
        )

        self.to_e_mul = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.to_e_add = nn.Linear(edge_dim, hidden_dim, bias=False)

        self.to_e_out = nn.Sequential(
            nn.Linear(hidden_dim, edge_dim),
            nn.Dropout(dropout)
        )

        if global_dim is not None:
            self.to_y_x_mul = nn.Linear(global_dim, hidden_dim, bias=False)
            self.to_y_x_add = nn.Linear(global_dim, hidden_dim, bias=False)
            self.to_y_e_mul = nn.Linear(global_dim, hidden_dim, bias=False)
            self.to_y_e_add = nn.Linear(global_dim, hidden_dim, bias=False)

            self.to_yx_out = nn.Linear(node_dim, global_dim)
            self.to_ye_out = nn.Linear(edge_dim, global_dim)

        self.use_global_info = global_dim is not None

    def forward(self,
        x: Tensor, e: Tensor, y: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None
    ):
        h = self.n_heads
        if mask is not None:
            x_mask = mask.unsqueeze(-1)
            e_mask1 = x_mask.unsqueeze(-1)
            e_mask2 = x_mask.unsqueeze(1)

        q = self.to_q(x)
        if mask is not None:
            q = q * x_mask

        k = self.to_k(x)
        v = self.to_v(x)
        if mask is not None:
            k = k * x_mask
            v = v * x_mask

        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim: Tensor = q.unsqueeze(2) * k.unsqueeze(1) * self.scale

        e_mul = self.to_e_mul(e)
        e_add = self.to_e_add(e)
        if mask is not None:
            e_mul = e_mul * e_mask1 * e_mask2
            e_add = e_add * e_mask1 * e_mask2
        e_mul, e_add = map(lambda t: rearrange(
            t, "b n m (h d) -> (b h) n m d", h=h), (e_mul, e_add))
        sim = (1. + e_mul) * sim + e_add

        e_out = rearrange(sim, "(b h) n m d -> b n m (h d)", h=h)
        if self.use_global_info:
            ye_mul = self.to_y_e_mul(y).unsqueeze(1).unsqueeze(1)
            ye_add = self.to_y_e_add(y).unsqueeze(1).unsqueeze(1)
            e_out = (1. + ye_mul) * e_out + ye_add

        e_out = self.to_e_out(e_out)
        if mask is not None:
            e_out = e_out * e_mask1 * e_mask2

        if mask is not None:
            attn_mask = e_mask2.expand(-1, q.shape[1], -1, h)
            attn_mask = rearrange(attn_mask, "b n m h -> (b h) n m ()").bool()
            sim = sim.masked_fill(~attn_mask, float("-inf"))
        attn = sim.softmax(dim=2)

        out = (attn * v.unsqueeze(1)).sum(dim=2)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        if self.use_global_info:
            yx_mul = self.to_y_x_mul(y).unsqueeze(1)
            yx_add = self.to_y_x_add(y).unsqueeze(1)
            out = (1. + yx_mul) * out + yx_add

        out = self.to_out(out)
        if mask is not None:
            out = out * x_mask

        if self.use_global_info:
            if mask is not None:
                yx_out = out.sum(dim=1) / x_mask.expand(-1, -1, out.shape[-1]).sum(dim=1)
                ye_out = e_out.sum(dim=(1, 2)) / \
                    (e_mask1 * e_mask2).expand(-1, -1, -1, e_out.shape[-1]).sum(dim=(1, 2))
            else:
                yx_out = out.mean(dim=1)
                ye_out = e_out.mean(dim=(1, 2))
            yx_out = self.to_yx_out(yx_out)
            ye_out = self.to_ye_out(ye_out)
            y_out = y + yx_out + ye_out
        else:
            y_out = None

        return out, e_out, y_out


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out*2)

    def forward(self, x: Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self,
        dim: int, dim_out: Optional[int]=None,
        mult=4, gated=False, dropout=0.
    ):
        super().__init__()
        hidden_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU()
        ) if not gated else GEGLU(dim, hidden_dim)

        self.mlp = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out)
        )

    def forward(self, x: Tensor):
        return self.mlp(x)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, t_dim: int):
        super().__init__()
        self.gelu = nn.GELU()
        self.linear = nn.Linear(t_dim, dim*2)
        self.layernorm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: Tensor, t_emb: Tensor):
        emb: Tensor = self.linear(self.gelu(t_emb)).unsqueeze(1)
        while emb.dim() < x.dim():
            emb = emb.unsqueeze(1)

        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.layernorm(x) * (1. + scale) + shift

        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self,
        dim: int, attn_dim: int,
        context_dim: Optional[int]=None, t_dim: Optional[int]=None,
        n_heads=8, gated_ff=True, dropout=0., ada_norm=False
    ):
        super().__init__()
        if ada_norm:
            assert t_dim is not None, "Parameter `t_dim` must be provided for AdaLN"

        self.attn = Attention(dim, None, n_heads, attn_dim, dropout)
        self.ff = FeedForward(dim, gated=gated_ff, dropout=dropout)

        self.attn_norm = AdaLayerNorm(dim, t_dim) if ada_norm else nn.LayerNorm(dim)
        self.ff_norm = AdaLayerNorm(dim, t_dim) if ada_norm else nn.LayerNorm(dim)

        if context_dim is not None:
            self.cross_attn = Attention(dim, context_dim, n_heads, attn_dim, dropout)
            self.ca_norm = AdaLayerNorm(dim, t_dim) if ada_norm else nn.LayerNorm(dim)

    def forward(self,
        x: Tensor, t_emb: Optional[Tensor]=None,
        context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None
    ):
        x_norm = self.attn_norm(x, t_emb) if t_emb is not None else self.attn_norm(x)
        x = self.attn(x_norm, None, mask) + x

        if context is not None:
            x_norm = self.ca_norm(x, t_emb) if t_emb is not None else self.ca_norm(x)
            x = self.cross_attn(x_norm, context, mask, context_mask) + x

        x_norm = self.ff_norm(x, t_emb) if t_emb is not None else self.ff_norm(x)
        x = self.ff(x_norm) + x

        return x


class GraphTransformerBlock(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int, attn_dim: int, global_dim: Optional[int]=None,
        context_dim: Optional[int]=None, t_dim: Optional[int]=None,
        n_heads=8, gated_ff=True, dropout=0., ada_norm=False,
        use_e_cross_attn=False
    ):
        super().__init__()
        if ada_norm:
            assert t_dim is not None, "Parameter `t_dim` must be provided for AdaLN"
        self.graph_attn = GraphAttention(node_dim, edge_dim, global_dim, n_heads, attn_dim, dropout)
        self.ff_x = FeedForward(node_dim, gated=gated_ff, dropout=dropout)
        self.ff_e = FeedForward(edge_dim, gated=gated_ff, dropout=dropout)

        self.ga_x_norm = AdaLayerNorm(node_dim, t_dim) if ada_norm else nn.LayerNorm(node_dim)
        self.ff_x_norm = AdaLayerNorm(node_dim, t_dim) if ada_norm else nn.LayerNorm(node_dim)
        self.ga_e_norm = AdaLayerNorm(edge_dim, t_dim) if ada_norm else nn.LayerNorm(edge_dim)
        self.ff_e_norm = AdaLayerNorm(edge_dim, t_dim) if ada_norm else nn.LayerNorm(edge_dim)

        if context_dim is not None:
            self.cross_attn = Attention(node_dim, context_dim, n_heads, attn_dim, dropout)
            self.ca_norm = AdaLayerNorm(node_dim, t_dim) if ada_norm else nn.LayerNorm(node_dim)
            if use_e_cross_attn:
                self.cross_attn_e = Attention(edge_dim, context_dim, n_heads, attn_dim, dropout)
                self.ca_norm_e = AdaLayerNorm(edge_dim, t_dim) if ada_norm else nn.LayerNorm(edge_dim)

        if global_dim is not None:
            self.ff_y = FeedForward(global_dim, gated=gated_ff, dropout=dropout)
            self.ga_y_norm = AdaLayerNorm(global_dim, t_dim) if ada_norm else nn.LayerNorm(global_dim)
            self.ff_y_norm = AdaLayerNorm(global_dim, t_dim) if ada_norm else nn.LayerNorm(global_dim)

        self.ada_norm = ada_norm
        self.use_e_cross_attn = use_e_cross_attn
        self.with_global_info = global_dim is not None

    def forward(self,
        x: Tensor, e: Tensor, y: Optional[Tensor]=None,
        t_emb: Optional[Tensor]=None, context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None
    ):
        x_norm = self.ga_x_norm(x, t_emb) if self.ada_norm else self.ga_x_norm(x)
        e_norm = self.ga_e_norm(e, t_emb) if self.ada_norm else self.ga_e_norm(e)
        if self.with_global_info:
            y_norm = self.ga_y_norm(y, t_emb) if self.ada_norm else self.ga_y_norm(y)
        else:
            y_norm = None
        x_, e_, y_ = self.graph_attn(x_norm, e_norm, y_norm, mask)
        x, e = x_ + x, e_ + e
        if y_ is not None:
            y = y_ + y

        if context is not None:
            x_norm = self.ca_norm(x, t_emb) if self.ada_norm else self.ca_norm(x)
            x = self.cross_attn(x_norm, context, mask, context_mask) + x
            if self.use_e_cross_attn:
                e_norm = self.ca_norm_e(e, t_emb) if self.ada_norm else self.ca_norm_e(e)
                e_norm = rearrange(e_norm, "b n m d -> b (n m) d")
                e_ = self.cross_attn_e(e_norm, context, mask, context_mask)
                e_ = rearrange(e_, "b (n m) d -> b n m d", n=e.shape[1])
                e = e_ + e

        x_norm = self.ff_x_norm(x, t_emb) if self.ada_norm else self.ff_x_norm(x)
        x = self.ff_x(x_norm) + x
        e_norm = self.ff_e_norm(e, t_emb) if self.ada_norm else self.ff_e_norm(e)
        e = self.ff_e(e_norm) + e
        if self.with_global_info:
            y_norm = self.ff_y_norm(y, t_emb) if self.ada_norm else self.ff_y_norm(y)
            y = self.ff_y(y_norm) + y
        else:
            y = None

        return x, e, y
    
    
class MultiModalGraphTransformerBlock(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int, attn_dim: int, global_dim: Optional[int]=None,
        context_dim: Optional[int]=None, t_dim: Optional[int]=None,
        n_heads=8, gated_ff=True, dropout=0., ada_norm=False,
        use_e_cross_attn=False
    ):
        super().__init__()
        
        self.miche_to_box = Attention(query_dim=node_dim, context_dim=node_dim, n_heads=1, hidden_dim=attn_dim//2, dropout=dropout)
        self.box_to_miche = Attention(query_dim=node_dim, context_dim=node_dim, n_heads=n_heads//2, hidden_dim=attn_dim//2, dropout=dropout)
        self.uni3d_to_shape = Attention(query_dim=node_dim, context_dim=node_dim, n_heads=n_heads//2, hidden_dim=attn_dim//2, dropout=dropout)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.norm3 = nn.LayerNorm(node_dim)
        
        self.graph_layer = GraphTransformerBlock(node_dim=node_dim, edge_dim=edge_dim, attn_dim=attn_dim, global_dim=global_dim, context_dim=context_dim, t_dim=t_dim, n_heads=n_heads, gated_ff=gated_ff, dropout=dropout, ada_norm=ada_norm)

        self.box_to_x = Attention(query_dim=node_dim, context_dim=node_dim, n_heads=1, hidden_dim=attn_dim//2, dropout=dropout)
        self.miche_to_x = Attention(query_dim=node_dim, context_dim=node_dim, n_heads=1, hidden_dim=attn_dim//2, dropout=dropout)
        
        self.norm4 = nn.LayerNorm(node_dim)
        self.norm5 = nn.LayerNorm(node_dim)

    def forward(self,
        box_emb: Tensor, miche_emb: Tensor, uni3d_emb: Tensor, e: Tensor, y: Optional[Tensor]=None,
        t_emb: Optional[Tensor]=None, context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None, batch_size = None
    ):
        B, _, D = box_emb.shape
        N = B // batch_size

        miche_emb = self.norm1(miche_emb + self.miche_to_box(miche_emb, box_emb))
        box_emb = self.norm2(box_emb + self.box_to_miche(box_emb, miche_emb))
        shape_emb = torch.cat([miche_emb, box_emb], dim=1)
        x = self.norm3(self.uni3d_to_shape(uni3d_emb, shape_emb)).view(batch_size,N,D)

        x, e, y = self.graph_layer(x, e, y, t_emb, context, mask, context_mask)
        
        x = x.view(B,1,D)
        
        miche_emb = self.norm4(miche_emb + self.miche_to_x(miche_emb, x))
        box_emb = self.norm5(box_emb + self.box_to_x(box_emb, x))
        
        return box_emb, miche_emb, uni3d_emb, e, y


class SimpleMultiModalGraphTransformerBlock(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int, attn_dim: int, global_dim: Optional[int]=None,
        context_dim: Optional[int]=None, t_dim: Optional[int]=None,
        n_heads=8, gated_ff=True, dropout=0., ada_norm=False,
        use_e_cross_attn=False, use_pc = False, use_fixed_miche = False
    ):
        super().__init__()
        
        self.use_pc = use_pc
        self.use_fixed_miche = use_fixed_miche

        self.box_to_miche = Attention(query_dim=node_dim, context_dim=node_dim, n_heads=n_heads, hidden_dim=attn_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(node_dim)
        
        if use_pc:
            self.box_to_pc = Attention(query_dim=node_dim, context_dim=node_dim, n_heads=n_heads, hidden_dim=attn_dim, dropout=dropout)
            self.norm_pc = nn.LayerNorm(node_dim)
            
        
        self.graph_layer = GraphTransformerBlock(node_dim=node_dim, edge_dim=edge_dim, attn_dim=attn_dim, global_dim=global_dim, context_dim=context_dim, t_dim=t_dim, n_heads=n_heads, gated_ff=gated_ff, dropout=dropout, ada_norm=ada_norm)

        if not self.use_fixed_miche:
            self.miche_to_box = Attention(query_dim=node_dim, context_dim=node_dim, n_heads=1, hidden_dim=attn_dim, dropout=dropout)
        
            self.norm2 = nn.LayerNorm(node_dim)

    def forward(self,
        box_emb: Tensor, miche_emb: Tensor, pc_emb: Tensor, e: Tensor, y: Optional[Tensor]=None,
        t_emb: Optional[Tensor]=None, context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None, batch_size = None
    ):
        B, _, D = box_emb.shape
        N = B // batch_size

        box_emb = self.norm1(box_emb + self.box_to_miche(box_emb, miche_emb)).view(batch_size,N,D)
        
        if self.use_pc and pc_emb is not None:
            box_emb = self.norm_pc(box_emb + self.box_to_pc(box_emb, pc_emb)).view(batch_size,N,D)

        box_emb, e, y = self.graph_layer(box_emb, e, y, t_emb, context, mask, context_mask)
        
        box_emb = box_emb.view(B,1,D)
        
        if not self.use_fixed_miche:
            miche_emb = self.norm2(miche_emb + self.miche_to_box(miche_emb, box_emb))
        
        return box_emb, miche_emb, pc_emb, e, y
    
class SimpleMultiModalGraphTransformerBlock_nopc(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int, attn_dim: int, global_dim: Optional[int]=None,
        context_dim: Optional[int]=None, t_dim: Optional[int]=None,
        n_heads=8, gated_ff=True, dropout=0., ada_norm=False,
        use_e_cross_attn=False, use_pc = False
    ):
        super().__init__()
        
        self.use_pc = use_pc
        
        self.box_to_miche = Attention(query_dim=node_dim, context_dim=node_dim, n_heads=n_heads, hidden_dim=attn_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(node_dim)
        
        self.graph_layer = GraphTransformerBlock(node_dim=node_dim, edge_dim=edge_dim, attn_dim=attn_dim, global_dim=global_dim, context_dim=context_dim, t_dim=t_dim, n_heads=n_heads, gated_ff=gated_ff, dropout=dropout, ada_norm=ada_norm)

    def forward(self,
        box_emb: Tensor, miche_emb: Tensor, pc_emb: Tensor, e: Tensor, y: Optional[Tensor]=None,
        t_emb: Optional[Tensor]=None, context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None, batch_size = None
    ):
        B, _, D = box_emb.shape
        N = B // batch_size

        box_emb = self.norm1(box_emb + self.box_to_miche(box_emb, miche_emb)).view(batch_size,N,D)
        
        box_emb, e, y = self.graph_layer(box_emb, e, y, t_emb, context, mask, context_mask)
        
        box_emb = box_emb.view(B,1,D)
        
        return box_emb, miche_emb, pc_emb, e, y
    
class SimpleMultiModalGraphTransformerBlock_nomiche(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int, attn_dim: int, global_dim: Optional[int]=None,
        context_dim: Optional[int]=None, t_dim: Optional[int]=None,
        n_heads=8, gated_ff=True, dropout=0., ada_norm=False,
        use_e_cross_attn=False, use_pc = True, use_fixed_miche = False
    ):
        super().__init__()
        
        self.use_pc = use_pc
        self.use_fixed_miche = use_fixed_miche

        self.box_to_pc = Attention(query_dim=node_dim, context_dim=node_dim, n_heads=n_heads, hidden_dim=attn_dim, dropout=dropout)
        self.norm_pc = nn.LayerNorm(node_dim)
        
        self.graph_layer = GraphTransformerBlock(node_dim=node_dim, edge_dim=edge_dim, attn_dim=attn_dim, global_dim=global_dim, context_dim=context_dim, t_dim=t_dim, n_heads=n_heads, gated_ff=gated_ff, dropout=dropout, ada_norm=ada_norm)

    def forward(self,
        box_emb: Tensor, miche_emb: Tensor, pc_emb: Tensor, e: Tensor, y: Optional[Tensor]=None,
        t_emb: Optional[Tensor]=None, context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None, batch_size = None
    ):
        B, _, D = box_emb.shape
        N = B // batch_size

        box_emb = self.norm_pc(box_emb + self.box_to_pc(box_emb, pc_emb))

        box_emb, e, y = self.graph_layer(box_emb, e, y, t_emb, context, mask, context_mask)
        
        
        return box_emb, miche_emb, pc_emb, e, y

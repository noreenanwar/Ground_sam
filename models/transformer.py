#!/usr/bin/env python
"""
Modified Transformer module with Dual-Stream Cross-Attention in the Decoder.

This module defines a transformer architecture with encoder and decoder layers,
including a new dual-stream decoder layer that separately processes semantic and spatial cues.
"""

import math
import copy
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import logging

# Optional: if you have a custom multi-head attention, ensure it is correctly imported.
# from .attention import MultiheadAttention  # Adjust if needed

# Configure module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Change to DEBUG to see detailed output.
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def add_with_pos(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
    """
    Adds a positional embedding to a tensor. If the last dimension of `pos`
    does not match the last dimension of `tensor` but divides it exactly,
    the positional embedding is repeated to match the size.
    """
    if pos is None:
        return tensor
    if pos.size(2) != tensor.size(2):
        if tensor.size(2) % pos.size(2) == 0:
            factor = tensor.size(2) // pos.size(2)
            pos = pos.repeat(1, 1, factor)
        else:
            raise ValueError(
                f"Dimension mismatch: tensor has {tensor.size(2)} channels but "
                f"pos has {pos.size(2)} channels and cannot be repeated to match."
            )
    return tensor + pos


class MLP(nn.Module):
    """Simple multi-layer perceptron (FFN)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor: Tensor) -> Tensor:
    """
    Generate sine embedding for a position tensor.
    
    Args:
        pos_tensor: Tensor of shape [num_queries, batch_size, 2] or [num_queries, 2].
    
    Returns:
        A tensor of positional embeddings.
    """
    if pos_tensor.dim() == 2:
        pos_tensor = pos_tensor.unsqueeze(1)  # Now shape: [num_queries, 1, 2]

    scale = 2 * math.pi
    # Here we assume an embedding dimension of 128 for sine embedding.
    dim_embed = 128  
    dim_t = torch.arange(dim_embed, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim_embed)

    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Invert the sigmoid function.
    
    Args:
        x: Input tensor.
        eps: Small value to avoid division by zero.
    
    Returns:
        Inverse sigmoid output.
    """
    return torch.log(x / (1 - x + eps) + eps)


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return add_with_pos(tensor, pos)

    def forward_post(self, src: Tensor, src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        q = k = self.with_pos_embed(src, pos)
        attn_output, attn_weights = self.self_attn(q, k, value=src,
                                                    attn_mask=src_mask,
                                                    key_padding_mask=src_key_padding_mask)
        self.attention_weights = attn_weights.detach()
        src2 = attn_output
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src: Tensor, src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        attn_output, attn_weights = self.self_attn(q, k, value=src2,
                                                    attn_mask=src_mask,
                                                    key_padding_mask=src_key_padding_mask)
        self.attention_weights = attn_weights.detach()
        src = src + self.dropout1(attn_output)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None, debug: bool = False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.debug = debug

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        output = src
        if self.debug:
            logger.debug(f"TransformerEncoder forward start - src shape: {src.shape}")
        for i, layer in enumerate(self.layers):
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.debug:
                logger.debug(f"Encoder layer {i} output shape: {output.shape}")
        if self.norm is not None:
            output = self.norm(output)
            if self.debug:
                logger.debug(f"Normalized encoder output shape: {output.shape}")
        return output


# ------------------------------
# New: Dual-Stream Transformer Decoder Layer
# ------------------------------
class DualStreamTransformerDecoderLayer(nn.Module):
    """
    A Transformer decoder layer with dual-stream cross-attention.
    It splits the query tensor into semantic and spatial parts and applies
    separate cross-attention mechanisms.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        # Standard self-attention.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Dual cross-attention modules.
        self.semantic_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.spatial_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Feed-forward network.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Normalization layers.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout layers.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return add_with_pos(tensor, pos)

    def dual_cross_attention(self, tgt: Tensor, memory: Tensor, pos: Optional[Tensor],
                             query_embed: Optional[Tensor], num_semantic: int,
                             memory_mask: Optional[Tensor] = None,
                             memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Splits queries into semantic and spatial parts and applies separate cross-attention.
        """
        # Incorporate query embedding if provided.
        q = tgt + (query_embed if query_embed is not None else 0)
        # Split queries: first `num_semantic` are semantic; the rest are spatial.
        q_sem = q[:num_semantic]      # shape: [num_semantic, batch, d_model]
        q_spat = q[num_semantic:]     # shape: [num_spatial, batch, d_model]
        # Use memory with added positional embeddings.
        k = add_with_pos(memory, pos)
        # Semantic cross-attention.
        out_sem, attn_sem = self.semantic_cross_attn(query=q_sem, key=k, value=memory,
                                                       attn_mask=memory_mask,
                                                       key_padding_mask=memory_key_padding_mask)
        # Spatial cross-attention.
        out_spat, attn_spat = self.spatial_cross_attn(query=q_spat, key=k, value=memory,
                                                       attn_mask=memory_mask,
                                                       key_padding_mask=memory_key_padding_mask)
        self.semantic_attention_weights = attn_sem.detach()
        self.spatial_attention_weights = attn_spat.detach()
        # Concatenate outputs.
        return torch.cat([out_sem, out_spat], dim=0)

    def forward_post(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                     query_embed: Optional[Tensor] = None, query_sine_embed: Optional[Tensor] = None,
                     num_semantic: int = None) -> Tensor:
        # Self-attention.
        q = tgt + (query_embed if query_embed is not None else 0)
        attn_output, attn_weights_self = self.self_attn(query=q, key=q, value=tgt,
                                                         attn_mask=tgt_mask,
                                                         key_padding_mask=tgt_key_padding_mask)
        self.self_attention_weights = attn_weights_self.detach()
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)
        # Dual cross-attention.
        if num_semantic is None:
            num_semantic = tgt.shape[0] // 2
        tgt2 = self.dual_cross_attention(tgt, memory, pos, query_embed, num_semantic,
                                         memory_mask=memory_mask,
                                         memory_key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # Feed-forward.
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                    query_embed: Optional[Tensor] = None, query_sine_embed: Optional[Tensor] = None,
                    num_semantic: int = None) -> Tensor:
        tgt2 = self.norm1(tgt)
        q = tgt2 + (query_embed if query_embed is not None else 0)
        attn_output, attn_weights_self = self.self_attn(query=q, key=q, value=tgt2,
                                                         attn_mask=tgt_mask,
                                                         key_padding_mask=tgt_key_padding_mask)
        self.self_attention_weights = attn_weights_self.detach()
        tgt = tgt + self.dropout1(attn_output)
        tgt2 = self.norm2(tgt)
        tgt2 = self.dual_cross_attention(tgt2, memory, pos, query_embed, num_semantic,
                                         memory_mask=memory_mask,
                                         memory_key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                query_embed: Optional[Tensor] = None, query_sine_embed: Optional[Tensor] = None,
                num_semantic: int = None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    pos, query_embed, query_sine_embed, num_semantic)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_embed, query_sine_embed, num_semantic)


# ------------------------------
# Transformer Decoder and Transformer
# ------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None,
                 return_intermediate: bool = False, d_model: int = 256, debug: bool = False,
                 num_semantic_queries: int = None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.debug = debug
        # The number of semantic queries (to be passed to each layer).
        self.num_semantic_queries = num_semantic_queries

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, query_embed: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask,
                           tgt_key_padding_mask, memory_key_padding_mask,
                           pos, query_embed, None, num_semantic=self.num_semantic_queries)
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm is not None else output)
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output
        if self.return_intermediate:
            hs = torch.stack(intermediate).transpose(1, 2)
            return hs, None  # Return reference points as needed
        return output.unsqueeze(0), None


class Transformer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_queries: int = 300,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu",
                 normalize_before: bool = False, return_intermediate_dec: bool = False, debug: bool = False,
                 num_semantic_queries: int = None):
        super().__init__()

        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, debug=debug)

        # Decoder: Use the dual-stream decoder layer.
        dual_decoder_layer = DualStreamTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(dual_decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, d_model=d_model, debug=debug,
                                          num_semantic_queries=num_semantic_queries)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.debug = debug
        if self.debug:
            logger.debug(f"Transformer initialized with d_model={d_model}, nhead={nhead}, "
                         f"{num_encoder_layers} encoder layers and {num_decoder_layers} decoder layers.")

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, query_embed: Tensor, pos_embed: Tensor) -> Tuple[Tensor, Tensor]:
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)        # [hw, bs, d_model]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [hw, bs, d_model]
        mask = mask.flatten(1)                       # [bs, hw]

        if self.debug:
            logger.debug(f"Transformer forward: src shape: {src.shape}, pos_embed shape: {pos_embed.shape}, mask shape: {mask.shape}")

        # Ensure query_embed is not empty.
        if isinstance(query_embed, list):
            query_embed = [e for e in query_embed if e.numel() > 0]
            if query_embed:
                query_embed = torch.stack(query_embed, dim=0)
            else:
                query_embed = torch.zeros(1, self.d_model, device=src.device)
        if query_embed.dim() == 3 and query_embed.size(0) < query_embed.size(1):
            logger.debug(f"Transposing query_embed from shape {query_embed.shape}")
            query_embed = query_embed.transpose(0, 1)
        tgt = torch.zeros_like(query_embed)
        if tgt.size(0) < tgt.size(1):
            logger.debug(f"Transposing target tensor from shape {tgt.shape}")
            tgt = tgt.transpose(0, 1)

        memory = self.encoder(src, mask=None, src_key_padding_mask=mask, pos=pos_embed)
        if self.debug:
            logger.debug(f"Encoder memory shape: {memory.shape}")

        hs, reference = self.decoder(
            tgt, memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_embed=query_embed,
        )
        if self.debug:
            logger.debug(f"Decoder output hs shape: {hs.shape}, reference shape: {reference.shape}")
        return hs, reference


def build_transformer(args) -> Transformer:
    # Note: Replace 'YOUR_NUM_SEMANTIC_QUERIES' with the actual number of semantic queries in your unified query set.
    return Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_queries=args.num_queries,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",  
        normalize_before=args.pre_norm,
        return_intermediate_dec=args.return_intermediate_dec,
        debug=args.debug,
        num_semantic_queries=100  
    )

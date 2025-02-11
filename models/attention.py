import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
import math

class MultiheadAttention(nn.Module):
    """
    MultiheadAttention module supporting both appearance and positional queries.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)

        # Final linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Optional zero-attention
        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        query: Tensor,                 # Expected [tgt_len, bsz, embed_dim] or [bsz, tgt_len, embed_dim]
        key: Tensor,                   # [src_len, bsz, kdim]
        value: Tensor,                 # [src_len, bsz, vdim]
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        appearance_queries: Optional[Tensor] = None,  # [app_q, bsz, embed_dim]
        positional_queries: Optional[Tensor] = None,  # [pos_q, bsz, embed_dim]
    ):
        """
        Args:
            query, key, value: 
                - query: [tgt_len, bsz, embed_dim] (or [bsz, tgt_len, embed_dim])
                - key:   [src_len, bsz, kdim]
                - value: [src_len, bsz, vdim]
            key_padding_mask: [bsz, src_len]
            need_weights: return attention weights
            attn_mask: optional mask of shape [tgt_len, src_len] or broadcastable
            appearance_queries, positional_queries: additional queries, each [some_q, bsz, embed_dim]

        Returns:
            attn_output:      [tgt_len, bsz, embed_dim]
            attn_output_weights: [bsz, tgt_len, src_len] if need_weights=True
        """
        # 1. Check if query is [bsz, tgt_len, embed_dim] instead of [tgt_len, bsz, embed_dim].
        #    If so, transpose it to match [tgt_len, bsz, embed_dim].
        if query.dim() == 3 and query.shape[0] != query.shape[1]:
            # For example: query is [bsz, tgt_len, embed_dim].
            # We'll assume query.shape[0] is the batch size, so we swap dim 0 and 1.
            if self.training:
                print(f"[DEBUG] Transposing query from [bsz, tgt_len, d_model] to [tgt_len, bsz, d_model]: {query.shape}")
            query = query.transpose(0, 1)  # Now [tgt_len, bsz, embed_dim]

        # 2. Combine additional queries along sequence length (dim=0).
        if appearance_queries is not None and positional_queries is not None:
            # Also check if they might be [bsz, app_q, embed_dim], then transpose them as well
            if appearance_queries.dim() == 3 and appearance_queries.shape[0] != appearance_queries.shape[1]:
                appearance_queries = appearance_queries.transpose(0, 1)
            if positional_queries.dim() == 3 and positional_queries.shape[0] != positional_queries.shape[1]:
                positional_queries = positional_queries.transpose(0, 1)

            query = torch.cat([query, appearance_queries, positional_queries], dim=0)
        elif appearance_queries is not None:
            if appearance_queries.dim() == 3 and appearance_queries.shape[0] != appearance_queries.shape[1]:
                appearance_queries = appearance_queries.transpose(0, 1)
            query = torch.cat([query, appearance_queries], dim=0)
        elif positional_queries is not None:
            if positional_queries.dim() == 3 and positional_queries.shape[0] != positional_queries.shape[1]:
                positional_queries = positional_queries.transpose(0, 1)
            query = torch.cat([query, positional_queries], dim=0)

        # 3. Now query should be [tgt_len, bsz, embed_dim], key/value are [src_len, bsz, embed_dim].
        tgt_len, bsz, embed_dim = query.size()
        src_len, key_bsz, _ = key.size()

        # 4. Ensure batch sizes match
        assert bsz == key_bsz, f"Batch size mismatch: query batch size ({bsz}) != key batch size ({key_bsz})"

        # 5. Linear projections
        q = self.q_proj(query)          # [tgt_len, bsz, embed_dim]
        k = self.k_proj(key)            # [src_len, bsz, embed_dim]
        v = self.v_proj(value)          # [src_len, bsz, embed_dim]

        if self.training:
            print(f"[DEBUG] Query shape after projection: {q.shape}")  # [tgt_len, bsz, embed_dim]
            print(f"[DEBUG] Key shape after projection: {k.shape}")    # [src_len, bsz, embed_dim]
            print(f"[DEBUG] Value shape after projection: {v.shape}")  # [src_len, bsz, embed_dim]

        # 6. Reshape for multi-head attention
        #    [tgt_len, bsz, embed_dim] -> [tgt_len, bsz*num_heads, head_dim]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)

        # 7. Compute attention weights: [tgt_len, bsz*num_heads, src_len]
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)

        # 8. (Optional) Apply attn_mask
        if attn_mask is not None:
            # We want attn_mask shaped like [tgt_len, bsz*num_heads, src_len]
            if attn_mask.dim() == 2:
                # Expand to [tgt_len, bsz*num_heads, src_len]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_mask = attn_mask.expand(tgt_len, bsz * self.num_heads, -1, -1)
                attn_mask = attn_mask.reshape(tgt_len, bsz * self.num_heads, src_len)
            attn_weights += attn_mask

        # 9. (Optional) Key padding mask
        if key_padding_mask is not None:
            # [bsz, src_len] -> [bsz*num_heads, src_len]
            key_padding_mask = key_padding_mask.repeat_interleave(self.num_heads, dim=0)
            # Expand to [tgt_len, bsz*num_heads, src_len]
            key_padding_mask = key_padding_mask.unsqueeze(0).expand(tgt_len, -1, -1)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))

        # 10. Softmax + dropout on attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 11. Weighted sum to get final outputs
        attn_output = torch.bmm(attn_weights, v)  # [tgt_len, bsz*num_heads, head_dim]

        # 12. Reshape back: [tgt_len, bsz, embed_dim]
        attn_output = attn_output.view(tgt_len, bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # 13. Return attention weights if needed
        if need_weights:
            # [tgt_len, bsz*num_heads, src_len] -> [tgt_len, num_heads, bsz, src_len]
            attn_weights = attn_weights.view(tgt_len, self.num_heads, bsz, src_len)
            # permute to [bsz, num_heads, tgt_len, src_len]
            attn_weights = attn_weights.permute(2, 1, 0, 3)
            # average heads -> [bsz, tgt_len, src_len]
            attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None


def build_attention(embed_dim, num_heads, dropout=0.1, bias=True):
    return MultiheadAttention(embed_dim, num_heads, dropout, bias)
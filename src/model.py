# src/model.py
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BERT4Rec(nn.Module):
    """
    Bidirectional Transformer for sequential recommendation (BERT4Rec-style).
    Trains with masked-item modeling over padded sequences of item ids.
    """


    def __init__(
        self,
        num_items: int,
        pad_id: int,
        mask_id: int,
        max_len: int = 50,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            num_items: Total number of unique items.
            pad_id: pad_id.
            mask_id: mask_id.
            max_len: Maximum length of user interaction sequences.
            embed_dim: Dimension of embeddings.
            num_heads: Number of attention heads in Transformer.
            num_layers: Number of Transformer encoder layers.
            dropout: Dropout rate.
        """
        
        super().__init__()

        # Special tokens
        self.num_items = num_items
        self.pad_id = pad_id
        self.mask_token_id = mask_id
        self.vocab_size = num_items + 2  # PAD + items + MASK

        # Embeddings
        self.item_embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self.pad_id)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer (predict item IDs, excluding mask token)
        self.output_layer = nn.Linear(embed_dim, self.num_items + 1, bias=False) 

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_layer.weight)


    @torch.no_grad()
    def mask_inputs(
        self, 
        input_ids: Tensor, 
        mlm_prob: float = 0.15
    ) -> Tuple[Tensor, Tensor]:
        """
        BERT-style masking.

        Args:
            input_ids: (batch_size, seq_len) with PAD=0, MASK=num_items+1.
            mlm_prob: probability of a given token being masked.
        Returns:
            input_ids_masked: same shape, with a subset replaced by [MASK]/random/original.
            labels: original ids at masked positions, else -100 (ignored by CE).
        """
        
        device = input_ids.device
        prob = torch.full_like(input_ids, mlm_prob, dtype=torch.float, device=device)
        maskable = input_ids.ne(self.pad_id) & input_ids.ne(self.mask_token_id)
        mask = torch.bernoulli(prob).bool() & maskable

        labels = input_ids.clone()
        labels[~mask] = -100  # ignore

        # 80% -> [MASK]
        replace_mask = torch.bernoulli(torch.full_like(input_ids, 0.8, dtype=torch.float)).bool() & mask
        # 10% -> random item in [1..num_items]
        random_mask = torch.bernoulli(torch.full_like(input_ids, 0.5, dtype=torch.float)).bool() & mask & ~replace_mask
        # remaining 10% keep original

        input_ids_masked = input_ids.clone()
        input_ids_masked[replace_mask] = self.mask_token_id

        rand_items = torch.randint(1, self.num_items + 1, input_ids.shape, device=device)
        input_ids_masked[random_mask] = rand_items[random_mask]

        return input_ids_masked, labels


    def forward(
        self,
        input_ids: Tensor, 
        labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Perform a forward pass of the BERT4Rec model.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with PAD=0, MASK=num_items+1
            labels: Tensor of shape (batch_size, seq_len) with valid targets in [0..num_items], PAD positions can be 0,
                and non-masked positions must be -100 (ignored). If None, loss=None.
        Returns:
            logits: Tensor of shape (batch_size, seq_len, num_items+1)
            loss:   Cross-entropy loss
        """

        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.position_embedding.num_embeddings, "sequence longer than max_len (position embedding size)"

        # Create position IDs
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # Embed items and positions
        x = self.item_embedding(input_ids) + self.position_embedding(pos)
        x = self.dropout(x)

        # Padding mask: True->pad, False->keep
        key_padding_mask = input_ids.eq(self.pad_id)  # (batch_size, seq_len)
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        logits = self.output_layer(x)  # (batch_size, seq_len, num_items+1)

        if labels is None:
            return logits, None

        # Cross-entropy over masked positions only (ignore_index=-100)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

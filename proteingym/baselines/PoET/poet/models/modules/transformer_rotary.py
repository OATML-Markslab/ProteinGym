from poet.models.modules import FlashMultiheadAttention, RotaryEmbedding
from poet.models.modules.transformer import (
    TieredTransformerDecoderLayer,
    TieredTransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


class RotaryFlashMultiheadAttention(FlashMultiheadAttention):
    def __init__(
        self, *args, rotary_scale=None, rotary_force_fp32=None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rotary_emb = RotaryEmbedding(
            dim_model=self.head_dim,
            scale=rotary_scale,
            force_fp32=rotary_force_fp32,
        )

    def _transform_qkv(
        self,
        query,
        key,
        value,
        query_positions=None,
        key_positions=None,
        transform_query=True,
        transform_key=True,
        transform_value=False,
    ):
        query, key = self.rotary_emb(
            query,
            key,
            q_positions=query_positions,
            k_positions=key_positions,
            transform_q=transform_query,
            transform_k=transform_key,
        )
        return query, key, value


class RotaryTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, rotary_scale=None, rotary_force_fp32=None, **kwargs):
        self.rotary_scale = rotary_scale
        self.rotary_force_fp32 = rotary_force_fp32
        super().__init__(*args, **kwargs)

    def _init_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=False,
    ):
        return RotaryFlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )


class TieredRotaryTransformerEncoderLayer(TieredTransformerEncoderLayer):
    def __init__(
        self,
        *args,
        rotary_scale=None,
        rotary_force_fp32=None,
        use_multi_rotary=True,
        **kwargs,
    ):
        self.rotary_scale = rotary_scale
        self.rotary_force_fp32 = rotary_force_fp32
        self.use_multi_rotary = use_multi_rotary
        super().__init__(*args, **kwargs)

    def _init_self_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=False,
    ):
        """
        Initialize the multi-head attention module used for each sequence independently.
        """
        return RotaryFlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )

    def _init_multi_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=False,
    ):
        """
        Initialize the multi-head attention module used for each sequence-of-sequences.
        """
        Module = FlashMultiheadAttention
        if self.use_multi_rotary:
            Module = RotaryFlashMultiheadAttention
        return Module(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )


class RotaryTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        *args,
        rotary_scale=None,
        rotary_force_fp32=None,
        use_cross_rotary=True,
        **kwargs,
    ):
        self.rotary_scale = rotary_scale
        self.rotary_force_fp32 = rotary_force_fp32
        self.use_cross_rotary = use_cross_rotary
        super().__init__(*args, **kwargs)

    def _init_self_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=True,
    ):
        return RotaryFlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )

    def _init_cross_mha_module(
        self, d_model, nhead, dropout=0, use_qkv_bias=False, batch_first=True
    ):
        Module = FlashMultiheadAttention
        if self.use_cross_rotary:
            Module = RotaryFlashMultiheadAttention
        return Module(
            d_model,
            nhead,
            self_attention=False,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=False,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )


class TieredRotaryTransformerDecoderLayer(TieredTransformerDecoderLayer):
    def __init__(
        self,
        *args,
        rotary_scale=None,
        rotary_force_fp32=None,
        use_multi_rotary=True,
        use_cross_rotary=True,
        **kwargs,
    ):
        self.rotary_scale = rotary_scale
        self.rotary_force_fp32 = rotary_force_fp32
        self.use_multi_rotary = use_multi_rotary
        self.use_cross_rotary = use_cross_rotary
        super().__init__(*args, **kwargs)

    def _init_self_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=True,
    ):
        """
        Initialize the multi-head attention module used for each sequence independently.
        """
        return RotaryFlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )

    def _init_multi_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=True,
    ):
        """
        Initialize the multi-head attention module used for each sequence-of-sequences.
        """
        Module = FlashMultiheadAttention
        if self.use_multi_rotary:
            Module = RotaryFlashMultiheadAttention
        return Module(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )

    def _init_cross_mha_module(
        self, d_model, nhead, dropout=0, use_qkv_bias=False, batch_first=True
    ):
        Module = FlashMultiheadAttention
        if self.use_cross_rotary:
            Module = RotaryFlashMultiheadAttention
        return Module(
            d_model,
            nhead,
            self_attention=False,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=False,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class RITAConfig(PretrainedConfig):
    model_type = "rita"

    def __init__(
        self,
        vocab_size=26,
        d_model=768,
        num_layers=12,
        max_seq_len=1024,
        num_heads=12,
        dropout=0.,
        ff_ratio=4,
        eos_token_id=2,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_feedforward = d_model*ff_ratio
        self.num_layers = num_layers
        self.max_seq_len=max_seq_len
        self.dropout = dropout
        self.eos_token_id=eos_token_id
        self.initializer_range=0.02

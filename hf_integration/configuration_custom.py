from transformers import PretrainedConfig

class CustomTransformerConfig(PretrainedConfig):
    model_type = "custom_transformer"

    def __init__(
        self,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        src_vocab_size=30000,
        tgt_vocab_size=30000,
        max_seq_length=350,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_length = max_seq_length

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


config = CustomTransformerConfig()
config.save_pretrained("config")
from transformers import PreTrainedModel
from .configuration_custom import CustomTransformerConfig
from src.model.model import build_transformer
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput

class CustomTransformerModel(PreTrainedModel):
    config_class = CustomTransformerConfig

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # Build your transformer using config
        self.model = build_transformer(
            src_vocab_size=config.src_vocab_size,
            tgt_vocab_size=config.tgt_vocab_size,
            src_seq_len=config.max_seq_length,
            tgt_seq_len=config.max_seq_length,
            d_model=config.d_model,
            N=config.n_layers,
            h=config.n_heads,
            dropout=config.dropout,
            d_ff=config.d_ff
        )

        # IMPORTANT
        self.post_init()
    


    def forward(
        self,
        src_input_ids,
        tgt_input_ids,
        src_attention_mask=None,
        tgt_attention_mask=None,
        labels=None,
    ):
        logits = self.model(
            src_input_ids,
            src_attention_mask,
            tgt_input_ids,
            tgt_attention_mask
        )

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits
        )
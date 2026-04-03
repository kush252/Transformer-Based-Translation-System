from hf_integration.model_packing import CustomTransformerModel
from hf_integration.configuration_custom import CustomTransformerConfig
import torch

def causal_mask(size):
    """Create causal mask for decoder - same as in dataset.py"""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

config = CustomTransformerConfig()
model = CustomTransformerModel(config)

# Create test data with proper 4D masks like training does
batch_size = 2
seq_len = 10
pad_token_id = config.pad_token_id

src_input_ids = torch.randint(0, 100, (batch_size, seq_len))
tgt_input_ids = torch.randint(0, 100, (batch_size, seq_len))

# Create 4D masks (batch, 1, seq_len, seq_len) like BilingualDataset does
src_attention_mask = torch.ones(batch_size, 1, 1, seq_len).int()
tgt_attention_mask = (torch.ones(batch_size, 1, seq_len).int() & causal_mask(seq_len)).unsqueeze(1)

output = model(
    src_input_ids=src_input_ids,
    tgt_input_ids=tgt_input_ids,
    src_attention_mask=src_attention_mask,
    tgt_attention_mask=tgt_attention_mask
)

print(model)
print(output)
from hf_integration.model_packing import CustomTransformerModel
from hf_integration.configuration_custom import CustomTransformerConfig
import torch

def causal_mask(size):
    """Create causal mask for decoder - same as in dataset.py"""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

config = CustomTransformerConfig().from_pretrained("hf_integration/config.json")
model = CustomTransformerModel(config)
checkpoint = torch.load('models/tmodel_04.pt', map_location=torch.device("cpu"))



# Rename checkpoint keys to add the 'model.' prefix
checkpoint_state = checkpoint['model_state_dict']
fixed_state = {'model.' + k: v for k, v in checkpoint_state.items()}

checkpoint_keys = set(fixed_state.keys())
model_keys = set(model.state_dict().keys())

missing_in_checkpoint = model_keys - checkpoint_keys
extra_in_checkpoint = checkpoint_keys - model_keys

print(f"in checkpoint (after prefix): {len(checkpoint_keys)}")
print(f"in model: {len(model_keys)}")

print("\nMissing in checkpoint:")
for key in list(missing_in_checkpoint)[:10]:  
    print(f"  {key}")
print(f"  ... ({len(missing_in_checkpoint)} total)")

print("\nExtra in checkpoint:")
for key in list(extra_in_checkpoint)[:10]:  
    print(f"  {key}")
print(f"  ... ({len(extra_in_checkpoint)} total)")

model.load_state_dict(fixed_state)

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

model.save_pretrained("test_hf_model")
config.save_pretrained("test_hf_model")
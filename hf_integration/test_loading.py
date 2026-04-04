import torch    
from transformers import AutoModel



save_path = "test_hf_model"
auto_model = AutoModel.from_pretrained(
    save_path,
    trust_remote_code=True
)

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).int()
    return mask == 0

batch_size = 2
seq_len = 10

src_input_ids = torch.randint(0, 100, (batch_size, seq_len))
tgt_input_ids = torch.randint(0, 100, (batch_size, seq_len))

src_attention_mask = torch.ones(batch_size, 1, 1, seq_len).int()
tgt_attention_mask = (
    torch.ones(batch_size, 1, seq_len).int() & causal_mask(seq_len)
).unsqueeze(1)

output = auto_model(
    src_input_ids=src_input_ids,
    tgt_input_ids=tgt_input_ids,
    src_attention_mask=src_attention_mask,
    tgt_attention_mask=tgt_attention_mask,
)

print("\nAutoModel test successful")
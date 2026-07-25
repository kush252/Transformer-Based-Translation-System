import torch
import warnings
from tqdm import tqdm
from pathlib import Path
from src.utils.config import get_config, get_weights_file_path
from src.pipelines.train import get_ds, get_model, greedy_decode
from hf_integration.configuration_custom import CustomTransformerConfig

# Import metrics
try:
    from torchmetrics.text.bleu import BLEUScore
    from torchmetrics.text.cer import CharErrorRate
    from torchmetrics.text.wer import WordErrorRate
except ImportError:
    print("Please install torchmetrics to calculate metrics: pip install torchmetrics")
    import sys
    sys.exit(1)

def evaluate_model(config, model_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # We only need the validation dataloader and tokenizers
    _, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, model_config.max_seq_length)
    
    # Sync actual special token IDs from tokenizers to config (same as train.py)
    model_config.pad_token_id = tokenizer_src.token_to_id("[PAD]")
    model_config.bos_token_id = tokenizer_tgt.token_to_id("[SOS]")
    model_config.eos_token_id = tokenizer_tgt.token_to_id("[EOS]")
    model_config.src_vocab_size = tokenizer_src.get_vocab_size()
    model_config.tgt_vocab_size = tokenizer_tgt.get_vocab_size()

    model = get_model(model_config).to(device)
    
    # Extract base model if wrapped in DataParallel
    base_model = model.module if hasattr(model, 'module') else model

    # Load latest checkpoint
    model_filename = get_weights_file_path(config, config['preload'])
    if model_filename and Path(model_filename).exists():
        print(f"Loading model from file: {model_filename}")
        state = torch.load(model_filename, map_location=device, weights_only=False)
        base_model.load_state_dict(state['model_state_dict'])
    else:
        print("Warning: No model checkpoint found. Evaluating with random weights.")

    model.eval()

    # Initialize metrics
    bleu = BLEUScore()
    cer = CharErrorRate()
    wer = WordErrorRate()

    expected_translations = []
    predicted_translations = []

    console_width = 80
    
    # Set to len(val_dataloader) for full evaluation
    num_examples = len(val_dataloader)
    
    with torch.no_grad():
        batch_iterator = tqdm(val_dataloader, desc="Evaluating", total=num_examples)
        for count, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            # Predict
            model_output = greedy_decode(base_model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, model_config.max_seq_length, device, model_config)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            
            expected_translations.append(target_text)
            predicted_translations.append(model_out_text)
            
            # Print first 2 examples to console
            if count < 2:
                print(f"\n{'='*console_width}")
                print(f"Source: {source_text}")
                print(f"Target: {target_text}")
                print(f"Predicted: {model_out_text}")
                
            if count >= num_examples - 1:
                break
                
    # Calculate metrics
    print(f"\n{'='*console_width}")
    print("Calculating metrics... This might take a few seconds.")
    
    # BLEUScore expects target texts to be in a list of lists of strings
    bleu_targets = [[target] for target in expected_translations]
    
    b_score = bleu(predicted_translations, bleu_targets)
    c_score = cer(predicted_translations, expected_translations)
    w_score = wer(predicted_translations, expected_translations)
    
    print(f"\n{'-'*30}")
    print(f"EVALUATION RESULTS:")
    print(f"{'-'*30}")
    print(f"BLEU Score: {b_score.item():.4f} (Higher is better, max 1.0)")
    print(f"CER:        {c_score.item():.4f} (Lower is better, min 0.0)")
    print(f"WER:        {w_score.item():.4f} (Lower is better, min 0.0)")
    print(f"{'-'*30}")
    print(f"Evaluated on {len(predicted_translations)} sentences.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    model_config = CustomTransformerConfig.from_pretrained("hf_integration/config.json")
    evaluate_model(config, model_config)

import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer
from tokenizers.models import WordLevel 
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
import json

from datasets import load_dataset
from transformers import AutoConfig
from src.model.model import build_transformer
from src.utils.dataset import BilingualDataset, causal_mask
from src.utils.config import get_config,get_weights_file_path

from tqdm import tqdm
import warnings

def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device,model_config):
    sos_idx = model_config.bos_token_id
    eos_idx = model_config.eos_token_id

    encoder_output = model.encode(source,source_mask)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1)==max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output,source_mask,decoder_input,decoder_mask)
        prob = model.project(out[:,-1,:])
        _,next_token = torch.max(prob,dim=1)
        decoder_input = torch.cat([decoder_input,torch.empty(1,1).type_as(source).fill_(next_token.item()).to(device)],dim=1)
        if next_token.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_state,writer,model_config,num_examples=2):
    model.eval()
    count=0

    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) ==1,"Batch size must be 1 for validation"
            model_output = greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device,model_config)

            source_text =  batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            print_msg(f"\n{'='*console_width}")
            print_msg(f"Source: {source_text}")
            print_msg(f"Target: {target_text}")
            print_msg(f"Model Output: {model_out_text}")

            if count==num_examples:
                break


def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]", "[SOS]", "[EOS]", "[MASK]"],min_frequency=2)

        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))


    return tokenizer

def get_ds(config, max_seq_length):
    ds_raw = load_dataset('opus_books',f"{config['lang_src']}-{config['lang_tgt']}",split='train')

    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    train_ds_size = int(len(ds_raw)*0.9)
    test_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,test_ds_size])

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],max_seq_length)
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],max_seq_length)

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))

    print(f"Max len src: {max_len_src}")
    print(f"Max len tgt: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def count_parameters(model):
    """Count total trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total

def get_model(config):
    """Build transformer using config as single source of truth (HuggingFace standard).
    
    Args:
        config: PretrainedConfig object with all model architecture parameters
    """
    model = build_transformer(
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
    return model


def train_model(config, model_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt = get_ds(config, model_config['max_seq_length'])
    
    # Sync actual special token IDs from tokenizers to config (for reproducibility)
    model_config.pad_token_id = tokenizer_src.token_to_id("[PAD]")
    model_config.bos_token_id = tokenizer_tgt.token_to_id("[SOS]")
    model_config.eos_token_id = tokenizer_tgt.token_to_id("[EOS]")
    
    # Update config with actual vocabulary sizes from tokenizers
    model_config.src_vocab_size = tokenizer_src.get_vocab_size()
    model_config.tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    model_config.save_pretrained("config/")
    print(f"Updated config with actual vocab sizes - src: {model_config.src_vocab_size}, tgt: {model_config.tgt_vocab_size}")
    print(f"Special tokens synced - PAD: {model_config.pad_token_id}, BOS: {model_config.bos_token_id}, EOS: {model_config.eos_token_id}")
    
    model = get_model(model_config).to(device)

    # Display model and vocabulary info
    model_params = count_parameters(model)
    src_vocab_size = model_config.src_vocab_size
    tgt_vocab_size = model_config.tgt_vocab_size
    train_size = len(train_dataloader)
    total_tokens_per_epoch = train_size * config['batch_size'] * model_config['max_seq_length']

    print(f"\n{'='*50}")
    print(f"Model Configuration:")
    print(f"{'='*50}")
    print(f"Total parameters: {model_params:,}")
    print(f"Model dimension (d_model): {model_config['d_model']}")
    print(f"Number of layers: {model_config['n_layers']}")
    print(f"Number of heads: {model_config['n_heads']}")
    print(f"Source vocab size: {src_vocab_size:,}")
    print(f"Target vocab size: {tgt_vocab_size:,}")
    print(f"Max sequence length: {model_config['max_seq_length']}")
    print(f"Training batches: {train_size:,}")
    print(f"Approx tokens per epoch: {total_tokens_per_epoch:,}")
    print(f"Total epochs: {config['num_epochs']}")
    print(f"{'='*50}\n")

    # Save model configuration metadata at the start of training
    metadata = {
        "config": config,
        "model_config": model_config,
        "model_parameters": int(model_params),
        "src_vocab_size": int(src_vocab_size),
        "tgt_vocab_size": int(tgt_vocab_size),
        "max_seq_len": int(model_config['max_seq_length']),
        "batch_size": int(config['batch_size']),
        "num_epochs": int(config['num_epochs']),
        "tokens_per_epoch": int(total_tokens_per_epoch),
        "device": str(device),
    }

    metadata_path = Path(config['model_folder']) / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config,config['preload'])
        print(f"Loading model from file: {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.pad_token_id,label_smoothing=0.1).to(device)

    last_loss_value = None

    for epoch in range(initial_epoch,config['num_epochs']):
        batch_iterator = tqdm(train_dataloader,desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input,encoder_mask)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1,model_config.tgt_vocab_size),label.view(-1))    
            batch_iterator.set_postfix({f"loss":f"{loss.item():6.3f}"})

            last_loss_value = loss.item()

            writer.add_scalar("train loss",loss.item(),global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,model_config['max_seq_length'],device,lambda msg: batch_iterator.write(msg),global_step,writer,model_config)
        
        model_filename = get_weights_file_path(config,f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)  

    # Update metadata with the final training loss after all epochs
    metadata["final_training_loss"] = float(last_loss_value) if last_loss_value is not None else None
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    model_config = AutoConfig.from_pretrained("hf_integration/config.json")
    train_model(config, model_config)

            
    
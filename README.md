# Transformer-Based Translation System

This project is a complete, from-scratch implementation of a Transformer model for Neural Machine Translation using PyTorch. The model is built according to the architecture proposed in the seminal paper *"Attention is All You Need"* (Vaswani et al., 2017). 

Currently, the model is configured to train on an English-to-Italian dataset (`opus_books`), translating text across languages.

## 🚀 Features & Concepts Implemented

This project dives deeply into the core workings of modern LLMs and Transformers, implementing all components manually rather than relying on high-level library abstractions for the model architecture.

### 1. Transformer Architecture (From Scratch)
* **Embedding Layers**: Maps vocabulary indices to dense vector representations.
* **Positional Encoding**: Uses sinusoidal functions (sine and cosine) to inject sequence order information into the embeddings.
* **Multi-Head Attention**: Implements scaled dot-product attention. Includes mechanisms for both **Self-Attention** (in encoder/decoder) and **Cross-Attention** (decoder attending to encoder output).
* **Feed-Forward Networks**: Two linear transformations with a ReLU activation in between.
* **Layer Normalization & Residual Connections**: Added around each sub-layer (attention and feed-forward) to stabilize and speed up training.
* **Encoder & Decoder Stacks**: Configurable number of identical blocks ($N=6$ by default).

### 2. Tokenization & Data Processing
* **Custom Tokenizer Training**: Uses Hugging Face's `tokenizers` library to train a **Word-Level** tokenizer completely from scratch on the target corpus.
* **Padding & Special Tokens**: Handles `[SOS]`, `[EOS]`, `[PAD]`, `[UNK]` and padding logic dynamically up to a specified maximum sequence length.
* **Masking**:
  * **Source Masking**: Ignores `[PAD]` tokens in the encoder.
  * **Causal/Subsequent Masking**: Prevents the decoder from "looking ahead" at future tokens during autoregressive training.

### 3. Training Pipeline
* **Dataset Management**: Uses Hugging Face `datasets` for efficient data downloading and streaming (`opus_books`).
* **Loss Function**: Implements `CrossEntropyLoss` with **Label Smoothing** ($0.1$) to prevent overconfidence and improve generalization.
* **Optimizer**: Adam optimizer.
* **Greedy Decoding**: Autoregressive generation used during the validation loop to preview translations in real-time.
* **Experiment Tracking**: Integrated with **TensorBoard** to display training loss metrics.
* **Checkpointing**: Automatically saves model state dictionaries per epoch and allows resuming from the latest checkpoint.

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **Deep Learning Framework**: PyTorch (`torch`, `torch.nn`)
* **Data Pipelines**: Hugging Face `datasets`
* **Tokenization**: Hugging Face `tokenizers`
* **Logging/Monitoring**: TensorBoard
* **Utilities**: `tqdm` (Progress bars)

## 📁 Project Structure

```text
Trans_trans/
├── src/
│   ├── model/
│   │   └── model.py         # The core Transformer components (Multi-Head Attention, Encoder, Decoder, etc.)
│   ├── pipelines/
│   │   └── train.py         # Training loop, data loaders, validation, and TensorBoard logging
│   ├── utils/
│   │   ├── config.py        # Hyperparameter configurations (batch size, lr, d_model, etc.)
│   │   └── dataset.py       # Custom PyTorch Dataset handling padding, tokenization, and causal masking
├── requirements.txt         # Project dependencies
```

## ⚙️ Configuration
The model hyperparameters can be tweaked inside `src/utils/config.py`:
- `batch_size`: 8 
- `num_epochs`: 20
- `d_model`: 512
- `seq_len`: 350
- `lr`: 10^-4

## 📊 Metrics To Track (Before/After First Training)

The current pipeline already logs **training loss** to TensorBoard and prints sample validation translations. Since you have not trained yet, use this table as a starter scoreboard.

| Metric | Why it matters | Value before training | Good early target (small EN→IT run) |
|---|---|---:|---:|
| Train Cross-Entropy Loss | Main optimization signal (`CrossEntropyLoss`) | N/A | $< 4.5$ by mid-training, $2.0$-$3.5$ by end |
| Perplexity ($e^{\text{loss}}$) | More interpretable LM quality from loss | N/A | $< 90$ mid-training, $8$-$35$ near end |
| Validation Loss | Generalization check on held-out split | N/A | Within about $10\%$-$25\%$ of train loss |
| BLEU (sacreBLEU) | N-gram overlap with references | N/A | $8$-$20$ for first clean baseline |
| chrF / chrF++ | Character-level quality, good for morphology | N/A | $30$-$55$ |
| Sample Translation Pass Rate | Quick manual quality sanity check | N/A | $\ge 60\%$ understandable samples |

### What You Can Measure Immediately
- `Train Loss`: already implemented and visible in TensorBoard.
- `Perplexity`: compute from logged loss using $\text{ppl} = e^{\text{loss}}$.
- `Sample quality`: already available from printed `Source/Target/Model Output` in validation.

### Recommended Next Metrics To Add
1. **Validation loss per epoch** in `run_validation(...)`.
2. **BLEU + chrF** on the validation subset (for more objective progress).
3. **Length ratio** (`len(pred)/len(ref)`) to catch under-generation.

If you want, I can also add BLEU/chrF computation directly in the training script so these numbers are logged to TensorBoard every epoch.

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run training:**
   ```bash
   python src/pipelines/train.py
   ```

3. **Monitor with TensorBoard:**
   ```bash
   tensorboard --logdir runs/tmodel
   ```

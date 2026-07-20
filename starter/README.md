# Starter

This repository is a small language-model training and evaluation starter project. The current code is intentionally simple: it trains a byte-level, GPT-style autoregressive model in pure PyTorch and evaluates checkpoints using bits per byte (BPB). The baseline works, but the comments in the code call out that it is deliberately mediocre and intended to be improved within the assignment caps.

## Current state of the code ->

- **Training pipeline:** `starter/train.py` reads the provided training corpus, tokenizes it with the byte tokenizer, creates random next-token prediction batches, trains a GPT model on CPU, and saves a checkpoint containing the model weights, config, step count, and loss curve.
- **Model:** `starter/model.py` defines a compact transformer/GPT model with token and position embeddings, causal self-attention blocks, MLP blocks, final layer norm, and a linear language-modeling head.
- **Tokenizer:** `starter/tokenizer.py` currently uses raw UTF-8 bytes as tokens, so the vocabulary size is 256 and arbitrary UTF-8 text can be encoded/decoded without a learned tokenizer file.
- **Evaluation:** `starter/evaluate.py` loads a saved checkpoint, rebuilds the configured model, verifies tokenizer losslessness, and reports BPB on an evaluation text file using a sliding-window scoring procedure.
- **Data:** `data/train_corpus.txt` and `data/dev_eval.txt` are the provided text files for training and development evaluation. The training corpus is about 7.3 MB, and the development evaluation file is about 159 KB.
- **Repository status:** no trained checkpoint or run notes are currently present in the repository. The assignment text says a final submission folder should include `ckpt.pt`, code, `RUNLOG.md`, `NOTES.md`, and `SUMMARY.html`.

## How to run the baseline

From the `starter` directory:

```bash
python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```

Baseline runtime is expected to be roughly 1.5-3 minutes on a laptop CPU, according to the original starter notes.

## Assignment caps and constraints

The code comments document these hard caps for a valid graded run:

- Maximum **2,000 optimizer steps** for the run that creates the submitted checkpoint.
- Maximum **2,000,000 total model parameters**.
- Training text must be the provided `data/train_corpus.txt` only.
- Use pure PyTorch, NumPy, and Python standard library only; no pretrained models or external pretrained assets.
- `starter/evaluate.py`'s checkpoint/text-file interface must keep working.
- The tokenizer must be lossless: `decode(encode(text)) == text`.

## File-by-file guide

### `starter/README.md`

This file. It explains the project state, how to run training/evaluation, the assignment constraints, and what each file contains.

### `starter/tokenizer.py`

Contains the baseline tokenizer implementation:

- `ByteTokenizer`: encodes strings as raw UTF-8 byte IDs and decodes byte IDs back to text.
- `vocab_size = 256`: one token for each possible byte value.
- `save(path)`: writes a tiny JSON marker describing the byte tokenizer.
- `load(path=None)`: returns a `ByteTokenizer` instance. `train.py` and `evaluate.py` call this with no arguments.

Important behavior: this tokenizer is robust and lossless for arbitrary UTF-8 text, but multi-byte characters such as Devanagari characters consume multiple tokens, reducing effective context length for those scripts.

### `starter/model.py`

Defines the neural language model:

- `Config`: stores model hyperparameters such as vocabulary size, context length, layer count, head count, embedding size, dropout, and whether to tie token/head weights.
- `SelfAttention`: implements causal multi-head self-attention using PyTorch's `scaled_dot_product_attention` with `is_causal=True`.
- `Block`: combines layer normalization, self-attention, residual connections, and a feed-forward MLP.
- `GPT`: assembles embeddings, transformer blocks, final normalization, and output head. Its `forward` method returns logits and optionally cross-entropy loss when targets are supplied.
- `n_params()`: counts model parameters for enforcing the assignment cap.

The default model is a 4-layer, 4-head, 160-dimensional GPT with a 128-token context window.

### `starter/train.py`

Contains the baseline training script:

- Parses command-line arguments for data path, steps, batch size, learning rate, seed, output checkpoint path, and log frequency.
- Enforces the 2,000-step cap.
- Loads and tokenizes the training corpus.
- Builds `GPT(Config())` and enforces the 2,000,000-parameter cap.
- Samples random contiguous token batches with `get_batch`.
- Trains with plain `torch.optim.Adam` and a constant learning rate.
- Logs average loss periodically.
- Saves a checkpoint with:
  - `model`: model state dict,
  - `config`: public config attributes,
  - `steps`: training step count,
  - `train_loss_curve`: list of per-step training losses.

### `starter/evaluate.py`

Contains the scoring script used for development and hidden evaluation:

- `load_model(ckpt_path)`: loads a checkpoint, reconstructs the model config, restores weights, and switches to eval mode.
- `bits_per_byte(model, cfg, tok, text)`: tokenizes text, verifies tokenizer round-trip losslessness, then computes negative log-likelihood over the text and converts it to bits per original UTF-8 byte.
- Uses a sliding window with 50% context carry-over so tokens are scored once while still receiving left context.
- `main()`: prints one JSON object with `bpb`, `n_params`, `steps`, `tokens_in_eval`, and `tokens_scored`.

### `data/train_corpus.txt`

The provided training corpus. It is plain UTF-8 text and is the only allowed training data for the assignment. In the current checkout it has 39,204 lines and 7,318,592 bytes.

### `data/dev_eval.txt`

The provided development evaluation text. It is used to measure BPB before submitting. In the current checkout it has 1,249 lines and 159,225 bytes.

## Expected submission artifacts

Before final submission, the original starter instructions say the submission folder should contain:

- `ckpt.pt` - trained checkpoint produced by `starter/train.py` or an improved compatible trainer.
- Code files, including any changes to model, tokenizer, training, or evaluation support code.
- `RUNLOG.md` - notes for each training run.
- `NOTES.md` - implementation notes and rationale.
- `SUMMARY.html` - assignment summary artifact described in the brief.

# Fornax LM

A GPT-style autoregressive language model built from scratch using PyTorch.
No pre-built transformer classes. No model hubs. Every component — attention,
tokenization, training, and inference — written from first principles.

---

## What it is

Fornax is a decoder-only transformer that you train on your own corpus and
run inference against through a web interface. It implements modern architectural
choices including RoPE positional encoding, SwiGLU activations, RMSNorm, and
KV-cached autoregressive generation.

It is not a fine-tuned checkpoint. It is not a wrapper. It is a full language
model pipeline from raw text to generated output.

---

## Stack

| Layer | Technology |
|---|---|
| Model | PyTorch 2.x |
| Tokenizer | HuggingFace Tokenizers (BPE) |
| Backend | FastAPI + SQLAlchemy + PostgreSQL |
| Migrations | Alembic |
| Frontend | React + Vite + Framer Motion |
| Tracking | Weights and Biases |
| Infrastructure | Docker + Docker Compose |

---

## Architecture

Decoder-only transformer with the following design decisions:

- Rotary Positional Embeddings (RoPE) for relative position awareness
- Multi-head causal self-attention with KV cache at inference
- SwiGLU feed-forward networks
- Pre-LayerNorm with RMSNorm
- BPE tokenizer trained on your corpus
- AdamW with cosine decay and linear warmup
- Top-k and nucleus (top-p) sampling at inference

---

## Getting started

Prerequisites: Docker and Docker Compose installed.

Clone the repository and create your environment file:

```bash
cp .env.example .env
```

Start the database and API:

```bash
docker compose up -d
```

Run database migrations:

```bash
docker compose exec api alembic upgrade head
```

Download a corpus and begin training:

```bash
python scripts/download_corpus.py
python train.py
```

Once a run reaches completed status, open the frontend:

```bash
cd frontend
npm install
npm run dev
```

Navigate to http://localhost:5173 to start generating.

---

## Training

All hyperparameters are configured through the Pydantic models in `config/`.
Key defaults:

| Parameter | Default |
|---|---|
| d_model | 256 |
| n_layers | 6 |
| n_heads | 8 |
| vocab_size | 8000 |
| max_seq_len | 256 |
| batch_size | 32 |
| max_steps | 5000 |
| learning rate | 3e-4 |

To resume a stopped run, set `RESUME_CHECKPOINT` in `.env` to the checkpoint path.

---

## Experiment tracking

Set `WANDB_API_KEY` in `.env` to enable Weights and Biases logging.
If the key is absent, training continues without remote logging.

---

## License

MIT
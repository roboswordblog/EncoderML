import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import tiktoken
from datasets import load_dataset

# ==========================================
# 1. SETUP DEVICE (MAC M2 MPS) & TOKENIZER
# ==========================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

# Using 'gpt2' base keeps vocab at ~50k so it fits comfortably in M2 Unified Memory
base_enc = tiktoken.get_encoding("gpt2")
special_tokens = {
    **base_enc._special_tokens,
    "<|pad|>": base_enc.n_vocab,
    "<|bos|>": base_enc.n_vocab + 1,
    "<|eos|>": base_enc.n_vocab + 2,
}

enc = tiktoken.Encoding(
    name="coding_ai_enc",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens=special_tokens,
)

VOCAB_SIZE = enc.n_vocab
PAD_ID = special_tokens["<|pad|>"]
BOS_ID = special_tokens["<|bos|>"]
EOS_ID = special_tokens["<|eos|>"]
MAX_SEQ_LEN = 256  # Keep compact for fast M2 training

# ==========================================
# 2. STREAMING HUGGING FACE DATASET
# ==========================================
class CodeSearchNetStream(IterableDataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset("code_search_net", "python", split=split, streaming=True)

    def __iter__(self):
        for item in self.dataset:
            docstring = item["func_documentation_string"].strip()
            code = item["func_code_string"].strip()

            # Skip empty or noisy samples
            if len(docstring) < 15 or len(code) < 20:
                continue

            # Encode and truncate
            src_ids = enc.encode(docstring, allowed_special="all")[:MAX_SEQ_LEN]
            code_ids = enc.encode(code, allowed_special="all")[:MAX_SEQ_LEN - 1]

            # Shift target for teacher forcing:
            # Decoder Input:  [<|bos|>, tok1, tok2, ...]
            # Target Labels:  [tok1, tok2, ..., <|eos|>]
            tgt_in = [BOS_ID] + code_ids
            tgt_out = code_ids + [EOS_ID]

            yield (
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_in, dtype=torch.long),
                torch.tensor(tgt_out, dtype=torch.long),
            )

def collate_batch(batch):
    src_list, tgt_in_list, tgt_out_list = zip(*batch)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=PAD_ID)
    tgt_in_padded = pad_sequence(tgt_in_list, batch_first=True, padding_value=PAD_ID)
    tgt_out_padded = pad_sequence(tgt_out_list, batch_first=True, padding_value=PAD_ID)
    return src_padded, tgt_in_padded, tgt_out_padded

# ==========================================
# 3. ENCODER-DECODER MODEL
# ==========================================
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, pad_id):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return self.tok_emb(x) + self.pos_emb(positions)

class CodingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3, dim_ff=1024, max_seq_len=256, pad_id=PAD_ID):
        super().__init__()
        self.embedding = LearnedPositionalEmbedding(vocab_size, d_model, max_seq_len, pad_id)
        # PyTorch's built-in Transformer handles multi-layer Encoder + Decoder cleanly
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=True,
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids, pad_id):
        src_pad_mask = (src_ids == pad_id)
        tgt_pad_mask = (tgt_ids == pad_id)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1), device=tgt_ids.device)

        src_emb = self.embedding(src_ids)
        tgt_emb = self.embedding(tgt_ids)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_is_causal=True,
        )
        return self.lm_head(out)

# ==========================================
# 4. TRAINING LOOP (M2 OPTIMIZED)
# ==========================================
train_loader = DataLoader(CodeSearchNetStream("train"), batch_size=8, collate_fn=collate_batch)
model = CodingTransformer(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, pad_id=PAD_ID).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# ignore_index=PAD_ID ensures the model isn't penalized for padding slots
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

model.train()
for step, (src_ids, tgt_in, tgt_out) in enumerate(train_loader):
    src_ids, tgt_in, tgt_out = src_ids.to(device), tgt_in.to(device), tgt_out.to(device)

    optimizer.zero_grad()
    # Use bfloat16 on Mac M2 for faster training & lower RAM usage
    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
        logits = model(src_ids, tgt_in, pad_id=PAD_ID)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

    # Save a checkpoint every 1,000 steps
    if step > 0 and step % 1000 == 0:
        torch.save(model.state_dict(), f"function_ai_step_{step}.pt")
        print(f"Saved checkpoint at step {step}")

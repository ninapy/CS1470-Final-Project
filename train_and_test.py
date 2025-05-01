import torch
import torch.nn as nn
import torch.optim as optim
from transformer import TransformerModel
from tokenization import train_dataloader, text_transforms, vocab_transforms, SRC_LANGUAGE, TGT_LANGUAGE, tokenize_de, tokenize_en


# ========== Hyperparameters ==========
d_model = 512
n_heads = 8
n_layers = 6
dim_feedforward = 2048
dropout = 0.1
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== PAD Token Indices ==========
src_pad_idx = vocab_transforms[SRC_LANGUAGE]['<pad>']
tgt_pad_idx = vocab_transforms[TGT_LANGUAGE]['<pad>']

# ========== Initialize Model ==========
model = TransformerModel(
    src_vocab_size=len(vocab_transforms[SRC_LANGUAGE]),
    tgt_vocab_size=len(vocab_transforms[TGT_LANGUAGE]),
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    src_pad_idx=src_pad_idx,
    tgt_pad_idx=tgt_pad_idx,
    device=device
).to(device)

# ========== Optimizer + Scheduler ==========
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler from "Attention Is All You Need"
def get_scheduler(optimizer, d_model, warmup_steps=4000):
    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_scheduler(optimizer, d_model)

# ========== Loss ==========
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

# ========== Training ==========
def train():
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        total_batches = len(train_dataloader)
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        for i, (src, tgt) in enumerate(train_dataloader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            optimizer.zero_grad()
            logits, _ = model(src, tgt_input)
            logits = logits.reshape(-1, logits.shape[-1])
            tgt_target = tgt_target.reshape(-1)

            loss = criterion(logits, tgt_target)
            loss.backward()

            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f"[Epoch {epoch+1} | Batch {i+1}/{total_batches}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / total_batches
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} complete | Avg Loss = {avg_loss:.4f} | LR = {current_lr:.6f}")


# ========== Run ==========
if __name__ == "__main__":
    train()

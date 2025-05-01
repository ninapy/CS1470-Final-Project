import torch
from torchtext.data.metrics import bleu_score

def compute_bleu(model, data_iterator, src_field, trg_field, device):
    """
    Computes BLEU score over a dataset.

    Args:
        model: your Transformer model
        data_iterator: validation or test set iterator
        src_field: the SRC (source) Field
        trg_field: the TRG (target) Field
        device: device to run on (cuda or cpu)
    """

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in data_iterator:
            src = batch.src.to(device)
            trg = batch.trg.to(device)

            output, _ = model(src, trg[:, :-1])
            output = output.argmax(dim=-1)

            for pred, target in zip(output, trg[:, 1:]):  # shift target to match prediction
                pred_tokens = [trg_field.vocab.itos[idx] for idx in pred]
                target_tokens = [trg_field.vocab.itos[idx] for idx in target]

                # Remove special tokens (like <pad> and <eos>)
                pred_tokens = [tok for tok in pred_tokens if tok not in ['<pad>', '<eos>']]
                target_tokens = [tok for tok in target_tokens if tok not in ['<pad>', '<eos>']]

                preds.append(pred_tokens)
                targets.append([target_tokens])  # BLEU expects a list of references

    return bleu_score(preds, targets)

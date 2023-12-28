import torch
from utils.tokenizer import load_data
from utils.game_engine import history_to_legal_moves
from functools import cache


def tokens_list():
    tokens=[f"{letter}{number}" for letter in "ABCDEFGH" for number in range(1,9)]
    tokens.append("XX") #end-of-game token
    tokens.append("PP") #pad token
    return tokens

vocab=tokens_list()
vocab_size=len(vocab)
# block_size=64
batch_size=8
max_iter=10000
train_data, val_data=load_data()
split_token_index=vocab.index("XX")
split_points_train=torch.tensor([position for position, token in enumerate(train_data) if token==split_token_index])
split_points_test=torch.tensor([position for position, token in enumerate(val_data) if token==split_token_index])

def get_batch(split, block_size, batch_size=batch_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    split_points = split_points_train if split == 'train' else split_points_test
    ix = split_points[torch.randint(len(split_points)-1, (batch_size,))]
    # ix= torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


def train_model(model):
    torch.manual_seed(1337)

    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps_to_print_on=[500*x for x in range(1, max_iter//500)]
    for step in range(max_iter):
        xb,yb=get_batch("train", block_size=model.window_length)
        logits, loss=model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in steps_to_print_on:
            divergence=evaluate_kl_divergence(model, num_samples=100)
            print(f"Loss and divergence after {step} steps: {loss.item():.4f}, {divergence:.4f}")

@cache
def get_data_and_legal_move_distribution(window_length, num_samples):
    xb,yb=get_batch("test", window_length, num_samples)
    legal_moves=history_to_legal_moves(xb)
    legal_move_distribution=legal_moves/legal_moves.sum(dim=-1, keepdim=True)
    return xb, legal_move_distribution

def evaluate_kl_divergence(model, num_samples=10):
    xb, legal_move_distribution=get_data_and_legal_move_distribution(window_length=model.window_length, num_samples=num_samples)
    logits, loss=model(xb, None)
    kl_loss=torch.nn.KLDivLoss(reduction='batchmean')
    log_softmax=torch.nn.LogSoftmax(dim=-1)
    divergence=kl_loss(log_softmax(logits), legal_move_distribution)
    return float(divergence)


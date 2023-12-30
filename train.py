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
# block_size=64 # left this to an argument call
batch_size=8
train_data, val_data=load_data()
split_token_index=vocab.index("XX")
split_points_train=torch.tensor([position for position, token in enumerate(train_data) if token==split_token_index])
split_points_test=torch.tensor([position for position, token in enumerate(val_data) if token==split_token_index])
device='cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split, block_size, batch_size=batch_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    split_points = split_points_train if split == 'train' else split_points_test
    ix = split_points[torch.randint(len(split_points)-2, (batch_size,))]
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


def train_model(model, num_steps=10000, report_every_n_steps=500):
    torch.manual_seed(1337)
    model.train()
    model.to(device)

    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps_to_print_on=[report_every_n_steps*x for x in range(1, num_steps//report_every_n_steps)]+[num_steps-1]
    for step in range(num_steps):
        xb,yb=get_batch("train", block_size=model.window_length)
        xb, yb =xb.to(device), yb.to(device)
        logits, loss=model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in steps_to_print_on:
            model.eval()
            divergence=evaluate_kl_divergence(model)
            accuracy=evaluate_top_one_accuracy(model)    
            print(f"Loss, divergence, accuracy after {step} steps: {loss.item():.4f}, {divergence:.4f}, {accuracy:.4f}")
            model.train()

@cache
def get_data_and_legal_moves(window_length, num_samples, normalize=True):
    xb,yb=get_batch("test", window_length, num_samples)
    legal_moves=history_to_legal_moves(xb)
    return xb, legal_moves

def evaluate_kl_divergence(model, num_samples=1000):
    xb, legal_moves=get_data_and_legal_moves(window_length=model.window_length, num_samples=num_samples)
    xb, legal_moves =xb.to(device), legal_moves.to(device)
    legal_move_distribution=legal_moves/legal_moves.sum(dim=-1, keepdim=True)
    logits, loss=model(xb, None)
    kl_loss=torch.nn.KLDivLoss(reduction='batchmean')
    log_softmax=torch.nn.LogSoftmax(dim=-1)
    divergence=kl_loss(log_softmax(logits), legal_move_distribution)
    return float(divergence)

def evaluate_top_one_accuracy(model, num_samples=1000):
    xb, legal_moves=get_data_and_legal_moves(window_length=model.window_length, num_samples=num_samples, normalize=False)
    xb, legal_moves =xb.to(device), legal_moves.to(device)
    logits, loss=model(xb, None)
    largest_entry_locations=logits.argmax(dim=-1, keepdim=True)
    one_hot_predictions = torch.zeros(legal_moves.shape).scatter(dim=-1, index=largest_entry_locations, src=torch.ones(legal_moves.shape))
    accuracy=(legal_moves*one_hot_predictions).sum()/one_hot_predictions.sum()
    return accuracy


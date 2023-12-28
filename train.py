import torch
from utils.tokenizer import load_data

def tokens_list():
    tokens=[f"{letter}{number}" for letter in "ABCDEFGH" for number in range(1,9)]
    tokens.append("XX") #end-of-game token
    tokens.append("PP") #pad token
    return tokens

vocab=tokens_list()
vocab_size=len(vocab)
block_size=64
batch_size=8
train_data, val_data=load_data()
split_token_index=vocab.index("XX")
split_points_train=torch.tensor([position for position, token in enumerate(train_data) if token==split_token_index])
split_points_test=torch.tensor([position for position, token in enumerate(val_data) if token==split_token_index])

def get_batch(split):
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
    steps_to_print=1
    for step in range(10000):
        xb,yb=get_batch("train")
        logits, loss=model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step==steps_to_print:
            print(f"Loss after {step} steps: {loss.item()}")
            steps_to_print*=2


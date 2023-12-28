import othello_gpt
from utils.tokenizer import encode, decode
import torch
import train


def test_training():
    model=othello_gpt.OthelloGPT(num_layers=1, d_model=8, n_heads=2, window_length=64)
    train.train_model(model)
    return



def test_generation():
    model=othello_gpt.OthelloGPT(num_layers=1, d_model=8, n_heads=2)
    start_text="C4"
    x=decode(model.generate(torch.unsqueeze(encode(start_text),dim=0), max_new_tokens=10)[0])
    print(x)


# test_generation()
test_training()


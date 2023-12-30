import othello_gpt
from utils.tokenizer import encode, decode
import torch
import train
import cProfile
import pickle

def test_training(save=False):
    model=othello_gpt.OthelloGPT(num_layers=2, d_model=32, n_heads=8, window_length=4)
    train.train_model(model, num_steps=1000, report_every_n_steps=500)
    if save:
        with open("trained_model_test.pkl", 'wb') as f:
            pickle.dump(model, f)
    return


def full_scale_training(save=False):
    model=othello_gpt.OthelloGPT(num_layers=8, d_model=512, n_heads=8, window_length=64)
    train.train_model(model)
    if save:
        with open("trained_model_full.pkl") as f:
            pickle.dump(model, f)
    return



def test_generation():
    model=othello_gpt.OthelloGPT(num_layers=1, d_model=8, n_heads=2)
    start_text="C4"
    x=decode(model.generate(torch.unsqueeze(encode(start_text),dim=0), max_new_tokens=10)[0])
    print(x)


test_training()
# full_scale_training()
# cProfile.run("test_training()")


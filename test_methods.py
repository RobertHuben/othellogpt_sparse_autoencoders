import othello_gpt
import autoencoder
from utils.tokenizer import encode, decode
import torch
import train
import cProfile
import pickle


device='cuda' if torch.cuda.is_available() else 'cpu'

def test_small_training(save=False):
    model=othello_gpt.OthelloGPT(num_layers=2, d_model=32, n_heads=8, window_length=4)
    train.train_othello_gpt_model(model, num_steps=2000, report_every_n_steps=100)
    if save:
        with open("trained_model_test.pkl", 'wb') as f:
            pickle.dump(model, f)
    return


def full_scale_training(save=False):
    model=othello_gpt.OthelloGPT(num_layers=8, d_model=512, n_heads=8, window_length=64)
    train.train_othello_gpt_model(model, batch_size=64, num_steps=100000, report_every_n_steps=500)
    if save:
        with open("trained_model_full.pkl", 'wb') as f:
            pickle.dump(model, f)
    return



def test_generation():
    model=othello_gpt.OthelloGPT(num_layers=1, d_model=8, n_heads=2)
    start_text="C4"
    x=decode(model.generate(torch.unsqueeze(encode(start_text),dim=0), max_new_tokens=10)[0])
    print(x)

def test_unpickle():
    # model=othello_gpt.OthelloGPT(8,512,8)

    with open("trained_model_full.pkl", 'rb') as f:
        model=pickle.load(f)
    start_text="XX C4"
    model_input=torch.unsqueeze(encode(start_text),dim=0).to(device)
    # xb,yb=train.get_batch("train", block_size=model.window_length)
    x=decode(model.generate(model_input, max_new_tokens=10)[0])
    print(x)


def test_sae_training(save=False):
    with open("trained_model_test.pkl", 'rb') as f:
        language_model=pickle.load(f)

    sae_model=autoencoder.SparseAutoencoder(language_model.d_model, feature_ratio=2, sparsity_coeff=.1)
    train.train_sparse_autoencoder(sae_model, language_model, target_layer=1, num_steps=2000, report_every_n_steps=50)
    return

def full_sae_run(target_layer, save=True):
    with open("trained_model_full.pkl", 'rb') as f:
        language_model=pickle.load(f)

    sae_model=autoencoder.SparseAutoencoder(language_model.d_model, feature_ratio=2, sparsity_coeff=.1)
    train.train_sparse_autoencoder(sae_model, language_model, target_layer=target_layer, num_steps=4000, report_every_n_steps=100)
    if save:
        with open(f"saes/sae_layer_{target_layer}.pkl", 'wb') as f:
            pickle.dump(sae_model, f)
    return


# test_small_training(save=True)
full_scale_training(save=True)
# cProfile.run("test_training()")

# test_unpickle()

# test_sae_training()

# for n in range(1, 9):
#     full_sae_run(target_layer=n)

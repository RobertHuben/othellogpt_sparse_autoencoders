import othello_gpt
import autoencoder
import linear_probes
from utils.tokenizer import encode, decode
import torch
from train import train_model
import cProfile


device='cuda' if torch.cuda.is_available() else 'cpu'

def test_small_training(save=False):
    num_epochs=2
    model=othello_gpt.OthelloGPT(num_layers=2, d_model=32, n_heads=8, window_length=4)
    train_model(model, train_dataset_type="gpt_train_small", eval_dataset_type="gpt_test", num_epochs=num_epochs, report_every_n_steps=10, batch_size=8)
    
    if save:
        with open("trained_model_test.pkl", 'wb') as f:
            torch.save(model, f)


def full_scale_training(save=False):
    num_epochs=2
    report_every_n_steps=500
    batch_size=64
    model=othello_gpt.OthelloGPT(num_layers=8, d_model=512, n_heads=8, window_length=64)
    train_model(model, train_dataset_type="gpt_train_small", eval_dataset_type="gpt_test", num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size)
    
    if save:
        with open("trained_model_full.pkl", 'wb') as f:
            torch.save(model, f)



def test_generation():
    model=othello_gpt.OthelloGPT(num_layers=1, d_model=8, n_heads=2)
    start_text="C4"
    x=decode(model.generate(torch.unsqueeze(encode(start_text),dim=0), max_new_tokens=10)[0])
    print(x)

def test_unpickle():
    # model=othello_gpt.OthelloGPT(8,512,8)

    with open("trained_model_full.pkl", 'rb') as f:
        model=torch.load(f)
    start_text="XX C4"
    model_input=torch.unsqueeze(encode(start_text),dim=0).to(device)
    # xb,yb=train.get_batch("train", block_size=model.window_length)
    x=decode(model.generate(model_input, max_new_tokens=10)[0])
    print(x)


# def test_sae_training(save=False):
#     with open("trained_model_test.pkl", 'rb') as f:
#         language_model=torch.load(f)

#     sae_model=autoencoder.SparseAutoencoder(language_model.d_model, feature_ratio=2, sparsity_coeff=.00086)
#     train.train_sparse_autoencoder(sae_model, language_model, target_layer=1, num_steps=2000, report_every_n_steps=50)
#     return

# def full_sae_run(target_layer, save=True):
#     with open("trained_model_full.pkl", 'rb') as f:
#         language_model=torch.load(f)

#     sae_model=autoencoder.SparseAutoencoder(language_model.d_model, feature_ratio=2, sparsity_coeff=.00086)
#     train.train_sparse_autoencoder(sae_model, language_model, sae_batch_size=64, target_layer=target_layer, num_steps=4000, report_every_n_steps=500)
#     if save:
#         with open(f"saes/sae_layer_{target_layer}.pkl", 'wb') as f:
#             torch.save(sae_model, f)
#     return

def test_linear_probes(target_layer, save=True):
    with open("trained_model_test.pkl", 'rb') as f:
        language_model=torch.load(f)
    num_epochs=2
    report_every_n_steps=8
    batch_size=64
    linear_probe_model=linear_probes.LinearProbe(language_model, layer_num=target_layer)
    train_model(linear_probe_model, train_dataset_type="probe_train_small", eval_dataset_type="probe_test", num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size)
    
    if save:
        with open(f"probes/probe_layer_{target_layer}.pkl", 'wb') as f:
            torch.save(linear_probe_model, f)

def full_probe_run(target_layer, save=True):
    with open("trained_model_full.pkl", 'rb') as f:
        language_model=torch.load(f, map_location=device)
    num_epochs=2
    report_every_n_steps=500
    batch_size=64
    linear_probe_model=linear_probes.LinearProbe(language_model, layer_num=1)
    train_model(linear_probe_model, train_dataset_type="probe_train_corpus", eval_dataset_type="probe_test_corpus", num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size)
    
    if save:
        with open(f"probes/probe_layer_{target_layer}.pkl", 'wb') as f:
            torch.save(linear_probe_model, f)


# test_small_training(save=True)
# full_scale_training(save=True)
# cProfile.run("test_training()")

# test_unpickle()

# test_sae_training()

# for n in range(1, 9):
#     full_sae_run(target_layer=n)

test_linear_probes(1)
# for n in range(1, 9):
#     full_probe_run(target_layer=n)

# full_probe_run(target_layer=4)
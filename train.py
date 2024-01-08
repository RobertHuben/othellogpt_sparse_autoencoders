import torch
from utils.dataloaders import get_dataloader
from othello_gpt import OthelloGPT
from linear_probes import LinearProbe
from autoencoder import SparseAutoencoder

device='cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, train_dataset_type, eval_dataset_type, batch_size=64, num_epochs=2, report_every_n_steps=500):
    '''
    model be a nn.Module object, and have a print_evaluation() method
    '''
    torch.manual_seed(1337)
    model.to(device)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
    step=0

    print(f"Beginning model training on {device}!")
    for epoch in range(num_epochs):
        train_dataloader=iter(get_dataloader(mode=train_dataset_type, window_length=model.window_length, batch_size=batch_size))
        print(f"Beginning epoch {epoch+1}/{num_epochs}. Epoch duration is {len(train_dataloader)} steps.")
        for input_batch,label_batch in train_dataloader:
            step+=1
            optimizer.zero_grad(set_to_none=True)
            logits, loss=model(input_batch, label_batch)
            loss.backward()
            optimizer.step()
            if step %report_every_n_steps==0:
                model.print_evaluation(loss, eval_dataset_type, step)
    else:
        model.print_evaluation(train_loss=loss, eval_dataset_type=eval_dataset_type, step="Omega", details=True)

# def train_othello_gpt_model(model, batch_size=8, num_epochs=2, report_every_n_steps=500):
#     torch.manual_seed(1337)
#     model.to(device)
#     model.train()

#     optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
#     step=0
#     print(f"Beginning model training on {device}!")
#     for epoch in range(num_epochs):
#         train_dataloader=iter(get_dataloader(mode="gpt_train", window_length=model.window_length, batch_size=batch_size))
#         print(f"Beginning epoch {epoch+1}/{num_epochs}. Epoch duration is {len(train_dataloader)} steps")
#         for input_batch,label_batch in train_dataloader:
#             step+=1
#             logits, loss=model(input_batch, label_batch)
#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()
#             if step %report_every_n_steps==0:
#                 give_othello_gpt_report(model, loss, step)
#     else:
#         give_othello_gpt_report(model, loss, step="Omega", details=True)

# def evaluate_othello_gpt(model, loss, step, details=False):
#     model.eval()
#     test_loss=evaluate_test_loss(model)
#     divergence=evaluate_kl_divergence(model)
#     accuracy=evaluate_top_one_accuracy(model)
#     if details:
#         accuracy_by_turn=evaluate_top_one_accuracy_by_turn(model)
#         print(f"After training step {step}, accuracy on turns 1, 2, 3, ...: {accuracy_by_turn}")
#     print(f"Train loss, test loss, divergence, accuracy after {step} steps: {loss.item():.4f}, {test_loss.item():.4f}, {divergence:.4f}, {accuracy:.4f}")
#     model.train()

# def evaluate_linear_probe(model, loss, step, details=False):
#     del details
#     accuracy=evaluate_top_one_board_state_accuracy(model)
#     print(f"Train loss and test accuracy after {step} steps: {loss.item():.4f}, {accuracy:.4f}")

# @cache
# def get_data_and_legal_moves(window_length, num_samples, key=0):
#     test_dataloader=iter(get_dataloader("gpt_test", window_length=window_length, batch_size=8))
#     test_labels, test_input= next(test_dataloader)
#     test_labels=test_labels.to("cpu")
#     legal_moves=history_to_legal_moves(test_labels)
#     test_labels, legal_moves=test_labels.to(device), legal_moves.to(device)
#     return test_labels, legal_moves



# def train_sparse_autoencoder(sae_model, language_model, target_layer=1, sae_batch_size=64, num_steps=1000, report_every_n_steps=50):
#     torch.manual_seed(1337)
#     sae_model.to(device)
#     language_model.to(device)
#     sae_model.train()

#     # freeze model weights
#     for parameter in language_model.parameters():
#         parameter.requires_grad=False
    
#     train_sae_dataloader=iter(get_dataloader(mode="sae_train", window_length=language_model.window_length, batch_size=sae_batch_size))

#     print(f"Beginning to train a SAE for layer {target_layer} on {device}!")
#     optimizer=torch.optim.AdamW(sae_model.parameters(), lr=1e-3)
#     steps_to_print_on=[report_every_n_steps*x for x in range(1, num_steps//report_every_n_steps)]+[num_steps-1]
#     for step in range(num_steps):
#         try:
#             input_batch,label_batch = next(train_sae_dataloader)
#         except StopIteration:
#             # loops over epochs
#             train_sae_dataloader=iter(get_dataloader(mode="sae_train", window_length=language_model.window_length, batch_size=sae_batch_size))
#             input_batch,_ = next(train_sae_dataloader)
        
#         model_hidden_layer=language_model.intermediate_residual_stream(input_batch, layer_num=target_layer)

#         reconstruction, hidden_layer, loss, reconstruction_loss, sparsity_loss=sae_model(model_hidden_layer)

#         loss.backward()
#         optimizer.step()
#         if step in steps_to_print_on:
#             sparsity_percent=(hidden_layer>0).sum()/hidden_layer.numel()
#             print(f"Total loss, rec loss, sparsity loss, sparsity percent after {step}/{num_steps} steps: {loss.item():.4f}, {reconstruction_loss.item():.4f}, {sparsity_loss.item():.4f}, {sparsity_percent.item():.4%}")




# def train_linear_probe(linear_probe_model, probe_batch_size=64, num_epochs=10, report_every_n_steps=50):
#     torch.manual_seed(1337)
#     linear_probe_model.to(device)
#     linear_probe_model.othello_gpt_model.to(device)
#     linear_probe_model.train()
#     window_length=linear_probe_model.othello_gpt_model.window_length

#     # # freeze model weights
#     # for parameter in linear_probe_model.othello_gpt_model.parameters():
#     #     parameter.requires_grad=False
    

#     print(f"Beginning to train a linear probe on layers {linear_probe_model.layer_num} on {device}!")
#     optimizer=torch.optim.AdamW(linear_probe_model.parameters(), lr=1e-3)
#     steps_to_print_on=[report_every_n_steps*x for x in range(1, 1000)]
#     step=0
#     for epoch in range(num_epochs):
#         train_probe_dataloader=iter(get_dataloader(mode="probe_train", window_length=window_length, batch_size=probe_batch_size))
#         print(f"Starting training epoch {epoch+1}/{num_epochs}!")
#         for input_batch,label_batch in train_probe_dataloader:
#             step+=1
#             predictions, loss=linear_probe_model(input_batch, label_batch)
#             loss.backward()
#             optimizer.step()
#             if step in steps_to_print_on:
#                 accuracy=evaluate_top_one_board_state_accuracy(linear_probe_model)
#                 print(f"Train loss and test accuracy after {step} steps: {loss.item():.4f}, {accuracy:.4f}")

# def evaluate_top_one_board_state_accuracy(linear_probe_model, num_samples=80):
#     batch_size=1
#     accuracies=[]
#     window_length=linear_probe_model.window_length

#     test_probe_dataloader=iter(get_dataloader(mode="probe_test", window_length=window_length, batch_size=batch_size))

#     for input_batch,label_batch in test_probe_dataloader:
#         predictions, loss=linear_probe_model(input_batch)
#         one_hot_predictions=predictions.argmax(dim=3)

#         correct_predictions=(label_batch==one_hot_predictions).sum()
#         total_predictions=(label_batch>=0).sum()
#         accuracies.append(correct_predictions/total_predictions)

#     return float(torch.tensor(accuracies).mean())
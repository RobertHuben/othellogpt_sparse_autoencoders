import torch
from utils.tokenizer import load_data
from utils.game_engine import history_to_legal_moves, tokens_list
from functools import cache
from torch.utils.data import Dataset, DataLoader, RandomSampler

device='cuda' if torch.cuda.is_available() else 'cpu'

class OthelloDataset(Dataset):
    def __init__(self, file_location, window_length=64, device="cpu"):
        super().__init__()
        self.vocab=tokens_list()
        self.reverse_vocab={text:num for num, text in enumerate(self.vocab)}
        self.window_length=window_length
        self.device=device
        with open(file_location,'r') as f:
            self.games=f.read().split("\n")

    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, index):
        game=self.games[index]
        extended_window_length=self.window_length+1
        extended_moves=[self.reverse_vocab[move] for move in game.split(" ")]
        if len(extended_moves)>extended_window_length:
            extended_moves=extended_moves[:extended_window_length]
        elif len(extended_moves)<extended_window_length:
            extended_moves.extend([self.reverse_vocab["PP"] for _ in range(extended_window_length-len(extended_moves))])
        extended_moves=torch.tensor(extended_moves, device=self.device)
        inputs =extended_moves[:self.window_length]
        labels =extended_moves[1:]
        return inputs, labels

class LabelledOthelloDataset(Dataset):
    def __init__(self, file_location, window_length=64, device="cpu", use_ally_enemy=True):
        super().__init__()
        self.vocab=tokens_list()
        self.reverse_vocab={text:num for num, text in enumerate(self.vocab)}
        self.window_length=window_length
        self.device=device
        self.turn_mask=[(-1)**n for n in range(self.window_length)] if use_ally_enemy else [1 for n in range(self.window_length)]
        with open(file_location,'r') as f:
            self.games=f.read().split("\n")

    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, index):
        game=self.games[index]
        game_moves, board_states=game.split("/")
        board_states_by_turn=[[(int(pos)*tm)%3 for pos in turn_state.split(" ")] for tm, turn_state in zip(self.turn_mask,board_states.split(";"))]
        extended_window_length=self.window_length+1
        extended_moves=[self.reverse_vocab[move] for move in game_moves.split(" ")]
        if len(extended_moves)>extended_window_length:
            extended_moves=extended_moves[:extended_window_length]
            board_states_by_turn=board_states_by_turn[:extended_window_length]
        elif len(extended_moves)<extended_window_length:
            extended_moves.extend([self.reverse_vocab["PP"] for _ in range(extended_window_length-len(extended_moves))])
            board_states_by_turn.extend([[-100 for _ in range(64)] for __ in range(extended_window_length-len(board_states_by_turn))])
        labels=torch.tensor(board_states_by_turn[:self.window_length], device=self.device)
        extended_moves=torch.tensor(extended_moves, device=self.device)
        inputs =extended_moves[:self.window_length]
        return inputs, labels



def get_dataloder(mode, window_length, batch_size, require_labels=False):
    if mode =="gpt_train":
        file_location="datasets/othello_gpt_training_corpus.txt"
        dataset_type=OthelloDataset
    elif mode=="gpt_test":
        file_location="datasets/othello_gpt_test_corpus.txt"
        dataset_type=OthelloDataset
    elif mode=="sae_train":
        file_location="datasets/sae_training_corpus.txt"
        dataset_type=OthelloDataset
    elif mode=="probe_train":
        file_location="datasets/probe_train_corpus.txt"
        dataset_type=LabelledOthelloDataset
    elif mode=="probe_train":
        file_location="datasets/probe_test_corpus.txt"
        dataset_type=LabelledOthelloDataset
    elif mode=="probe_test":
        file_location="datasets/board_state_classifier_test_corpus.txt"
        dataset_type=LabelledOthelloDataset
    dataset=dataset_type(file_location, window_length=window_length, device=device)
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_othello_gpt_model(model, batch_size=8, num_steps=10000, report_every_n_steps=500):
    torch.manual_seed(1337)
    model.to(device)
    model.train()

    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_dataloader=iter(get_dataloder(mode="gpt_train", window_length=model.window_length, batch_size=batch_size))
    steps_to_print_on=[report_every_n_steps*x for x in range(1, num_steps//report_every_n_steps)]+[num_steps-1]
    print(f"Beginning model training on {device}!")
    for step in range(num_steps):
        try:
            input_batch,label_batch = next(train_dataloader)
        except StopIteration:
            # loops over epochs
            train_dataloader=iter(get_dataloder(mode="gpt_train", window_length=model.window_length, batch_size=batch_size))
            input_batch,label_batch = next(train_dataloader)
        logits, loss=model(input_batch, label_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in steps_to_print_on:
            if step == steps_to_print_on[0]:
                print("Beginning first evaluation, this may take some time to cache the results")
            model.eval()
            test_loss=evaluate_test_loss(model)
            divergence=evaluate_kl_divergence(model)
            accuracy=evaluate_top_one_accuracy(model)
            if step == steps_to_print_on[-1]:
                accuracy_by_turn=evaluate_top_one_accuracy_by_turn(model)
                print(f"After training step {step}/{num_steps}, accuracy on turns 1, 2, 3, ...: {accuracy_by_turn}")

            print(f"Train loss, test loss, divergence, accuracy after {step}/{num_steps} steps: {loss.item():.4f}, {test_loss.item():.4f}, {divergence:.4f}, {accuracy:.4f}")
            model.train()

@cache
def get_data_and_legal_moves(window_length, num_samples, key=0):
    test_dataloader=iter(get_dataloder("gpt_test", window_length=window_length, batch_size=8))
    test_labels, test_input= next(test_dataloader)
    test_labels=test_labels.to("cpu")
    legal_moves=history_to_legal_moves(test_labels)
    test_labels, legal_moves=test_labels.to(device), legal_moves.to(device)
    return test_labels, legal_moves

def evaluate_test_loss(model):
    test_dataloader=iter(get_dataloder("gpt_test", window_length=model.window_length, batch_size=8))
    test_labels, test_input= next(test_dataloader)
    logits, loss=model(test_labels, test_input)
    return loss

def evaluate_kl_divergence(model, num_samples=80):
    batch_size=8
    batches=num_samples//batch_size
    divergences=[]
    for n in range(batches):
        xb, legal_moves=get_data_and_legal_moves(window_length=model.window_length, num_samples=batch_size, key=n)
        legal_move_distribution=legal_moves/legal_moves.sum(dim=-1, keepdim=True)
        logits, loss=model(xb, None)
        kl_loss=torch.nn.KLDivLoss(reduction='batchmean')
        log_softmax=torch.nn.LogSoftmax(dim=-1)
        divergences.append(kl_loss(log_softmax(logits), legal_move_distribution))
    return float(torch.tensor(divergences).mean())

def evaluate_top_one_accuracy(model, num_samples=80):
    batch_size=8
    batches=num_samples//batch_size
    accuracies=[]
    for n in range(batches):
        xb, legal_moves=get_data_and_legal_moves(window_length=model.window_length, num_samples=batch_size, key=n)
        logits, loss=model(xb, None)
        largest_entry_locations=logits.argmax(dim=-1, keepdim=True)
        one_hot_predictions = torch.zeros(legal_moves.shape).to(device)
        one_hot_predictions = one_hot_predictions.scatter(dim=-1, index=largest_entry_locations, src=torch.ones(legal_moves.shape, device=device))
        accuracies.append((legal_moves*one_hot_predictions).sum()/one_hot_predictions.sum())
    return float(torch.tensor(accuracies).mean())

def evaluate_top_one_accuracy_by_turn(model, num_samples=80):
    batch_size=8
    batches=num_samples//batch_size
    accuracies=[]
    for n in range(batches):
        xb, legal_moves=get_data_and_legal_moves(window_length=model.window_length, num_samples=batch_size, key=n)
        logits, loss=model(xb, None)
        largest_entry_locations=logits.argmax(dim=-1, keepdim=True)
        one_hot_predictions = torch.zeros(legal_moves.shape).to(device)
        one_hot_predictions = one_hot_predictions.scatter(dim=-1, index=largest_entry_locations, src=torch.ones(legal_moves.shape, device=device))
        accuracies.append((legal_moves*one_hot_predictions).sum(dim=(0,2))/one_hot_predictions.sum(dim=(0,2)))
    return torch.stack(accuracies).mean(dim=0)


def train_sparse_autoencoder(sae_model, language_model, target_layer=1, sae_batch_size=64, num_steps=1000, report_every_n_steps=50):
    torch.manual_seed(1337)
    sae_model.to(device)
    language_model.to(device)
    sae_model.train()

    # freeze model weights
    for parameter in language_model.parameters():
        parameter.requires_grad=False
    
    train_sae_dataloader=iter(get_dataloder(mode="sae_train", window_length=language_model.window_length, batch_size=sae_batch_size))

    print(f"Beginning to train a SAE for layer {target_layer} on {device}!")
    optimizer=torch.optim.AdamW(sae_model.parameters(), lr=1e-3)
    steps_to_print_on=[report_every_n_steps*x for x in range(1, num_steps//report_every_n_steps)]+[num_steps-1]
    for step in range(num_steps):
        try:
            input_batch,label_batch = next(train_sae_dataloader)
        except StopIteration:
            # loops over epochs
            train_sae_dataloader=iter(get_dataloder(mode="sae_train", window_length=language_model.window_length, batch_size=sae_batch_size))
            input_batch,_ = next(train_sae_dataloader)
        
        model_hidden_layer=language_model.intermediate_residual_stream(input_batch, layer_num=target_layer)

        reconstruction, hidden_layer, loss, reconstruction_loss, sparsity_loss=sae_model(model_hidden_layer)

        loss.backward()
        optimizer.step()
        if step in steps_to_print_on:
            sparsity_percent=(hidden_layer>0).sum()/hidden_layer.numel()
            print(f"Total loss, rec loss, sparsity loss, sparsity percent after {step}/{num_steps} steps: {loss.item():.4f}, {reconstruction_loss.item():.4f}, {sparsity_loss.item():.4f}, {sparsity_percent.item():.4%}")




def train_linear_probe(linear_probe_model, probe_batch_size=64, num_epochs=10, report_every_n_steps=50):


    torch.manual_seed(1337)
    linear_probe_model.to(device)
    linear_probe_model.othello_gpt_model.to(device)
    linear_probe_model.train()
    window_length=linear_probe_model.othello_gpt_model.window_length

    # freeze model weights
    for parameter in linear_probe_model.othello_gpt_model.parameters():
        parameter.requires_grad=False
    

    print(f"Beginning to train a linear probe on layers {linear_probe_model.layer_num} on {device}!")
    optimizer=torch.optim.AdamW(linear_probe_model.parameters(), lr=1e-3)
    steps_to_print_on=[report_every_n_steps*x for x in range(1, 1000)]
    step=0
    for epoch in range(num_epochs):
        train_probe_dataloader=iter(get_dataloder(mode="probe_train", window_length=window_length, batch_size=probe_batch_size))
        print(f"Starting training epoch {epoch+1}/{num_epochs}!")
        for input_batch,label_batch in train_probe_dataloader:
            step+=1
            predictions, loss=linear_probe_model(input_batch, label_batch)
            loss.backward()
            optimizer.step()
            if step in steps_to_print_on:
                accuracy=evaluate_top_one_board_state_accuracy(linear_probe_model)
                print(f"Train loss and test accuracy after {step} steps: {loss.item():.4f}, {accuracy:.4f}")

def evaluate_top_one_board_state_accuracy(linear_probe_model, num_samples=80):
    batch_size=8
    batches=num_samples//batch_size
    accuracies=[]

    window_length=linear_probe_model.othello_gpt_model.window_length

    test_probe_dataloader=iter(get_dataloder(mode="probe_test", window_length=window_length, batch_size=batch_size))

    for n in range(batches):
        input_batch,label_batch = next(test_probe_dataloader)
        predictions, loss=linear_probe_model(input_batch)
        one_hot_predictions=predictions.argmax(dim=3)

        correct_predictions=(label_batch==one_hot_predictions).sum()
        total_predictions=(label_batch>=0).sum()
        accuracies.append(correct_predictions/total_predictions)

    return float(torch.tensor(accuracies).mean())
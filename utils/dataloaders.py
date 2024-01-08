import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils.game_engine import history_to_legal_moves, tokens_list
from functools import cache

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
        extended_moves=[self.reverse_vocab[move] for move in game.split(" ") if move]
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



def get_dataloader(mode, window_length, batch_size, require_labels=False):
    mode_lookups={
        "gpt_train":        ["datasets/othello_gpt_training_corpus.txt",        OthelloDataset],
        "gpt_train_small":  ["datasets/small_othello_gpt_training_corpus.txt",  OthelloDataset],
        "gpt_test":         ["datasets/othello_gpt_test_corpus.txt",            OthelloDataset],
        "sae_train":        ["datasets/sae_training_corpus.txt",                OthelloDataset],
        "probe_train":      ["datasets/probe_train_corpus.txt",                 LabelledOthelloDataset],
        "probe_train_small":["datasets/small_probe_training_corpus.txt",        LabelledOthelloDataset],
        "probe_test":       ["datasets/probe_test_corpus.txt",                  LabelledOthelloDataset],
    }
    file_location, dataset_type=mode_lookups[mode]
    # if mode =="gpt_train":
    #     file_location="datasets/othello_gpt_training_corpus.txt"
    #     dataset_type=OthelloDataset
    # if mode =="gpt_train_small":
    #     file_location="datasets/small_othello_gpt_training_corpus.txt"
    #     dataset_type=OthelloDataset
    # elif mode=="gpt_test":
    #     file_location="datasets/othello_gpt_test_corpus.txt"
    #     dataset_type=OthelloDataset
    # elif mode=="sae_train":
    #     file_location="datasets/sae_training_corpus.txt"
    #     dataset_type=OthelloDataset
    # elif mode=="probe_train":
    #     file_location="datasets/probe_train_corpus.txt"
    #     dataset_type=LabelledOthelloDataset
    # elif mode=="probe_test":
    #     file_location="datasets/probe_test_corpus.txt"
    #     dataset_type=LabelledOthelloDataset
    # # elif mode=="probe_test":
    #     file_location="datasets/board_state_classifier_test_corpus.txt"
    #     dataset_type=LabelledOthelloDataset
    dataset=dataset_type(file_location, window_length=window_length, device=device)
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

@cache
def get_data_and_legal_moves(window_length, num_samples, eval_dataset_type="gpt_test", key=0):
    # key is a dummy variable used for the caching
    test_dataloader=iter(get_dataloader(eval_dataset_type, window_length=window_length, batch_size=num_samples))
    test_labels, test_input= next(test_dataloader)
    test_labels=test_labels.to("cpu")
    legal_moves=history_to_legal_moves(test_labels)
    test_labels, legal_moves=test_labels.to(device), legal_moves.to(device)
    return test_labels, legal_moves

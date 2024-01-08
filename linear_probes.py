import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.dataloaders import get_dataloader

class LinearProbe(torch.nn.Module):

    def __init__(self, othello_gpt_model, layer_num):
        super().__init__()
        self.othello_gpt_model=othello_gpt_model
        self.layer_num=layer_num
        self.board_size=64
        self.classifier=nn.ModuleList(nn.Linear(othello_gpt_model.d_model, self.board_size) for _ in range(3))
        self.window_length=self.othello_gpt_model.window_length
        # freeze othello gpt model weights
        for parameter in self.othello_gpt_model.parameters():
            parameter.requires_grad=False

    def forward(self, input, target=None):
        #input is B-by-W, where W is the length of the context window
        #target are B-W-64, and are 0,1,2 for the 3 classes, with -100 for indices that should be ignored (e.g after the game)
        logits=self.othello_gpt_model.intermediate_residual_stream(input, self.layer_num) #B-W-512
        predictions=torch.stack([classifier_shard(logits) for classifier_shard in self.classifier], dim=3) #B-W-64-3
        if target is None:
            loss=None
        else:
            loss_function=torch.nn.CrossEntropyLoss()
            loss=loss_function(input=predictions.flatten(end_dim=-2), target=target.flatten())
        return predictions, loss

    def print_evaluation(self, train_loss, eval_dataset_type, step_number="N/A", details=False):
        del details
        accuracy=self.evaluate_top_one_board_state_accuracy(eval_dataset_type)
        print(f"Train loss and test accuracy after {step_number} steps: {train_loss.item():.4f}, {accuracy:.4f}")

    def evaluate_top_one_board_state_accuracy(self, eval_dataset_type="probe_test", num_samples=80):
        batch_size=1
        accuracies=[]
        window_length=self.window_length
        test_probe_dataloader=iter(get_dataloader(mode=eval_dataset_type, window_length=window_length, batch_size=batch_size))

        for input_batch,label_batch in test_probe_dataloader:
            predictions, loss=self(input_batch)
            one_hot_predictions=predictions.argmax(dim=3)
            correct_predictions=(label_batch==one_hot_predictions).sum()
            total_predictions=(label_batch>=0).sum()
            accuracies.append(correct_predictions/total_predictions)

        return float(torch.tensor(accuracies).mean())
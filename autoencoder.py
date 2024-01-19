import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.dataloaders import get_dataloader
from tqdm import tqdm

device='cuda' if torch.cuda.is_available() else 'cpu'


class SparseAutoencoder(nn.Module):

    def __init__(self, othello_gpt_model, layer_num, feature_ratio, sparsity_coeff=.1, initialization_weights=None):
        super().__init__()
        self.othello_gpt_model=othello_gpt_model
        self.window_length=self.othello_gpt_model.window_length
        self.layer_num=layer_num
        self.sparsity_coeff=sparsity_coeff
        self.hidden_layer_size=int(self.othello_gpt_model.d_model*feature_ratio)
        if initialization_weights==None:
            initialization_weights=torch.normal(0, 1, (self.othello_gpt_model.d_model, self.hidden_layer_size))
        self.encoder_decoder_matrix=nn.Parameter(initialization_weights)
        self.encoder_bias=nn.Parameter(torch.normal(0,1, (self.hidden_layer_size,)))
        self.activation=nn.ReLU()
        self.layernorm=nn.LayerNorm((self.othello_gpt_model.d_model))
        for parameter in self.layernorm.parameters():
            parameter.requires_grad=False
        # freeze othello gpt model weights
        for parameter in self.othello_gpt_model.parameters():
            parameter.requires_grad=False


    def forward(self, input, labels):
        del labels # we're trained on unlabeled, but want a second argument to match other methods
        logits=self.othello_gpt_model.intermediate_residual_stream(input, layer_num=self.layer_num) #run the model
        normalized_logits=self.layernorm(logits) #layernorm regularizes the input to self
        normalized_encoder_decoder_matrix=F.normalize(self.encoder_decoder_matrix, p=2, dim=1) #need to L2 regularize the matrix
        hidden_layer=self.activation(normalized_logits@normalized_encoder_decoder_matrix + self.encoder_bias)
        reconstruction=hidden_layer@normalized_encoder_decoder_matrix.transpose(0,1)
        sparsity_loss=torch.norm(hidden_layer, p=1, dim=-1).sum()/hidden_layer.numel()
        reconstruction_loss=torch.norm(normalized_logits-reconstruction, p=2, dim=-1).sum()/input.numel()
        total_loss=reconstruction_loss+self.sparsity_coeff*sparsity_loss
        return (reconstruction,hidden_layer,reconstruction_loss, sparsity_loss, logits), total_loss
    
    def print_evaluation(self, train_loss, eval_dataset_type, step_number="N/A", details=False):
        del details
        test_loss, percent_active = self.evaluate_test_losses_and_sparsity(eval_dataset_type)
        fraction_variance_unexplained= self.evaluate_variance_unexplaiend(eval_dataset_type)
        tqdm.write(f"Train loss, test loss, test features active, and test FVU, percent after {step_number} steps: {train_loss.item():.4f}, {test_loss.item():.4f}, {percent_active:.4%}, {fraction_variance_unexplained:.4%}")

    def evaluate_test_losses_and_sparsity(self, eval_dataset_type):

        test_dataloader=iter(get_dataloader(eval_dataset_type, window_length=self.window_length, batch_size=8))
        total_test_loss=torch.zeros((1), requires_grad=False, device=device)
        total_active_features=0
        for test_input, test_labels in test_dataloader:
            del test_labels
            (reconstruction,hidden_layer,reconstruction_loss, sparsity_loss, logits), total_loss=self(test_input, None)
            total_active_features+=(hidden_layer>0).sum()
            total_test_loss+=total_loss
        
        return total_test_loss/len(test_dataloader), total_active_features/(len(test_dataloader)*hidden_layer.numel())

    def evaluate_variance_unexplaiend(self, eval_dataset_type):
        test_dataloader=iter(get_dataloader(eval_dataset_type, window_length=self.window_length, batch_size=8))
        all_logits=[]
        all_errors=[]
        for test_input, test_labels in test_dataloader:
            del test_labels
            (reconstruction,hidden_layer,reconstruction_loss, sparsity_loss, logits), total_loss=self(test_input, None)
            all_logits.append(logits)
            all_errors.append(logits-reconstruction)
        all_logits=torch.concat(all_logits, dim=0).flatten(end_dim=-2)
        all_errors=torch.concat(all_errors, dim=0).flatten(end_dim=-2)
        input_variance=torch.var(all_logits, dim=0).sum()
        error_variance=torch.var(all_errors, dim=0).sum()
        fraction_variance_unexplained=error_variance/input_variance
        return fraction_variance_unexplained





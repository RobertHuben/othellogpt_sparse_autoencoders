import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.dataloaders import get_dataloader
from tqdm import tqdm
from torcheval.metrics import BinaryAUROC

device='cuda' if torch.cuda.is_available() else 'cpu'


class SparseAutoencoder(nn.Module):

    def __init__(self, othello_gpt_model, layer_num, feature_ratio, sparsity_coeff=.1, initialization_weights=None, normalize_inputs=False, window_start_trim=0, window_end_trim=0):
        super().__init__()
        self.othello_gpt_model=othello_gpt_model
        self.window_length=self.othello_gpt_model.window_length
        self.layer_num=layer_num
        self.sparsity_coeff=sparsity_coeff
        self.hidden_layer_size=int(self.othello_gpt_model.d_model*feature_ratio)
        self.normalize_inputs=normalize_inputs
        if initialization_weights==None:
            initialization_weights=torch.normal(0, 1, (self.othello_gpt_model.d_model, self.hidden_layer_size))
        self.encoder_decoder_matrix=nn.Parameter(initialization_weights)
        self.encoder_bias=nn.Parameter(torch.normal(0,1, (self.hidden_layer_size,)))
        self.activation=nn.ReLU()
        self.layernorm=nn.LayerNorm((self.othello_gpt_model.d_model))
        self.window_start_trim=window_start_trim
        self.window_end_trim=window_end_trim
        self.write_updates_to=None

        for parameter in self.layernorm.parameters():
            parameter.requires_grad=False
        # freeze othello gpt model weights
        for parameter in self.othello_gpt_model.parameters():
            parameter.requires_grad=False


    def forward(self, input, labels):
        del labels # we're trained on unlabeled, but want a second argument to match other methods
        logits=self.othello_gpt_model.intermediate_residual_stream(input, layer_num=self.layer_num) #run the model
        trimmed_logits=logits[:, self.window_start_trim:(self.window_length-self.window_end_trim), :]
        if self.normalize_inputs:
            trimmed_logits=self.layernorm(trimmed_logits) #layernorm regularizes the input to self
        normalized_encoder_decoder_matrix=F.normalize(self.encoder_decoder_matrix, p=2, dim=1) #need to L2 regularize the matrix
        hidden_layer=self.activation(trimmed_logits@normalized_encoder_decoder_matrix + self.encoder_bias)
        reconstruction=hidden_layer@normalized_encoder_decoder_matrix.transpose(0,1)
        sparsity_loss=torch.norm(hidden_layer, p=1, dim=-1).sum()/hidden_layer.numel()
        reconstruction_loss=torch.norm(trimmed_logits-reconstruction, p=2, dim=-1).sum()/trimmed_logits.numel()
        total_loss=reconstruction_loss+self.sparsity_coeff*sparsity_loss
        return (reconstruction,hidden_layer,reconstruction_loss, sparsity_loss, trimmed_logits), total_loss
    
    def print_evaluation(self, train_loss, eval_dataset_type, step_number="N/A", details=False):
        del details
        reconstructions, hidden_layers, input_layers, total_losses=self.catenate_outputs_on_test_set(eval_dataset_type)
        test_loss=self.evaluate_test_losses(total_losses)
        percent_active=self.evaluate_sparsity(hidden_layers)
        percent_dead_neurons=self.evaluate_dead_neurons(hidden_layers)
        fraction_variance_unexplained=self.evaluate_variance_unexplaiend(reconstructions, input_layers)
        tqdm.write(f"Train loss and test (loss, features active, features dead, and unexplained variance) after {step_number} steps: {train_loss.item():.4f}, {test_loss:.4f}, {percent_active:.4%}, {percent_dead_neurons:.4%} {fraction_variance_unexplained:.4%}")
        if self.write_updates_to:
            with open(self.write_updates_to, 'a') as f:
                f.write(f"Train loss, test loss, test features active, and test FVU percent after {step_number} steps: {train_loss.item():.4f}, {test_loss.item():.4f}, {percent_active:.4%}, {fraction_variance_unexplained:.4%}\n")

    def catenate_outputs_on_test_set(self, eval_dataset_type):
        test_dataloader=iter(get_dataloader(eval_dataset_type, window_length=self.window_length, batch_size=8))
        reconstructions=[]
        hidden_layers=[]
        input_layers=[]
        total_losses=[]
        for test_input, test_labels in test_dataloader:
            del test_labels
            (reconstruction,hidden_layer,reconstruction_loss, sparsity_loss, input_layer), total_loss=self(test_input, None)
            reconstructions.append(reconstruction)
            hidden_layers.append(hidden_layer)
            input_layers.append(input_layer)
            total_losses.append(total_loss)
        reconstructions=torch.cat(reconstructions)
        hidden_layers=torch.cat(hidden_layers)
        input_layers=torch.cat(input_layers)
        total_losses=torch.stack(total_losses)
        return reconstructions, hidden_layers, input_layers, total_losses

    def evaluate_test_losses(self, total_losses):
        return total_losses.mean().item()
    
    def evaluate_sparsity(self, hidden_layer):
        sparsity=(hidden_layer>0).sum()/hidden_layer.numel()
        return sparsity
    
    def evaluate_dead_neurons(self, hidden_layer):
        hidden_layer_flattened=hidden_layer.flatten(end_dim=-2)
        dead_neurons=hidden_layer_flattened.max(dim=0).values==0
        fraction_dead=dead_neurons.float().mean()
        return fraction_dead

    def evaluate_variance_unexplaiend(self, reconstruction, input_layers):
        all_logits=input_layers.flatten(end_dim=-2)
        all_errors=(input_layers-reconstruction).flatten(end_dim=-2)
        input_variance=torch.var(all_logits, dim=0).sum()
        error_variance=torch.var(all_errors, dim=0).sum()
        fraction_variance_unexplained=error_variance/input_variance
        return fraction_variance_unexplained

    def evaluate_features_as_classifiers(self, eval_dataset_type):
        test_dataloader=iter(get_dataloader(eval_dataset_type, window_length=self.window_length, batch_size=10))
        activations=[]
        boards=[]
        for test_input, test_labels in tqdm(test_dataloader):
            (reconstruction,hidden_layer,reconstruction_loss, sparsity_loss, normalized_logits), total_loss=self(test_input, None)
            activations.append(hidden_layer)
            boards.append(test_labels)
        all_activations=torch.cat(activations).flatten(end_dim=-2)
        all_boards=torch.cat(boards).flatten(end_dim=-2)
        f1_scores=torch.zeros((all_activations.shape[1], all_boards.shape[1], 3))
        aurocs=torch.zeros((all_activations.shape[1], all_boards.shape[1], 3))
        for i, feature_activation in tqdm(enumerate(all_activations.transpose(0,1))):
            for j, board_position in enumerate(all_boards.transpose(0,1)):
                for k, piece_class in enumerate([0,1,2]):
                    is_feature_active=feature_activation>0
                    is_target_piece=board_position==piece_class
                    ended_game_mask= board_position>-100
                    tp=(is_feature_active*is_target_piece* ended_game_mask).sum()
                    fp=(is_feature_active* ~is_target_piece* ended_game_mask).sum()
                    tn=(~is_feature_active* ~is_target_piece* ended_game_mask).sum()
                    fn=(~is_feature_active*is_target_piece* ended_game_mask).sum()
                    f1_score=calculate_f1_score(tp, fp, tn, fn)
                    f1_scores[i,j,k]=float(f1_score)
                    metric = BinaryAUROC()
                    metric.update(feature_activation[ended_game_mask], is_target_piece[ended_game_mask].int())
                    aurocs[i,j,k]=float(metric.compute())
                    # print(f"F1 Score: {f1_score:.4f}")
        with open("analysis_results/aurocs.pkl", 'wb') as f:
            torch.save(aurocs, f)
        with open("analysis_results/f1_scores.pkl", 'wb') as f:
            torch.save(f1_scores, f)
        return

    

def calculate_f1_score(tp, fp, tn, fn):
    return 2*tp/(2*tp+fp+fn)


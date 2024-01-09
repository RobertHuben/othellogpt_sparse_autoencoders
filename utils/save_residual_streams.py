import torch
from utils.dataloaders import get_dataloader
from tqdm import tqdm

def save_residual_stream_from_dataloader(dataloader_mode, othello_gpt_model,layer_num, destination_file="save_residual_streams.pkl"):
    dataloader=iter(get_dataloader(dataloader_mode, window_length=othello_gpt_model.window_length, batch_size=1))
    residual_streams=[]
    labels=[]
    for input, label in tqdm(dataloader):
        logits=othello_gpt_model.intermediate_residual_stream(input, layer_num) #B-W-d_model
        labels.append(label)
        residual_streams.append(logits)
    residual_streams_as_tensor=torch.cat(residual_streams, dim=0)
    labels_as_tensor=torch.cat(labels, dim=0)
    with open(destination_file, 'wb') as f:
        torch.save((residual_streams_as_tensor, labels_as_tensor), f)
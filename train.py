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
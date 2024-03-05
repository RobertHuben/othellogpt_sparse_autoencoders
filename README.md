Code for training and evaluating a sparse autoencoder on OthelloGPT.

Contents:
1. **analysis.py** contains many methods for evaluating the sparse autoencoders (see below for details)
2. **autoencoder.py** architecture for a sparse autoencoder, as in [Cunningham et al]([url](https://arxiv.org/abs/2309.08600)https://arxiv.org/abs/2309.08600)
3. **linear_probes.py** architecture for a linear probe on the residual stream of a language model, for classifying board states
4. **model_training.py** control code for training OthelloGPT/probes/autoencoder (see below for details)
5. **othello_gpt.py** architecture for an OthelloGPT model, as in [Li et al]([url](https://arxiv.org/abs/2210.13382)https://arxiv.org/abs/2210.13382)
6. **requirements.txt** requirements. Install with ```pip install -r requirements.txt```
7. **train.py** code that executes a training run on OthelloGPT/probes/autoencoder
8. **utils/game_engine.py** plays Othello
9. **utils/generate_training_corpus.py** generates a corpus of Othello games, for training or assessment (see below for details)
10. Some other auxilary methods in utils/

How to use this code:

1. Download datasets and trained models from https://drive.google.com/drive/folders/1xMkEctaqAUjoPXGY-9dBu-pE3SJjKx2K
2. If you did not download the trained sparse autoencoder, or want to train your own, run model_training.py's ```full_sae_training()``` method. The OthelloGPT model must have an ```intermediate_residual_stream()``` method (see othello_gpt.py for details). 
3. If you did not download the cached AUROCs, or want to make new ones, run analysis.py's ```evaluate_all_probe_classification()```, ```evaluate_all_legal_moves_classification()```, and ```evaluate_all_content_classification()```.
4. To print the best AUROCs, run analysis.py's ```find_top_aurocs_legal()```or ```find_top_aurocs_contents```.
5. To create density plots of the features as classifiers for positions, run analysis.py's ```create_density_plot_contents(feature_number=N, board_position=M)``` or ```    create_density_plot_legal(feature_number=N, board_position=M)```. Feature numbers N run 0-1023, board positions M run 0-63. The values of N and M correspond to the outputs of ```find_top_aurocs_legal``` or ```find_top_aurocs_contents```.
6. To show the top- and random-activating board states, run analysis.py's ```show_top_activating(N, marked_position=M)```. N is again the feature number, and M is the board position that will be marked with a red circle.

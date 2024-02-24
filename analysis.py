import torch
import matplotlib.pyplot as plt
from utils.dataloaders import get_dataloader
import utils.dataloaders
from tqdm import tqdm
from utils.game_engine import history_to_legal_moves
import utils.game_engine
from torcheval.metrics import BinaryAUROC

device='cuda' if torch.cuda.is_available() else 'cpu'

def analyze_cosine_similarities(autoencoder_location="saes/sae.pkl"):
    '''
    cosine similarities between autoencoder features and linear probe directions
    '''
    probe_directions=get_probe_directions()
    autoencoder_directions=get_autoencoder_directions(autoencoder_location)
    cosine_similarities=probe_directions@autoencoder_directions.transpose(0,1)
    max_cosine_similarities_of_probes=cosine_similarities.max(dim=1).values
    plt.hist(max_cosine_similarities_of_probes)
    plt.show()

def analyze_cosine_similarities_2(autoencoder_location="saes/sae.pkl"):
    '''
    cosine similarities between autoencoder features and model unembed directions
    '''
    model_directions=get_model_unembed_directions()
    autoencoder_directions=get_autoencoder_directions(autoencoder_location)
    cosine_similarities=model_directions@autoencoder_directions.transpose(0,1)
    max_cosine_similarities_of_probes=cosine_similarities.max(dim=1).values
    plt.hist(max_cosine_similarities_of_probes)
    plt.show()

def get_random_directions(shape):
    directions=torch.normal(torch.zeros(shape), torch.ones(shape))
    directions=normalize_rows(directions)
    return directions

def get_probe_directions(probe_location="probes/probe_layer_6.pkl"):
    with open(probe_location, 'rb') as f:
        probe=torch.load(f, map_location=device)
    probe_directions=torch.concat([probe.classifier[i].weight.data for i in range(3)])
    probe_directions=normalize_rows(probe_directions)
    return probe_directions

def get_autoencoder_directions(autoencoder_location="saes/sae_layer_6.pkl"):
    with open(autoencoder_location, 'rb') as f:
        sae=torch.load(f, map_location=device)
    sae_directions=sae.encoder_decoder_matrix.data.transpose(0,1)
    sae_directions=normalize_rows(sae_directions)
    return sae_directions

def get_model_unembed_directions(model_location="trained_model_full.pkl"):
    with open(model_location, 'rb') as f:
        othello_gpt=torch.load(f, map_location=device)
    model_directions=othello_gpt.unembed.weight.data
    model_directions=normalize_rows(model_directions)
    return model_directions

def normalize_rows(data):
    return data/data.norm(p=2, dim=1, keepdim=True)

def calculate_f1_score(tp, fp, tn, fn):
    return 2*tp/(2*tp+fp+fn)

def evaluate_all_content_classification(activations_location="analysis_results/all_activations.pkl", board_locations="analysis_results/all_boards.pkl"):
    with open(activations_location, 'rb') as f:
        all_activations=torch.load(f)
    with open(board_locations, 'rb') as f:
        all_boards=torch.load(f)
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
    with open("analysis_results/contents_aurocs.pkl", 'wb') as f:
        torch.save(aurocs, f)
    with open("analysis_results/contents_f1_scores.pkl", 'wb') as f:
        torch.save(f1_scores, f)


def evaluate_all_legal_moves_classification(activations_location="analysis_results/all_activations.pkl", legal_moves_location="analysis_results/all_legal_moves.pkl"):
    with open(activations_location, 'rb') as f:
        all_activations=torch.load(f)
    with open(legal_moves_location, 'rb') as f:
        all_legal_moves=torch.load(f)
    f1_scores=torch.zeros((all_activations.shape[1], all_legal_moves.shape[1], 2))
    aurocs=torch.zeros((all_activations.shape[1], all_legal_moves.shape[1], 2))
    for i, feature_activation in tqdm(enumerate(all_activations.transpose(0,1))):
        for j, board_position in enumerate(all_legal_moves.transpose(0,1)):
            for k, legality_class in enumerate([0,1]):
                is_feature_active=feature_activation>0
                is_target_legality=board_position==legality_class
                ended_game_mask= board_position>-100
                tp=(is_feature_active*is_target_legality* ended_game_mask).sum()
                fp=(is_feature_active* ~is_target_legality* ended_game_mask).sum()
                tn=(~is_feature_active* ~is_target_legality* ended_game_mask).sum()
                fn=(~is_feature_active*is_target_legality* ended_game_mask).sum()
                f1_score=calculate_f1_score(tp, fp, tn, fn)
                f1_scores[i,j,k]=float(f1_score)
                metric = BinaryAUROC()
                metric.update(feature_activation[ended_game_mask], is_target_legality[ended_game_mask].int())
                aurocs[i,j,k]=float(metric.compute())
                # print(f"F1 Score: {f1_score:.4f}")
    with open("analysis_results/legal_aurocs.pkl", 'wb') as f:
        torch.save(aurocs, f)
    with open("analysis_results/legal_f1_scores.pkl", 'wb') as f:
        torch.save(f1_scores, f)

def plot_roc_curve(labels, scores):
    scores_sorted, order=torch.sort(scores, descending=True)
    labels_sorted=labels[order]
    points=[]
    tp=0
    fp=0
    previous_score=-1000000
    for score,label in zip(scores_sorted,labels_sorted):
        if score !=previous_score:
            points.append((tp,fp))
        previous_score=score
        if label:
            tp+=1
        else:
            fp+=1
    points.append((tp,fp))
    roc_y, roc_x=torch.tensor(points).transpose(0,1).float()
    roc_y/=tp
    roc_x/=fp
    plt.scatter(roc_x, roc_y)
    plt.show()

def create_roc(feature_number,board_position, piece_type):
    with open("analysis_results/all_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open("analysis_results/all_boards.pkl", 'rb') as f:
        all_boards=torch.load(f)
    scores=all_activations.transpose(0,1)[feature_number]
    labels=all_boards.transpose(0,1)[board_position]==piece_type
    plot_roc_curve(labels, scores)

def create_hist_contents(feature_number, board_position, separate_at_0=False):
    '''
    histogram for features as board classifiers
    '''
    with open("analysis_results/all_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open("analysis_results/all_boards.pkl", 'rb') as f:
        all_boards=torch.load(f)
    n_bins=10
    scores=all_activations.transpose(0,1)[feature_number]
    labels=all_boards.transpose(0,1)[board_position]
    enemy=scores[torch.where(labels==2)].detach().numpy()
    empty=scores[torch.where(labels==0)].detach().numpy()
    ally=scores[torch.where(labels==1)].detach().numpy()
    fig, ax = plt.subplots()
    ax.hist((enemy, empty, ally), n_bins, density=True, histtype='bar', stacked=True, label=["enemy", "empty", "ally"])
    ax.set_xlabel("Feature Activation")
    ax.set_ylabel("Frequency")
    ax.set_title(f'Feature {feature_number} activations against Position {board_position} contents')
    plt.legend()
    save_file_name=f"analysis_results/figures/hist_contents_feat_{feature_number}_pos_{board_position}.png"
    fig.savefig(save_file_name)

def create_hist_playables(feature_number, board_position, separate_at_0=False):
    '''
    histogram for features as playability classifiers
    '''
    with open("analysis_results/all_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open("analysis_results/all_legal_moves.pkl", 'rb') as f:
        all_legal_moves=torch.load(f)
    n_bins=10
    scores=all_activations.transpose(0,1)[feature_number]
    all_legal_moves=all_legal_moves.transpose(0,1)[board_position]
    legal=scores[torch.where(all_legal_moves==1)].detach().numpy()
    illegal=scores[torch.where(all_legal_moves==0)].detach().numpy()
    fig, ax = plt.subplots()
    ax.hist((legal, illegal), n_bins, density=True, histtype='bar', stacked=True, label=["legal", "illegal"])
    ax.set_xlabel("Feature Activation")
    ax.set_ylabel("Frequency")
    ax.set_title(f'Feature {feature_number} activations against whether Position {board_position} is a legal move')
    plt.legend()
    save_file_name=f"analysis_results/figures/hist_playable_feat_{feature_number}_pos_{board_position}.png"
    fig.savefig(save_file_name)

def save_probe_data(probe_location="probes/probe_layer_6.pkl"):
    with open(probe_location, 'rb') as f:
        probe=torch.load(f, map_location=device)
    probe.save_state_on_dataset()

def create_hist_for_probe(board_position, primary_class=0, separate_at_0=False):
    with open("analysis_results/probe_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open("analysis_results/probe_all_boards.pkl", 'rb') as f:
        all_boards=torch.load(f)
    n_bins=10
    scores=torch.nn.functional.softmax(all_activations[:, board_position, :].transpose(0,1), dim=0)[primary_class]
    labels=all_boards.transpose(0,1)[board_position]
    enemy=scores[torch.where(labels==2)].detach().numpy()
    empty=scores[torch.where(labels==0)].detach().numpy()
    ally=scores[torch.where(labels==1)].detach().numpy()
    fig, ax = plt.subplots()
    ax.hist((enemy, empty, ally), n_bins, density=True, histtype='bar', stacked=True, label=["enemy", "empty", "ally"])
    ax.set_xlabel("Feature Activation")
    ax.set_ylabel("Frequency")
    ax.set_title(f'Position {board_position} classification accuracy at Position {board_position}.')
    plt.legend()
    save_file_name=f"analysis_results/figures/hist_probe_pos_{board_position}.png"
    fig.savefig(save_file_name)

def find_top_aurocs_contents(k=10):
    with open("analysis_results/contents_aurocs.pkl", 'rb') as f:
        aurocs=torch.load(f)
    top_values, top_locations=torch.topk(aurocs.flatten(), k=k)
    top_locations_tuples=[location_to_triple(x, aurocs.shape) for x in top_locations]
    print("(Feature_number, board_position, class), AUROC:")
    print("\n".join([f"{top_locations_tuple}, {top_value:.4f}" for top_locations_tuple, top_value in zip(top_locations_tuples, top_values)]))
    return

def find_top_aurocs_legal(k=10):
    with open("analysis_results/legal_aurocs.pkl", 'rb') as f:
        aurocs=torch.load(f)
    top_values, top_locations=torch.topk(aurocs.flatten(), k=k)
    top_locations_tuples=[location_to_triple(x, aurocs.shape) for x in top_locations]
    print("(Feature_number, board_position, class), AUROC:")
    print("\n".join([f"{top_locations_tuple}, {top_value:.4f}" for top_locations_tuple, top_value in zip(top_locations_tuples, top_values)]))
    return

def location_to_triple(index, original_shape):
    c=int(index%original_shape[-1])
    b=int(index//original_shape[-1]%original_shape[-2])
    a=int(index//original_shape[-1]//original_shape[-2])
    return (a,b,c)

def show_top_activating(feature_number, top_k=5, marked_position=-1):
    with open("analysis_results/feature_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open("analysis_results/features_all_boards.pkl", 'rb') as f:
        all_boards=torch.load(f)
    num_data=all_boards.shape[0]
    these_activations=all_activations[:, feature_number]
    best_activations, best_indices=torch.topk(these_activations, k=top_k)
    # activations_and_boards=torch.cat((all_activations,all_boards), dim=1)
    # sorted_boards_and_activations=activations_and_boards[activations_and_boards[:, feature_number].argsort()]
    # sorted_boards=sorted_boards_and_activations[:,-64:]
    # sorted_activations=sorted_boards_and_activations[:,feature_number]
    # best_indices=torch.arange(start=num_data-top_k, end=num_data, dtype=int)
    random_indices=torch.randint(low=0, high=num_data, size=(top_k,))
    best_boards=all_boards[best_indices]
    # best_activations=sorted_activations[best_indices]
    random_boards=all_boards[random_indices]
    random_activations=these_activations[random_indices]

    fig, ax=plt.subplots(figsize=(10,5))
    for n in range(top_k):
        ax_1=plt.subplot(2, top_k, n+1)
        plot_board(ax_1, best_boards[n], best_activations[n], marked_position=marked_position)
        ax_2=plt.subplot(2, top_k, n+1+top_k)
        plot_board(ax_2, random_boards[n], random_activations[n], marked_position=marked_position)

    plt.savefig(f"analysis_results/figures/top_act_boards_feature_{feature_number}.png")
    plt.close()


def plot_board(ax, board, activation, marked_position=-1):

    black_pieces_locations=[]
    white_pieces_locations=[]
    for n in range(64):
        x=n%8
        y=n//8
        if board[n]==1:
            black_pieces_locations.append((x,y))
        elif board[n]==2:
            white_pieces_locations.append((x,y))
    
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_aspect(1)
    ax.set_axisbelow(True)

    marker_size=50

    if black_pieces_locations:
        ax.scatter(x=torch.tensor(black_pieces_locations)[:,0],y=torch.tensor(black_pieces_locations)[:,1], c="black", s=marker_size, linewidths=1.5, edgecolors="black")
    if white_pieces_locations:
        ax.scatter(x=torch.tensor(white_pieces_locations)[:,0],y=torch.tensor(white_pieces_locations)[:,1], c="white", s=marker_size, linewidths=1.5, edgecolors="black")
    
    if marked_position>=0:
        x_center=marked_position%8
        y_center=marked_position//8
        ax.scatter(x=[x_center], y=[y_center], facecolors="none", edgecolors="red", s=marker_size*2)
    
    ax.set_title(f"Activation {activation:.2f}")
    ax.invert_yaxis()

    ax.set_yticks(range(8), labels=range(1, 8+1))
    ax.set_xticks(range(8), labels=["A", "B", "C", "D", "E", "F", "G", "H"])
    return

def save_activations_boards_and_legal_moves(sae_location="saes/sae_layer_6.pkl", eval_dataset_type="probe_test"):
    torch.manual_seed(1)
    with open(sae_location, 'rb') as f:
        sae=torch.load(f, map_location=device)
    test_dataloader=iter(get_dataloader(eval_dataset_type, window_length=sae.window_length, batch_size=10))
    activations=[]
    boards=[]
    legal_moves=[]
    for test_input, test_labels in tqdm(test_dataloader):
        (reconstruction,hidden_layer,reconstruction_loss, sparsity_loss, normalized_logits), total_loss=sae(test_input, None)
        activations.append(hidden_layer)
        boards.append(sae.trim_to_window(test_labels)) # index n is the board state after before move n+start_window_length
        legal_moves.append(sae.trim_to_window(history_to_legal_moves(test_input))) #index in are the legal moves on turn n+1+start_window_length
    all_activations=torch.cat(activations).flatten(end_dim=-2) # shape (dw,f), where d= dataset size (2000), w=trimmed window length (52), f=num_features (1024)
    all_boards=torch.cat(boards).flatten(end_dim=-2) # shape (dw,b), where d= dataset size (2000), w=trimmed window length (52), b=board_size(64)
    all_legal_moves=torch.cat(legal_moves).flatten(end_dim=-2) # shape (dw,b), where d= dataset size (2000), w=trimmed window length (52), b=board_size(64)
    with open("analysis_results/all_activations.pkl", 'wb') as f:
        torch.save(all_activations, f)
    with open("analysis_results/all_boards.pkl", 'wb') as f:
        torch.save(all_boards, f)
    with open("analysis_results/all_legal_moves.pkl", 'wb') as f:
        torch.save(all_legal_moves, f)

def combine(legal_moves, board):
    return (board+10*legal_moves).reshape((8,8))

# def combine(legal_moves, board):
#     return (board.reshape(8,8).transpose(0,1)+10*legal_moves.reshape((8,8)))

# def save_feature_activations():
#     with open("saes/sae_layer_6_trimmed_alpha_77_e-3.pkl", 'rb') as f:
#         sae=torch.load(f, map_location=device)
#     sae.save_state_on_dataset()

# def compare_stuff():
#     with open("analysis_results/all_activations.pkl", 'rb') as f:
#         all_activations=torch.load(f)
#     with open("analysis_results/feature_activations.pkl", 'rb') as f:
#         feature_activations=torch.load(f)

#     return


def manual_test():
    with open("saes/sae_layer_6_trimmed_alpha_77_e-3.pkl", 'rb') as f:
        sae=torch.load(f, map_location=device)
    
    moves="D3 E3 F3 E2 F1 C2 F4 G3 B1 C6 H3 D2 C4 C3 F5 C5 E6 G4 B5 B2 A2 B6 G5 E1 C1 F6 F2 E7 B7 H4 E8 D1 G1 G2 F7 A1 H2 C7 B3 F8 D8 H1 G7 D7 H5 D6 C8 A4 A5 H8 G6 H6 A3 A7 A6 B4 A8 G8 B8"
    # moves="D3 E3 F3 E2 F2 C6 E6 F5 E1 F7 G6 G2 G4 H5 F6 D1 H6 H7 C1 F1 F8 C5 D6 D7 G5 F4 B4 B5 D8 B3 E7 B1 H3 C2 A5 A4 B6 C8 H1 H4 G1 G3 C4 A7 E8 B7 C7 D2 H2 G7 C3 A3 H8 A6 B2 G8 A1"
    dataloader=utils.dataloaders.OthelloDataset(None)
    input= torch.tensor(dataloader.string_to_int_list(moves)).unsqueeze(dim=0)
    (reconstruction,hidden_layer,reconstruction_loss, sparsity_loss, trimmed_logits), total_loss= sae(input, None)
    feature_number=111
    specific_hidden_layer=hidden_layer[0, 0, feature_number]
    print(specific_hidden_layer)
    return

if __name__=="__main__":
    autoencoder_location="saes/sae_layer_6_trimmed_alpha_77_e-3.pkl"

    # evaluate_all_content_classification()
    # evaluate_all_legal_moves_classification()

    save_activations_boards_and_legal_moves(sae_location="saes/sae_layer_6_trimmed_alpha_77_e-3.pkl")
    # for n in range(1024):
    #     show_top_activating(n)

    # for a in range(1024):
    #     for b in range (64):
    #         create_hist(a,b)
    # get_autoencoder_directions()
    # analyze_cosine_similarities(autoencoder_location=autoencoder_location)
    # analyze_cosine_similarities_2(autoencoder_location=autoencoder_location)
    # find_top_aurocs_legal(k=10)
    # create_hist_playables(111,43)
    # show_top_activating(111, marked_position=29)
    # show_top_activating(142, marked_position=43)

    # manual_test()

    # find_top_aurocs_contents(k=100)

    # save_probe_data()
    # create_hist_for_probe(board_position=20)
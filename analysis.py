import torch
import matplotlib.pyplot as plt

device='cuda' if torch.cuda.is_available() else 'cpu'

def analyze_cosine_similarities():
    probe_directions=get_probe_directions()
    # autoencoder_directions=get_autoencoder_directions("saes/sae_layer_6_alpha_20.pkl")
    autoencoder_directions=get_autoencoder_directions("saes/sae_layer_6_alpha_1e-2.pkl")
    cosine_similarities=probe_directions@autoencoder_directions.transpose(0,1)
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

def normalize_rows(data):
    return data/data.norm(p=2, dim=1, keepdim=True)

def calculate_f1_score(tp, fp, tn, fn):
    return 2*tp/(2*tp+fp+fn)

def evaluate_all_classification(autoencoder_location="saes/sae_layer_6_alpha_10.pkl"):
    with open(autoencoder_location, 'rb') as f:
        sae=torch.load(f, map_location=device)
    sae.evaluate_features_as_classifiers("probe_test")

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

def create_hist(feature_number, board_position, separate_at_0=False):
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
    save_file_name=f"analysis_results/figures/hist_feat_{feature_number}_pos_{board_position}.png"
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

def find_top_aurocs(k=10):
    with open("analysis_results/aurocs.pkl", 'rb') as f:
        aurocs=torch.load(f)
    top_values, top_locations=torch.topk(aurocs.flatten(), k=k)
    top_locations_tuples=[location_to_triple(x) for x in top_locations]
    print("\n".join([f"{top_locations_tuple}, {top_value:.4f}" for top_locations_tuple, top_value in zip(top_locations_tuples, top_values)]))
    return

def location_to_triple(index):
    c=int(index%3)
    b=int(index//3%64)
    a=int(index//3//64)
    return (a,b,c)

if __name__=="__main__":
    # for a in range(1024):
    #     for b in range (64):
    #         create_hist(a,b)
    # get_autoencoder_directions()
    analyze_cosine_similarities()
    # evaluate_all_classification()
    # create_roc(467,34,0)
    # create_hist(707, 56)
    # find_top_aurocs(k=100)

    # save_probe_data()
    # create_hist_for_probe(board_position=20)
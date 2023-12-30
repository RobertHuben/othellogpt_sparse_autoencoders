from random import seed
from game_engine import generate_random_game, move_log_to_string
import time


def generate_training_corpus(num_games=100, random_seed=42, to_write_location="training_corpus.txt", write_mode='w'):
    seed(random_seed)
    games_as_list=[move_log_to_string(generate_random_game(), insert_terminal_XX=True) for _ in range(num_games)]
    games_as_string="\n".join(games_as_list)
    with open(to_write_location, write_mode) as f:
        if write_mode=="a":
            f.write("\n")
        f.write(games_as_string)
    return

if __name__=="__main__":
    for n in range(80):
        start=time.time()
        k=1000
        generate_training_corpus(k, write_mode='a')
        end=time.time()
        print(f"Time for {k}: {end-start} seconds")
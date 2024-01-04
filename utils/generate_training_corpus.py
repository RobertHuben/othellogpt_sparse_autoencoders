from game_engine import generate_random_game, move_log_to_string
import time

def generate_training_corpus(num_games=100, to_write_location="datasets/training_corpus.txt", write_mode='w'):
    games_as_list=[move_log_to_string(generate_random_game(), insert_terminal_XX=True) for _ in range(num_games)]
    games_as_string="\n".join(games_as_list)
    with open(to_write_location, write_mode) as f:
        if write_mode=="a":
            f.write("\n")
        f.write(games_as_string)
    return

if __name__=="__main__":
    num_batches=90
    batch_size=1000
    write_location="datasets/sae_training_corpus.txt"
    for n in range(num_batches):
        start=time.time()
        generate_training_corpus(batch_size, write_mode='a', to_write_location=write_location)
        end=time.time()
        print(f"Time for {batch_size}: {end-start} seconds")
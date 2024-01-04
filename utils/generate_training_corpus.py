from game_engine import generate_random_game, move_log_to_string, history_to_board_states
import time

def generate_move_log_corpus(num_games=100, to_write_location="datasets/test_file.txt", write_mode='w'):
    games_as_list=[move_log_to_string(generate_random_game(), insert_terminal_XX=True) for _ in range(num_games)]
    games_as_string="\n".join(games_as_list)
    with open(to_write_location, write_mode) as f:
        if write_mode=="a":
            f.write("\n")
        f.write(games_as_string)
    return

def generate_move_log_and_board_state_corpus(num_games=100, to_write_location="datasets/test_file.txt", write_mode='w'):
    games=[generate_random_game() for _ in range(num_games)]
    games_as_list=[move_log_to_string(game, insert_terminal_XX=True)+"/"+history_to_board_states(game) for game in games]
    games_as_string="\n".join(games_as_list)
    with open(to_write_location, write_mode) as f:
        if write_mode=="a":
            f.write("\n")
        f.write(games_as_string)
    return

if __name__=="__main__":
    num_batches=10
    batch_size=100
    write_location="datasets/board_state_classifier_test_corpus.txt"
    for n in range(num_batches):
        start=time.time()
        # generate_move_log_corpus(batch_size, write_mode='a', to_write_location=write_location)
        generate_move_log_and_board_state_corpus(batch_size, write_mode='a', to_write_location=write_location)
        end=time.time()
        print(f"Time for {batch_size}: {end-start} seconds")
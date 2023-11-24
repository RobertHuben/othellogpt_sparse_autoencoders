import numpy as np
from random import choice

class OthelloGame:

    def __init__(self, board_size=8):
        self.board_size=board_size
        self.board=np.zeros((board_size,board_size))
        self.board[3,3]=-1
        self.board[3,4]=1
        self.board[4,3]=1
        self.board[4,4]=-1
        self.turns_history=[]
        self.active_player=1

    def make_move(self, coordinates):
        '''
        adjusts the board state as if a move was made at coordinates, returning True if successful
        if the move is illegal, returns False
        inputs:
            coordinates: a 2-ple of positions, from {0,1,...,7}^2
        '''
        pieces_to_flip=self.pieces_to_flip_with_move_at(coordinates)
        if pieces_to_flip:
            for location in pieces_to_flip:
                self.board[tuple(location)]*=-1
            self.board[coordinates]=self.active_player
            self.turns_history.append(coordinates)
            self.active_player*=-1
            return True
        else:
            return False

    def coordinates_on_board(self, coordinates):
        '''
        checks if the listed coordinates are on the board
        '''
        return coordinates[0]>=0 and coordinates[1]>=0 and coordinates[0]<self.board_size and coordinates[1]<self.board_size
                             

    def get_directions(self):
        '''
        returns a list of the eight directions
        '''
        all_directions=[np.array([x,y]) for x in [-1,0,1] for y in [-1,0,1]]
        all_directions.pop(4) # this is the (0,0) direction so should be removed
        return all_directions

    def pieces_to_flip_with_move_at(self, coordinates):
        '''
        returns a list of the pieces that would be flipped if the active player placed a piece at coordinates
        returns the empty list if the move is invalid
        '''
        if self.board[coordinates]!=0:
            return []
        pieces_to_flip=[]
        directions=self.get_directions()
        for direction in directions:
            enemy_pieces_in_this_direction=[]
            next_coordinates=coordinates+direction
            while self.coordinates_on_board(next_coordinates):
                if self.board[tuple(next_coordinates)]*self.active_player==-1:
                    enemy_pieces_in_this_direction.append(next_coordinates)
                elif self.board[tuple(next_coordinates)]*self.active_player==1:
                    pieces_to_flip.extend(enemy_pieces_in_this_direction)
                elif self.board[tuple(next_coordinates)]==0:
                    break
                next_coordinates=next_coordinates+direction

    def is_legal_move(self, coordinates):
        return bool(self.pieces_to_flip_with_move_at(coordinates))


    def list_legal_moves(self):
        legal_moves=[]
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.is_legal_move((x,y)):
                    legal_moves.append((x,y))
        return legal_moves

    def print_board(self):
        piece_markers={0:".", 1:"O", -1:"*"}
        top_row=" ABCDEFGH"
        to_print_lines=[top_row]
        for i, row in enumerate(self.board.transpose()):
            this_row_as_text=f"{i+1}{''.join([piece_markers[piece] for piece in row])}"
            to_print_lines.append(this_row_as_text)
        to_print="\n".join(to_print_lines)
        print(to_print)

def generate_random_game():
    '''
    returns a list of game moves chosen at random from the legal moves
    '''
    game=OthelloGame()
    legal_moves=game.list_legal_moves()
    while legal_moves:
        next_move=choice(legal_moves)
        game.make_move(next_move)
        legal_moves=game.list_legal_moves()
    return game.turns_history

def play_back_move_log(move_log):
    game=OthelloGame()
    for move in move_log:
        game.make_move(move)
        game.print_board()

random_move_log=generate_random_game()
play_back_move_log(random_move_log)
import math

from copy import deepcopy
from games4e import *

class EinStein(Game):

    def __init__(self):
        self.initial = GameState(to_move='R', utility=0, board={'R': (0, 0), 'B': (2, 2)}, moves=[(1, 1), (1, 0), (0, 1)])

    def compute_moves(self, board, to_move):
        moves = []
        if board[to_move]:
            if to_move == 'R':
                if board[to_move][0] < 2:
                    moves.append((board[to_move][0] + 1, board[to_move][1]))
                    if board[to_move][1] < 2:
                        moves.append((board[to_move][0] + 1, board[to_move][1] + 1))
                if board[to_move][1] < 2:
                    moves.append((board[to_move][0], board[to_move][1] + 1))
            if to_move == 'B':
                if board[to_move][0] > 0:
                    moves.append((board[to_move][0] - 1, board[to_move][1]))
                    if board[to_move][1] > 0:
                        moves.append((board[to_move][0] - 1, board[to_move][1] - 1))
                if board[to_move][1] > 0:
                    moves.append((board[to_move][0], board[to_move][1] - 1))
        return moves

    def display(self, state):
        displayed_board = [[' ' for _ in range(3)] for _ in range(3)]
        for player_i in ['R', 'B']:
            if state.board[player_i] is not None:
                displayed_board[state.board[player_i][0]][state.board[player_i][1]] = f'{player_i}'
        print('\n'.join(['|' + '|'.join(row) + '|' for row in displayed_board]), end='\n\n')

    def terminal_test(self, state):
        return state.utility != 0

    def actions(self, state):
        return state.moves

    def result(self, state, move):
        # Task 1.1
        # Return a state resulting from the move.
        # Replace the line below with your code
        
        # determine the player and the opponent in the state 
        player = state.to_move
        if player == 'R':
            opponent = 'B'
        else:
            opponent = 'R'

        #make a deep copy of the board 
        new_board = deepcopy(state.board)

        #executes the move on the board 
        new_board[player] = move 
        
        #checks for capture 
        if new_board[opponent] == new_board[player]: 
            new_board[opponent] = None  #captures opponents piece

        # compute moves and utility
        new_utility = self.compute_utility(new_board)
        new_moves = self.compute_moves(new_board, opponent)

        
        #return new game state
        return GameState(to_move = opponent,
                         utility = new_utility,
                         board = new_board,
                         moves = new_moves)

    def utility(self, state, player):
        # Task 1.2
        # Return the state's utility to the player.
        # Replace the line below with your code.
        return state.utility if player == 'R' else -state.utility

    def compute_utility(self, board):
        # Task 1.3
        # Return the utility of the board.
        # Replace the line below with your code.
        ###################### ASSUMING BOARD IS ALWAYS A 3X3 ###########
        #blue win = -1, red win = 1 otherwise 0 
        #get pos for both 
        red_pos = board.get('R')
        blue_pos = board.get('B')

        #Create the board size always a 3x3 square 
        board = 3

        #red win 
        if red_pos ==(board - 1, board-1):
            return 1
        if blue_pos is None: 
            return 1
        

        #Blue win 
        if blue_pos ==(0,0):
            return -1
        if red_pos is None: 
            return -1
        
        return 0



class MehrSteine(StochasticGame):

    def __init__(self, board_size):
        self.board_size = board_size
        self.num_piece = int((board_size - 1) * (board_size - 2) / 2)
        board = {'R': [], 'B': []}
        for i in range(board_size - 2):
            for j in range(board_size - 2 - i):
                board['R'].append((i, j))
                board['B'].append((board_size - 1 - i, board_size - 1 - j))
        self.initial = StochasticGameState(to_move='R', utility=0, board=board, moves=None, chance=None)

    def compute_moves(self, board, to_move, index):
        moves = []
        coordinates = board[to_move][index]
        if to_move == 'R':
            if coordinates[0] < self.board_size - 1:
                moves.append((index, (coordinates[0] + 1, coordinates[1])))
                if coordinates[1] < self.board_size - 1:
                    moves.append((index, (coordinates[0] + 1, coordinates[1] + 1)))
            if coordinates[1] < self.board_size - 1:
                moves.append((index, (coordinates[0], coordinates[1] + 1)))
        if to_move == 'B':
            if coordinates[0] > 0:
                moves.append((index, (coordinates[0] - 1, coordinates[1])))
                if coordinates[1] > 0:
                    moves.append((index, (coordinates[0] - 1, coordinates[1] - 1)))
            if coordinates[1] > 0:
                moves.append((index, (coordinates[0], coordinates[1] - 1)))
        return moves

    def display(self, state):
        spacing = 1 if self.num_piece == 1 else math.floor(math.log(self.num_piece - 1, 10)) + 1
        displayed_board = [[' ' * (spacing + 1) for _ in range(self.board_size)] for _ in range(self.board_size)]
        for player_i in ['R', 'B']:
            for piece_i in range(self.num_piece):
                if state.board[player_i][piece_i] is not None:
                    displayed_board[state.board[player_i][piece_i][0]][state.board[player_i][piece_i][1]] = player_i + str(piece_i).rjust(spacing)
        print('\n'.join(['|' + '|'.join(row) + '|' for row in displayed_board]), end='\n\n')

    def terminal_test(self, state):
        return state.utility != 0

    def actions(self, state):
        return state.moves


    def result(self, state, move):
        # Task 2.1
        # Return a state resulting from the move.
        # Replace the line below with your code.
        # determine the player and the opponent in the state 
        player = state.to_move
        if player == 'R':
            opponent = 'B'
        else:
            opponent = 'R'

        #create a deep copy
        new_board = deepcopy(state.board)

        
        #unpack the move tuple
        index, (new_x, new_y) = move
        # excute the move
        new_board[player][index] = (new_x, new_y)

        #check for captures 
        opponent_index = 0
        for opponenet_pos in new_board[opponent]:
            if opponenet_pos == (new_x, new_y):
                #captures the piece 
                new_board[opponent][opponent_index] = None  
            opponent_index += 1

        # check to capture your own piece
        player_index = 0
        for player_pos in new_board[player]:
            if player_pos == (new_x, new_y) and player_index != index:
                new_board[player][player_index] = new_board[player][index] 
            player_index += 1 

        #return new game state
        new_utility = self.compute_utility(new_board)

        return StochasticGameState(to_move=opponent, utility=new_utility, board=new_board, moves=None, chance=None)



    def utility(self, state, player):
        # Task 2.2
        # Return the state's utility to the player.
        # Replace the line below with your code.
        return state.utility if player == 'R' else -state.utility

    def compute_utility(self, board):
        # Task 2.3
        # Return the utility of the board.
        # Replace the line below with your code.
        #get pos of blue and red 
        red_pos = board.get('R', [])
        blue_pos = board.get('B', [])

        #RED WINS check for red inbottom right, blue has no pieces 
        if any(pos == (self.board_size -1, self.board_size - 1) for pos in red_pos):
            return 1
        if all(pos is None for pos in blue_pos):
            return 1
        
        #BLUE WINS check blue in top left, red has no pieces 
        if any(pos== (0,0) for pos in blue_pos):
            return -1
        if all (pos is None for pos in red_pos):
            return -1
        
        #all other cases return 0
        return 0


    def chances(self, state):
        # Task 2.4
        # Return a list of possible chance outcomes.
        # Replace the line below with your code.
        possible_rolls = list(range(self.num_piece))
        return possible_rolls

    def outcome(self, state, chance):
        # Task 2.5
        # Return a state resulting from the chance outcome.
        # Replace the line below with your code.
        #who's turn is it 
        player = state.to_move 
        if player == 'R':
            opponent = 'B'
        else:
            opponent ='R'
        
        #get the dice roll, chance 
        piece_index = chance
        #find the piece to move 
        ##################should this be 1 or 0 ################
        # init everything that is needed
        lower_index = piece_index - 1 
        lower_moves = []
        higher_index = piece_index + 1
        higher_moves = []

        new_moves = []

	#check if the piece is not on the board
        if state.board[player][piece_index] is None:
            while lower_index >= 0:
            #check one lower and one higher, continue untill piece is found, if both found then concatinate both into one list of moves. 
                #checks lower piece
                if lower_index >= 0 and state.board[player][lower_index] is not None: 
                    lower_moves = self.compute_moves(state.board, player, lower_index)
                    #print(" lower loop")
                    break
                lower_index -= 1
                      
            #runs until finds a higher piece, or runs out of higher pieces. 
            while higher_index <= self.num_piece -1:
                #checks for the higher piece. 
                if higher_index < self.num_piece and state.board[player][higher_index] is not None: 
                    higher_moves = self.compute_moves(state.board, player, higher_index)
                    #print("higher loop")
                    break
                higher_index += 1

            #combines the higher and lower moves. 
            new_moves = higher_moves + lower_moves     
        
        #else just runs it with the choosen piece
        else: 
            
            new_moves = self.compute_moves(state.board, player, piece_index)

        #return the new updated moves and state
        return StochasticGameState(
                    to_move = player,
                    utility = state.utility, 
                    board = state.board,
                    moves = new_moves,
                    chance = chance
                )

    def probability(self, chance):
        # Task 2.6
        # Return the probability of a chance outcome.
        # Replace the line below with your code.
        return 1/self.num_piece

def stochastic_monte_carlo_tree_search(state, game, playout_policy, N=1000):

    def select(n):
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        if not n.children and not game.terminal_test(n.state):
            n.children = {MCT_Node(state=game.outcome(game.result(n.state, action), chance), parent=n): action for action in game.actions(n.state) for chance in game.chances(game.result(n.state, action))}
        return select(n)

    def simulate(game, state):
        player = game.to_move(state)
        while not game.terminal_test(state):
            action = playout_policy(game, state)
            state = game.result(state, action)
            chance = random.choice(game.chances(state))
            state = game.outcome(state, chance)
        v = game.utility(state, player)
        return -v

    def backprop(n, utility):
        if utility > 0:
            n.U += utility
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state=state)

    for _ in range(N):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(game, child.state)
        backprop(child, result)

    max_state = max(root.children, key=lambda p: p.N)

    return root.children.get(max_state)

def schwarz_score(game, state):
    schwarz = {}
    valid_pieces = [piece_i for piece_i in range(game.num_piece) if state.board['R'][piece_i] is not None]
    if len(valid_pieces) == 0:
        schwarz['R'] = (game.board_size - 1) * game.num_piece
    elif len(valid_pieces) == 1:
        schwarz['R'] = game.board_size - 1 - min(state.board['R'][valid_pieces[0]])
    else:
        schwarz_per_piece = []
        for index_i, piece_i in enumerate(valid_pieces):
            if index_i == 0:
                schwarz_per_piece.append((game.board_size - 1 - min(state.board['R'][piece_i])) * game.num_piece / valid_pieces[1])
            elif index_i == len(valid_pieces) - 1:
                schwarz_per_piece.append((game.board_size - 1 - min(state.board['R'][piece_i])) * game.num_piece / (game.num_piece - valid_pieces[-2] - 1))
            else:
                schwarz_per_piece.append((game.board_size - 1 - min(state.board['R'][piece_i])) * game.num_piece / (valid_pieces[index_i + 1] - valid_pieces[index_i - 1] - 1))
        schwarz['R'] = min(schwarz_per_piece)
    valid_pieces = [piece_i for piece_i in range(game.num_piece) if state.board['B'][piece_i] is not None]
    if len(valid_pieces) == 0:
        schwarz['B'] = (game.board_size - 1) * game.num_piece
    elif len(valid_pieces) == 1:
        schwarz['B'] = max(state.board['B'][valid_pieces[0]])
    else:
        schwarz_per_piece = []
        for index_i, piece_i in enumerate(valid_pieces):
            if index_i == 0:
                schwarz_per_piece.append(max(state.board['B'][piece_i]) * game.num_piece / valid_pieces[1])
            elif index_i == len(valid_pieces) - 1:
                schwarz_per_piece.append(max(state.board['B'][piece_i]) * game.num_piece / (game.num_piece - valid_pieces[-2] - 1))
            else:
                schwarz_per_piece.append(max(state.board['B'][piece_i]) * game.num_piece / (valid_pieces[index_i + 1] - valid_pieces[index_i - 1] - 1))
        schwarz['B'] = min(schwarz_per_piece)
    return schwarz

def schwarz_diff_to_weight(diff, max_schwarz):
    # Task 3
    # Return a weight value based on the relative difference in Schwarz scores.
    # Replace the line below with your code.
    #calculates the correct score.
    score = diff/max_schwarz

    #checks all the diff/max_schwarz values then returns corrisponding weight
    if score < -0.5:
        return 1
    elif score < -0.375:
        return 2
    elif score < -0.25:
        return 4
    elif score < -0.125:
        return 8
    elif score < 0:
        return 16
    elif score < 0.125:
        return 32
    elif score < 0.25:
        return 64
    elif score < 0.375:
        return 128
    elif score < 0.5:
        return 256
    else:
        return 512
        






def random_policy(game, state):
    return random.choice(list(game.actions(state)))

def schwarz_policy(game, state):
    actions = list(game.actions(state))
    to_move = state.to_move
    opponent = 'B' if to_move == 'R' else 'R'
    weights = []
    for action in actions:
        state_prime = game.result(state, action)
        schwarz = schwarz_score(game, state_prime)
        schwarz_diff = schwarz[opponent] - schwarz[to_move]
        weights.append(schwarz_diff_to_weight(schwarz_diff, (game.board_size - 1) * game.num_piece))
    return random.choices(actions, weights=weights)[0]

def random_mcts_player(game, state):
    return stochastic_monte_carlo_tree_search(state, game, random_policy, 100)

def schwarz_mcts_player(game, state):
    return stochastic_monte_carlo_tree_search(state, game, schwarz_policy, 100)

if __name__ == '__main__':

    # Task 1 test code
    
    num_win = 0
    num_loss = 0
    for _ in range(50):
        if EinStein().play_game(alpha_beta_player, random_player) == 1:
            num_win += 1
        else:
            num_loss += 1
            
    for _ in range(50):
        if EinStein().play_game(random_player, alpha_beta_player) == 1:
            num_loss += 1
        else:
            num_win += 1
    print(f'alpha-beta pruned minimax player vs. random-move player: {num_win} wins and {num_loss} losses', end='\n\n')
    

    # Task 2 test code

    num_win = 0
    num_loss = 0
    for _ in range(50):
        if MehrSteine(4).play_game(random_mcts_player, random_player) == 1:
            num_win += 1
        else:
            num_loss += 1
    for _ in range(50):
        if MehrSteine(4).play_game(random_player, random_mcts_player) == 1:
            num_loss += 1
        else:
            num_win += 1
    print(f'MCTS with random playout vs. random-move player: {num_win} wins and {num_loss} losses', end='\n\n')
    

    # Task 3 test code
    
    num_win = 0
    num_loss = 0
    for _ in range(500):
        if MehrSteine(4).play_game(schwarz_mcts_player, random_mcts_player) == 1:
            num_win += 1
        else:
            num_loss += 1
    for _ in range(500):
        if MehrSteine(4).play_game(random_mcts_player, schwarz_mcts_player) == 1:
            num_loss += 1
        else:
            num_win += 1
    print(f'MCTS with Schwarz-based playout vs. MCTS with random playout: {num_win} wins and {num_loss} losses', end='\n\n')
    

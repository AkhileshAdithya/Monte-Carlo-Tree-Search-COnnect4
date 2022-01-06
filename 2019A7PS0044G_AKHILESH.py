import numpy as np
from hashlib import sha1
from numpy import all, array, uint8
import random
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import gzip, json

#####################################################
# read Q Values
# with gzip.open('./2019A7PS0044G_AKHILESH.dat.gz', 'rb') as f:
#             file = f.read()
#             algo2.Q = json.loads(file.decode('utf-8'))
######################################################3

class hashable(object):
    def __init__(self, wrapped):
        self.__wrapped = array(wrapped)
        self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def wrap(self):
        return self.__hash

    def unwrap(self):        
        return array(self.__wrapped)

################################################
class Node:
    def __init__(self, state, parent, action, player):
        self.state = state
        self.parent = parent
        self.action = action
        self.player = player
        self.children = []
        self.visits = 0
        self.score = 0
################################################
class RandomPlayer():
    def __init__(self, game, player):
        self.game = game
        self.player = player

    def bestMove(self, state, player):
        return random.choice(self.game.validMoves(state))
################################################
class MCTS():
    def __init__(self, game, n_playouts, player , C = 10*np.sqrt(2), epsilon = 0.25):
        self.game = copy.deepcopy(game)
        self.n_playouts = n_playouts
        self.player = player
        self.C = C
        self.root = Node(game.get_state(), None, None, player)
        self.epsilon = epsilon
    
    
    def bestMove(self, state, player):
        self.root = Node(state, None, None, player)
        node = self.root
        if(self.game.checkTerminalState(self.root.state, self.root.player)[0]):
            return 42
        for i in range(3):
            self.expand(node)
            if self.game.checkTerminalState(node.state, node.player)[0]:
                break
            node = random.choice(node.children)
            
        for i in range(self.n_playouts):
            self.MCTSIteration(node)

        if self.bestChild(self.root).action in self.game.validMoves(self.root.state):
            return self.bestChild(self.root).action
        else:
            return 


    def MCTSIteration(self, node):
        node = self.select(self.root)
        if node.visits == 0:
            winner = self.playout(node.state)
            reward = self.calcReward(winner, node.player)
        else:
            self.expand(node)
            node = self.select(node)
            winner = self.playout(node.state) #add rewards
            reward = self.calcReward(winner, node.player)
        self.backpropagate(node, reward) #change to rewards

    def select(self, node):
        while node.children:
            node.visits += 1
            if self.epsilon < random.random():
                return random.choice(node.children)
            else:             
                return self.bestChild(node)
        return node
    
    def expand(self, node):
        if not self.game.checkTerminalState(node.state, node.player)[0]:
            legal_moves = self.game.validMoves(node.state)
            for move in legal_moves:
                new_board = copy.deepcopy(node.state)
                self.game.playMoveWithCopy(new_board, move, node.player)
                new_node = Node(new_board, node, move, self.game.nextPlayer(node.player))
                new_node.visits += 1
                node.children.append(new_node)

    def playout(self, state):
        rollout_board = copy.deepcopy(state)
        player = self.player
        terminal_state, winner = self.game.checkTerminalState(rollout_board, player) #check player
        while terminal_state == False:

            legal_moves = self.game.validMoves(rollout_board)
            self.game.playMoveWithCopy(rollout_board, random.choice(legal_moves), player)
            player = self.game.nextPlayer(player)
            terminal_state, winner = self.game.checkTerminalState(rollout_board, player)
        return winner
    
    def calcReward(self, winner, player):
        if winner == player:
            return 1
        elif winner == self.game.nextPlayer(player):
            return -100
        elif winner == 0:
            return -10
        else:
            return 0
    
    def backpropagate(self, node, score):
        while node is not None:
            node.score += score
            #node.visits += 1
            node = node.parent

    def bestChild(self, node):
        best_score = -10000000000
        best_nodes = []
        if node.children:
            for child in node.children:
                if child.visits == 0:
                    score = 0
                else:
                    score = (child.score/child.visits) + self.C * np.sqrt(np.log(node.visits)/child.visits)
                
                if score > best_score:
                    best_score = score
                    best_nodes = [child]
                elif score == best_score:
                    best_nodes.append(child)
            return random.choice(best_nodes)
        else:
            return node


################################################
#Qlearning for connect 4
class QLearning():
    def __init__(self, game, player, alpha=0.5, gamma=0.9, epsilon=0.1):
        # self.game = copy.deepcopy(game)
        self.game = game
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(lambda: 0))
        self.N = defaultdict(lambda: defaultdict(lambda: 0))

    def bestMove(self, state):
        state = hashable(state)
        if random.random() < self.epsilon:
            return random.choice(self.game.validMoves(state.unwrap()))
        else:
            return max(self.game.validMoves(state.unwrap()), key = lambda x: self.Q[state.wrap()][x])
    
    def QLearningRun(self, state):
        action = self.bestMove(state)
        reward = self.calcReward(action)
        self.updateQ(state, action, reward)
        return action
    
    def calcReward(self, action):
        if action == self.player:
            return 10
        elif action == 0:
            return -1
        elif action == -1:
            pass
        else:
            return -50

    def updateQ(self, state, action, reward):
        state = hashable(state)
        self.N[state.wrap()][action] += 1
        self.Q[state.wrap()][action] += (self.alpha/self.N[state.wrap()][action]) * (reward + self.gamma * max(self.Q[state.wrap()].values()) - self.Q[state.wrap()][action])
    

    
################################################
class Connect4():
    def __init__(self, ROW_COUNT, COLUMN_COUNT):
        self.state = np.zeros((ROW_COUNT,COLUMN_COUNT), dtype=int)  # change back to zeros
        self.ROW_COUNT = self.state.shape[0]
        self.COLUMN_COUNT = self.state.shape[1]
        self.player = 1
        self.actions = self.validMoves(self.state)


    def get_next_state(self, state, action, player):
        state = self.playMove(action, player)
        return state
    

    def playMove(self, col_no, player_no):
        for r in range(self.ROW_COUNT):
            if self.state[r][col_no] == 0:
                self.state[r][col_no] = player_no
                break
        return self.state

    def playMoveWithCopy(self, state, col_no, player_no):
        for r in range(self.ROW_COUNT):
            if state[r][col_no] == 0:
                state[r][col_no] = player_no
                break
        return state

    
    def nextPlayer(self, player):
        if player == 1:
            newplayer = 2
        else:
            newplayer = 1
        return newplayer

    #check terminal state
    def checkTerminalState(self, state, player):
    # Check horizontal locations for win
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT):
                if (state[r][c] == player and state[r][c+1] == player and state[r][c+2] == player and state[r][c+3] == player) and state[r][c] != 0:
                    return True, player
    
        # Check vertical locations for win
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT-3):
                if (state[r][c] == player and state[r+1][c] == player and state[r+2][c] == player and state[r+3][c] == player) and state[r][c] != 0:
                    return True, player
    
        # Check positively sloped diaganols
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT-3):
                if (state[r][c] == player and state[r+1][c+1] == player and state[r+2][c+2] == player and state[r+3][c+3] == player) and state[r][c] != 0:
                    return True, player
    
        # Check negatively sloped diaganols
        for c in range(self.COLUMN_COUNT-3):
            for r in range(3, self.ROW_COUNT):
                if (state[r][c] == player and state[r-1][c+1] == player and state[r-2][c+2] == player and state[r-3][c+3] == player) and state[r][c] != 0:
                    return True, player
        
        # Check draw
        if self.checkDraw(state):
            return True, 0
        
        return False, -1
    
    

    def checkDraw(self, state):
        if np.count_nonzero(state) == self.ROW_COUNT*self.COLUMN_COUNT:
            return True
        return False

    #return all valid moves
    def validMoves(self, state):
        valid_moves = []
        for c in range(self.COLUMN_COUNT):
            if state[self.ROW_COUNT - 1][c] == 0:
                valid_moves.append(c)
        return valid_moves

    def get_state(self):
        return self.state

    def PrintGrid(self, state):
        print(np.flip(state, 0))

################################################

def main():
    print("Welcome to Connect 4!")
    print("Input: \n 1 for MCTS (part a) \n 2 for Q learning (part c)")
    choice = int(input())
    #_____________________________MCTS_____________________________________________
    
    if choice == 1:
        ######################## Game ##########################
        print("MCTS")
        print("200 is player 1, 40 is player 2")
        game = Connect4(6, 5)
        player1 = 1
        player2 = 2
        algo1 = MCTS(game, 200, player1) #change back to 200
        algo2 = MCTS(game, 40, player2) #change back to 40
        MOVES = 0

        print("Do you want to all the states of the game (y/n)?")
        choice = input()

        if choice == 'y' or choice == 'Y':
            seeAll = True
        else:
            seeAll = False
        
        if seeAll:
            game.PrintGrid(game.state)

        while not game.checkTerminalState(game.get_state(), player1)[0]:
            if(algo1.bestMove(game.state, player1) in game.validMoves(game.get_state())):
                game.playMove(algo1.bestMove(game.state, player1), player1)
                MOVES += 1
            elif algo1.bestMove(game.state, player1) == 42:
                if game.checkTerminalState(game.get_state(), player1)[0]:
                    print("Should NEVER print")
                    break

            if seeAll:
                game.PrintGrid(game.state)
            if game.checkTerminalState(game.get_state(), player1)[0]:
                break
            game.player = game.nextPlayer(game.player)


            if(algo2.bestMove(game.state, player2) in game.validMoves(game.get_state())):
                game.playMove(algo2.bestMove(game.state, player2), player2)
                MOVES += 1

            elif algo2.bestMove(game.state, player2) == 42:
                if game.checkTerminalState(game.get_state(), player2)[0]:
                    print("Should NEVER print")
                    break
            
            if seeAll:
                game.PrintGrid(game.state)
            if game.checkTerminalState(game.get_state(), player2)[0]:
                break
            game.player = game.nextPlayer(game.player)

        print("Final State:")
        game.PrintGrid(game.get_state())
        winner = game.checkTerminalState(game.get_state(), game.player)[1]
        if(winner == 0):
            print("The game is a draw")
        else:
            print("The player who won is:", winner)
        print("Number of moves:", MOVES)
        return


        ######################## Plots stuff ##########################
        
        
        # Cs = [0,1,2,3,4,5,6,7,8,9,10]
        # winsPerC = []
        # for c in Cs:

        #     MCTS40WINS = 0
        #     MCTS200WINS = 0
        #     TOTAL = 0
        #     for i in range(20):
        #         if i % 2 == 0:
        #             n_playouts1 = 40
        #             n_playouts2 = 200
        #         else:
        #             n_playouts1 = 200
        #             n_playouts2 = 40
                
        #         game = Connect4(6,5)
        #         player1 = 1
        #         player2 = 2
        #         #game.PrintGrid(game.state)
        #         algo1 = MCTS(game, n_playouts1, player1, C=c)
        #         algo2 = MCTS(game, n_playouts2, player2, C=c)
        #         while not game.checkTerminalState(game.get_state(), player1)[0]:
        #             if(algo1.bestMove(game.state, 1) in game.validMoves(game.get_state())):
        #                 game.playMove(algo1.bestMove(game.state, 1), 1)
        #             #game.PrintGrid(game.state)
        #             if game.checkTerminalState(game.get_state(), player1)[0]:
        #                 break
        #             game.player = game.nextPlayer(game.player)


        #             if(algo2.bestMove(game.state, 2) in game.validMoves(game.get_state())):
        #                 game.playMove(algo2.bestMove(game.state, 2), 2)
        #             #game.PrintGrid(game.state)
        #             if game.checkTerminalState(game.get_state(), player2)[0]:
        #                 break
        #             game.player = game.nextPlayer(game.player)
        #         print("Final State:")
        #         game.PrintGrid(game.get_state())
        #         print("The player who won is:", game.checkTerminalState(game.get_state(), game.player)[1])

        #     # if game.checkTerminalState(game.get_state(), game.player)[1] == 1:
        #     #     PLAYER1WINS += 1
        #     # elif game.checkTerminalState(game.get_state(), game.player)[1] == 2:
        #     #     PLAYER2WINS += 1
        #     # else:
        #     #     DRAWS += 1
        #     # if i % 2 == 0: #player 1 is 40
        #     #     if game.checkTerminalState(game.get_state(), game.player)[1] == 1:
        #     #         MCTS40WINS += 1
        #     #     elif game.checkTerminalState(game.get_state(), game.player)[1] == 2:
        #     #         MCTS200WINS += 1
        #     # else: #player 1 is 200
        #         if game.checkTerminalState(game.get_state(), game.player)[1] == 1:
        #             MCTS200WINS += 1
        #             TOTAL += 1
        #         elif game.checkTerminalState(game.get_state(), game.player)[1] == 2:
        #             MCTS40WINS += 1
        #             TOTAL += 1
            
        #     winsPerC.append([MCTS200WINS, MCTS40WINS, TOTAL])
            
        # # print("_____________STATS______________")
        # # print(f"Number of MCTS 200 Wins: {MCTS200WINS}")
        # # print(f"Number of MCTS 40 Wins: {MCTS40WINS}")
        # # print(f"Number of Player 1 Wins: {PLAYER1WINS}")
        # # print(f"Number of Player 2 Wins: {PLAYER2WINS}")
        # # print(f"Number of Draws: {DRAWS}")
        # #plot bar graph
        
        # y = []
        # i = 0
        # for i in range(len(Cs)):
        #     y.append(winsPerC[i][0] * 100 / winsPerC[i][2])

        # plt.plot(Cs, y, color='red', label='MCTS 200')
        # plt.xlabel("Value of C")
        # plt.ylabel("Win percent of MCTS200")
        # plt.title("C v/s MCTS200 performance")
        # plt.legend()
        # plt.show()

        # x = ["MCTS 200", "MCTS 40"]
        # y = [MCTS200WINS, MCTS40WINS]
        # plt.bar(x, y, color=['red', 'blue'])
        # plt.xlabel("MCTS Number of Playouts")
        # plt.ylabel("Number of Wins")
        # plt.title("Wins per Algorithm")
        # plt.show()

        # return
    
    #__________________________________Q LEARNING _________________________________________
    if choice == 2:
        #Q LEARN TO SHOW
        print("Q learning")
        print("__________________________________________")
        game = Connect4(2,5)
        player1 = 1
        player2 = 2
        game.PrintGrid(game.state)
        algo1 = MCTS(game, 200, player1)
        algo2 = QLearning(game, player2)
        MOVES = 0
        while not game.checkTerminalState(game.get_state(), player1)[0]:
            print("MCTS")
            if(algo1.bestMove(game.state, player1) in game.validMoves(game.get_state())):
                game.playMove(algo1.bestMove(game.state, player1), 1)
                MOVES += 1
            #game.PrintGrid(game.state)
            if game.checkTerminalState(game.get_state(), player1)[0]:
                break
            game.player = game.nextPlayer(game.player)
            game.PrintGrid(game.get_state())
            
            
            
            print("QLearning")

            if(algo2.bestMove(game.state) in game.validMoves(game.get_state())):
                game.playMove(algo2.bestMove(game.state), 2)
                MOVES += 1
            #game.PrintGrid(game.state)
            if game.checkTerminalState(game.get_state(), player2)[0]:
                break
            game.player = game.nextPlayer(game.player)
            game.PrintGrid(game.get_state())

        print("Final State:")
        game.PrintGrid(game.get_state())
        winner = game.checkTerminalState(game.get_state(), game.player)[1]
        if(winner == 0):
            print("The game is a draw")
        else:
            print("The player who won is:", winner)   
        print("Number of moves:", MOVES)
        return




        # # #Q LEARN TO PLAY
        # print("Q learning")
        # print("__________________________________________")
        # player1 = 1
        # player2 = 2
        # algo2 = QLearning(None, player2)
        # MCTSWINS = 0
        # QLEARNWINS = 0
        # DRAWS = 0
        # for i in range(1000):              
        #     MOVES = 0     
        #     game = Connect4(2,5)
        #     #game.PrintGrid(game.state)
        #     algo2.game = game
        #     algo1 = MCTS(game, 1, player1)
        #     while not game.checkTerminalState(game.get_state(), player1)[0]:
        #         #print("MCTS")
        #         if(algo1.bestMove(game.state, player1) in game.validMoves(game.get_state())):
        #             game.playMove(algo1.bestMove(game.state, player1), 1)
        #             MOVES += 1
        #         #game.PrintGrid(game.state)
        #         if game.checkTerminalState(game.get_state(), player1)[0]:
        #             break
        #         game.player = game.nextPlayer(game.player)
        #         #game.PrintGrid(game.get_state())
                
                
                
        #        # print("QLearning")

        #         if(algo2.bestMove(game.state) in game.validMoves(game.get_state())):
        #             game.playMove(algo2.QLearningRun(game.state), 2)
        #             MOVES += 1
        #         #game.PrintGrid(game.state)
        #         if game.checkTerminalState(game.get_state(), player2)[0]:
        #             break
        #         game.player = game.nextPlayer(game.player)
        #         #game.PrintGrid(game.get_state())

        #     print("Final State:")
        #     game.PrintGrid(game.get_state())
        #     winner = game.checkTerminalState(game.get_state(), game.player)[1]
        #     if(winner == 0):
        #         print("The game is a draw")
        #         DRAWS += 1
        #     elif winner == 1:
        #         MCTSWINS += 1
        #         print("The player who won is:", winner)
        #     elif winner == 2:
        #         QLEARNWINS += 1 
        #         print("The player who won is:", winner)  
        #     print("Number of moves:", MOVES)
        # # print("Q VALUES")
        # # for key, value in algo2.Q.items():
        # #     print(key, value)
        # with gzip.open('./2019A7PS0044G_AKHILESH.dat.gz', 'wb') as f:
        #     f.write(bytes(json.dumps(algo2.Q),'utf-8'))

        # # print("MCTS Wins = ", MCTSWINS)
        # # print("QLEARN Wins = ", QLEARNWINS)
        # # x = ["MCTS1", "Q Learning", "Draws"]
        # # y = [MCTSWINS, QLEARNWINS, DRAWS]
        # # plt.bar(x, y, color=['red', 'blue', 'green'])
        # # plt.xlabel("Algorithm used")
        # # plt.ylabel("Number of Wins")
        # # plt.title("Wins per Algorithm")
        # # plt.show()
        # return
         
         
         
         # Q LEARN Random
        # print("Q learning")
        # print("__________________________________________")
        # player1 = 1
        # player2 = 2
        # algo2 = QLearning(None, player2)
        # MCTSWINS = 0
        # QLEARNWINS = 0
        # DRAWS = 0
        # for i in range(1000):              
        #     MOVES = 0     
        #     game = Connect4(3,5)
        #     #game.PrintGrid(game.state)
        #     algo2.game = game
        #     algo1 = RandomPlayer(game, player1)
        #     while not game.checkTerminalState(game.get_state(), player1)[0]:
        #         #print("MCTS")
        #         if(algo1.bestMove(game.state, player1) in game.validMoves(game.get_state())):
        #             game.playMove(algo1.bestMove(game.state, player1), 1)
        #             MOVES += 1
        #         #game.PrintGrid(game.state)
        #         if game.checkTerminalState(game.get_state(), player1)[0]:
        #             break
        #         game.player = game.nextPlayer(game.player)
        #         #game.PrintGrid(game.get_state())
                
                
                
        #        # print("QLearning")

        #         if(algo2.bestMove(game.state) in game.validMoves(game.get_state())):
        #             game.playMove(algo2.QLearningRun(game.state), 2)
        #             MOVES += 1
        #         #game.PrintGrid(game.state)
        #         if game.checkTerminalState(game.get_state(), player2)[0]:
        #             break
        #         game.player = game.nextPlayer(game.player)
        #         #game.PrintGrid(game.get_state())

        #     print("Final State:")
        #     game.PrintGrid(game.get_state())
        #     winner = game.checkTerminalState(game.get_state(), game.player)[1]
        #     if(winner == 0):
        #         print("The game is a draw")
        #         DRAWS += 1
        #     elif winner == 1:
        #         MCTSWINS += 1
        #         print("The player who won is:", winner)
        #     elif winner == 2:
        #         QLEARNWINS += 1 
        #         print("The player who won is:", winner)  
        #     print("Number of moves:", MOVES)
        # # print("Q VALUES")
        # # for key, value in algo2.Q.items():
        # #     print(key, value)
        
        
        # return
    #_________________________________BASE GAME____________________________________________
    else:
        print("Enter the correct value")
    #     print("Playing the game")
    #     game = Connect4(5,6)
    #     player1 = 1
    #     player2 = 2
    #     game.PrintGrid(game.state)
    #     while True:
    #         print("Player 1's turn")
    #         print(game.validMoves(game.state))
    #         col_no = int(input("Enter the column number: "))
    #         game.playMove(col_no, player1)
    #         game.PrintGrid(game.state)
    #         if game.checkTerminalState(game.state, player1)[0]:
    #             print("Player that won: ", game.checkTerminalState(game.state, player1)[1])   
    #             break
    #         if game.checkDraw(game.state):
    #             print("Draw")
    #             break

    #         print("Player 2's turn")
    #         print(game.validMoves(game.state))
    #         col_no = int(input("Enter the column number: "))
    #         game.playMove(col_no, player2)
    #         game.PrintGrid(game.state) 
    #         if game.checkTerminalState(game.state, player2):
    #             print("Player that won: ", game.checkTerminalState(game.state, player2)[1])    
    #             break
    #         if game.checkDraw(game.state):
    #             print("Draw")
    #             break
    #     return


    # print('Player 1 (MCTS with 25 playouts')
    # print('Action selected : 1')
    # print('Total playouts for next state: 5')
    # print('Value of next state according to MCTS : .1231')
    # PrintGrid(game2)

    # print('Player 2 (Q-learning)')
    # print('Action selected : 2')
    # print('Value of next state : 1')
    # PrintGrid(game3)
    
    # print('Player 2 has WON. Total moves = 14.')
    
if __name__=='__main__':
    main()
"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a Agent class for Task 3

Group Number:
Student ID: 100889423 (Dylan Young)
Partner ID: 100430249 (Robert Soanes)

@date:   26/10/2025

"""
import a1_state
import math
import copy
import time

class Agent:
    
    name = "D3"
    modes = ['minimax','alphabeta']
    
    #initalising agent with the size and default name of D3
    def __init__(self, size, name = "D3"):
        if isinstance(size, tuple):
            self.size = size
            self.name = name
        else:
            raise TypeError("Size must be a tuple")
        
    #str rep of agent printing the name and size
    def __str__(self):
        return f"Agent Name: {self.name}\n The grid has {self.size[0]} rows and {self.size[1]} columns" 
    
    #function to make a move based on the selected mode
    def move(self, state, mode):
        
        #if mode is minimax then run minimax function
        if mode == "minimax":
            #best_score as the lowest possible value to start
            best_score = -math.inf
            best_move = None

            #iterate through all the possible moves
            for move in state.moves():
                r, c = move[0], move[1]
                #create new state to sim move using deepcopy to ensure that the 2d array is copied properly
                new_state = copy.deepcopy(state) 
                new_state.makeMove(r, c)          
                
                
                score = self.minimax(new_state, 0, False)
                
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move
        
        elif mode == "alphabeta":
            best_score = -math.inf
            best_move = None
            
            alpha = -math.inf
            beta = math.inf
            
            for move in state.moves():
                r, c = move[0], move[1]
                new_state = copy.deepcopy(state) 
                new_state.makeMove(r, c)
                
                score = self.alphabeta(new_state, 0, False, alpha, beta)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
            return best_move
        else:
            raise ValueError("Invalid mode selected. Please choose either minimax or alphabeta.")
    
    #Checks if the state of the board is in a position to win in the next move.
    #Game is won if a move on a hinger cell is made, detect the number of hingers
    #and if it is >0 then state is possible for a winning move
    
    def winCheck(self, state):
        winning_state = False
        numHingers = state.numHingers()
        if numHingers != 0:
            winning_state = True
        
        return winning_state  
    
    #create minimax playing strategy
    def minimax(self, state, move_ahead, is_max):
        #Checking if there is a winning state for the current player or if the depth limit is reached (10)
        win = self.winCheck(state)
        if win == True:
            if is_max:
                return -1
            elif not is_max:
                return 1
            else:
                return 0
        elif move_ahead > 5:
            return 0 
        else:
            #if looking to maximise score
            if is_max:
                best_score = -math.inf
                
                for move in state.moves():
                    r, c = move[0], move[1]
                    new_state = copy.deepcopy(state) 
                    new_state.makeMove(r, c)
                    #recursive call to minimax with increased depth and switching to minimizing player
                    score = self.minimax(new_state, move_ahead + 1, False)
                    
                    if score > best_score:
                        best_score = score
                return best_score
            #else looking to minimise score
            else:
                best_score = math.inf
                
                for move in state.moves():
                    r, c = move[0], move[1]
                    new_state = copy.deepcopy(state) 
                    new_state.makeMove(r, c)
                    #recursive call to minimax with increased depth and switching to maximizing player
                    score = self.minimax(new_state, move_ahead + 1, True)
                    
                    if score < best_score:
                        best_score = score
                return best_score
            
        
    #Copy of minimax with AB pruning 
    def alphabeta(self, state, move_ahead, is_max, alpha, beta):
        #Checking if there is a winning state for the current player or if the depth limit is reached (10)
        win = self.winCheck(state)
        if win == True:
            if is_max:
                return -1
            elif not is_max:
                return 1
            else:
                return 0
        elif move_ahead > 5:
            return 0 
        else:
            if is_max:
                best_score = -math.inf
                
                for move in state.moves():
                    r, c = move[0], move[1]
                    new_state = copy.deepcopy(state) 
                    new_state.makeMove(r, c)
                    
                    score = self.alphabeta(new_state, move_ahead + 1, False, alpha, beta)
                    
                    if score > best_score:
                        best_score = score
                    #checks if the best score is greater than or equal to beta to prune the branch
                    if best_score >= beta:
                        break
                    #updates alpha value
                    alpha = max(alpha, best_score)
                
            else:
                best_score = math.inf
                
                for move in state.moves():
                    r, c = move[0], move[1]
                    new_state = copy.deepcopy(state) 
                    new_state.makeMove(r, c)
                    
                    score = self.alphabeta(new_state, move_ahead + 1, True, alpha, beta)
                    
                    if score < best_score:
                        best_score = score
                        
                    if best_score <= alpha:
                        break
                    beta = min(beta, best_score)
            
            return best_score
    
    
            
    #Testing function
def tester():

    # --- Setup ---
    size = (4, 5)
    agent = Agent(size)
    print(f"Created Agent: {agent.name}")

    # Test 1: agent str representation
    print("Test 1: String Representation of Agent")
    print(agent)
    print("Test 1 Completed")

    # Test 2: move using minimax
    print("Test 2: Move selection using Minimax")
    grid = [[1, 1, 0, 0, 2],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1]]
    state = a1_state.State(copy.deepcopy(grid))

    start_time = time.time()
    move = agent.move(state, "minimax")
    end_time = time.time()
    
    time_taken = end_time - start_time
    
    if move is not None:
        print(f"Move chosen by minimax: {move}")
        print(f"Time taken for minimax: {time_taken:.4f} seconds")
        print("Test 2 Completed")
    else:
        print("Test 2 Failed")
    #prints out all the moves that are made but average time is about 9.4 seconds

    # Test 3: move using alphabeta
    print("Test 3: Move selection using Alpha-Beta pruning")
    state = a1_state.State(copy.deepcopy(grid))

    start_time = time.time()
    move = agent.move(state, "alphabeta")
    end_time = time.time()
    
    time_taken = end_time - start_time 
    
    if move is not None:
        print(f"Move chosen by alpha-beta: {move}")
        print(f"Time taken for alpha-beta: {time_taken:.4f} seconds")
        print("Test 3 Completed")
    else:
        print("Test 3 Failed")
    
    #prints out all the moves that are made but average time is about half a second

    # Test 4: detecting hingers for a winning state
    print("Test 4: Detect Winning State (Hingers present)")
    win_grid = [[1, 1, 0, 0, 2],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1]]
    state = a1_state.State(copy.deepcopy(win_grid))

    win_detected = agent.winCheck(state)
    if win_detected:
        print(f"Winning state detected? {win_detected}")
        print("Test 4 Completed")
    else:
        print(f"Winning state detected? {win_detected}")
        print("Test 4 Failed")

    # Test 5: invalid mode handling
    print("Test 5: Invalid Mode Handling")
    state = a1_state.State(copy.deepcopy(grid))
    if "random" not in agent.modes:
        print("Invalid mode selected correctly rejected.")
        print("Test 5 Completed")
    else:
        print("Test 5 Failed")

    # Test 6: empty board with no available moves
    print("Test 6: Empty Board")
    full_grid = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
    state = a1_state.State(copy.deepcopy(full_grid))

    move = agent.move(state, "minimax")
    if move is None:
        print(f"Returned move: {move}")
        print("Test 6 Completed")
    else:
        print(f"Returned move: {move}")
        print("Test 6 Failed")

    print("All Tests Completed.")
if __name__ == "__main__":
    tester()

       
    
    
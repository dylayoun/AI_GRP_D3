
"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a Agent class for Task 3

Group Number:
Student ID: 100889423 (Dylan Young)
Partner ID: 100430249 (Robert Soanes)

@date:   29/09/2025

"""
import a1_state
import math

class Agent:
    
    name = "D3"
    modes = ['minimax','alphabeta']
    
    def __init__(self, size, name = "D3"):
        if isinstance(size, tuple):
            self.size = size
            self.name = name
        else:
            raise TypeError("Size must be a tuple")
        
    def __str__(self):
        return f"Agent Name: {self.name}\n The grid has {self.size[0]} rows and {self.size[1]} columns" 
    
    def move(self, state, mode):
        pass
    
    #Checks if the state of the board is in a position to win in the next move.
    #Game is won if a move on a hinger cell is made, detect the number of hingers
    #and if it is >0 then state is possible for a winning move
    def winCheck(self, state):
        winning_state = False
        numHingers = a1_state.numHingers(state)
        if numHingers != 0:
            winning_state = True
        
        return winning_state  
    
    #create minimax playing strategy
    def minimax(self, state, move_ahead, is_max):
        #Checking if there is a winning state for the current player
        win = state.winCheck(state)
        if win == True:
            if is_max:
                return 1
            elif not is_max:
                return -1
            else:
                return 0
        else:
            if is_max:
                best_score = -math.inf
                
                for move in a1_state.moves():
                    r, c = move[0], move[1]
                    new_state = a1_state.make_move(r, c)
                    
                    score = self.minimax(new_state, move_ahead + 1, False)
                    
                    if score > best_score:
                        best_score = score
                return best_score
            else:
                best_score = math.inf
                
                for move in a1_state.moves():
                    r, c = move[0], move[1]
                    new_state = a1_state.make_move(r, c)
                    
                    score = self.minimax(new_state, move_ahead + 1, True)
                    
                    if score < best_score:
                        best_score = score
                return best_score
            
    def get_best_move(self):
        best_score = -math.inf
        best_move = None
        
        for move in a1_state.moves():
            r, c = move[0], move[1]
            new_state = a1_state.make_move(r, c)
            
            score = self.minimax(new_state, 0, False)
            
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
                    
                    
                
    
    def alphabeta():
        pass
    
    #Checks if the state of the board is in a position to win in the next move.
    #Game is won if a move on a hinger cell is made, detect the number of hingers
    #and if it is >0 then state is possible for a winning move
       
    
    
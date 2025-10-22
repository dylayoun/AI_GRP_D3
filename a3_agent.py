
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

class Agent:
    
    name = "D3"
    modes = ['minimax','alphabeta']
    
    def __init__(self, size):
        if size.isinstance(tuple):
            self.size = size
        else:
            raise TypeError("Size must be a tuple")
    
    def __init__(self, size, name = "D3"):
        if isinstance(size, tuple):
            self.size = size
            self.name = name
        else:
            raise TypeError("Size must be a tuple")
        
    def __str__(self):
        return f"Agent Name: {self.name}\n The grid has {self.size[0]} rows and {self.size[1]} columns" 
    
    def move(state, mode):
        pass
    
    #Checks if the state of the board is in a position to win in the next move.
    #Game is won if a move on a hinger cell is made, detect the number of hingers
    #and if it is >0 then state is possible for a winning move
    def winCheck(state):
        winning_state = False
        numHingers = a1_state.numHingers(state)
        if numHingers != 0:
            winning_state = True
        
        return winning_state  
    
    #create minimax playing strategy
    def minimax(state, move_ahead, is_max):
        win = winCheck(state)
        if win == True:
            return("Game state is in a winning position for the current player")
        else:
    
    def alphabeta():
        pass
    
    #Checks if the state of the board is in a position to win in the next move.
    #Game is won if a move on a hinger cell is made, detect the number of hingers
    #and if it is >0 then state is possible for a winning move
       
    
    
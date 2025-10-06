"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: Dylan Young (100889423)
@date:   29/09/2025

"""


class State:
    
    def __init__ (self, grid):
        #check the parameter grid is a 2d list using isinstance
        if isinstance(grid, list):
            for row in grid:
                if isinstance(row, list):
                    self.grid = grid
                else:
                    raise ValueError("Grid must be a 2D list")
        else:
            raise ValueError("Grid must be a 2D list")  
                
            
        
    def __str__(self):
        #printing the grid in an understandable format
        for row in self.grid:
            row_str = ' '.join(map(str,row))
            
            print(row_str)
    
    def moves():
        
    
    def numRegions():
        pass
    
    def numHingers():
        pass
    
    

State([[0, 1, 0], [1, 0, 1, 4], [0, 0, 0]]).__str__()

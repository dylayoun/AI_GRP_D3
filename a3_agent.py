
"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a Agent class for Task 3

@author: Dylan Young (100889423)
@date:   29/09/2025

"""


class Agent:
    
    def __init__(self, size):
        if size.isinstance(tuple):
            self.size = size
        else:
            raise TypeError("Size must be a tuple")
    
    def __init__(self, size, name = "D3"):
        if size.isinstance(tuple):
            self.size = size
            self.name = name
        else:
            raise TypeError("Size must be a tuple")
        
    def __str__(self):
        return f"Agent Name: {self.name}\n The grid has {self.size[0]} rows and {self.size[1]} columns" 
    
    def move(state, mode):
        pass
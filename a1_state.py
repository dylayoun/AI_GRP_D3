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
                    self.row_len = len(grid)
                    self.col_len = len(grid[0])
                else:
                    raise TypeError("Grid must be a 2D list")
        else:
            raise TypeError("Grid must be a 2D list")  
                
            
        
    def __str__(self):
        #printing the grid in an understandable format
        for row in self.grid:
            #turns the row into a string with whitespace between each element
            row_str = ' '.join(map(str,row))
            
            print(row_str)
        return ""
    
    def moves(self):
        moves = []
        for row in range(self.row_len):
            for element in range(self.col_len):
                if self.grid[row][element] > 0:
                    #NEED TO DETERMINE THE COST OF EACH MOVE (one plus the number of active cells the cell is adjacent to)
                    #Still to debug cost is not correct in all cases
                    adjCells = self.getAdjacentVal(self.grid, row, element)
                    for i in adjCells:
                        if i > 0:
                            cost = 1 + adjCells.count(i)
                    moves.append((row,element,cost))
        return moves
                
    #function for use of numRegions
    def getAdjacentPos(self, grid, posa, posb):
        #defining the return list and the x and y length of the grid
        adjPos = []
        x = len(grid)
        y = len(grid[0])
     
        #defining the 8 possible neighbours of a specified cell
        neighbours = [(-1,-1), (-1,0,), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        #looping through the neighbours and checking if they are within the grid bounds
        for nx, ny in neighbours:
            #assigning the row and col value of the neighbour
            r = posa + nx
            c = posb + ny
            if 0 <= r < x and 0 <= c < y:
                adjPos.append((r,c))
        
        #returns the position of all the adjacent cells
        return adjPos
    
    def getAdjacentVal(self, grid, posa, posb):
    #defining the return list and the x and y length of the grid
        adjPos = []
        x = len(grid)
        y = len(grid[0])
        
        #defining the 8 possible neighbours of a specified cell
        neighbours = [(-1,-1), (-1,0,), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        #looping through the neighbours and checking if they are within the grid bounds
        for nx, ny in neighbours:
            #assigning the row and col value of the neighbour
            r = posa + nx
            c = posb + ny
            if 0 <= r < x and 0 <= c < y:
                adjPos.append(grid[r][c])
        
        #returns the position of all the adjacent cells
        return adjPos
            
    
    def numRegions(self):
        regCount = 0
        
        #creating a 2d list matching the grid size for visited cells
        visited = []
        for _ in range(self.row_len):
            row = []
            for _ in range(self.col_len):
                row.append(False)
            visited.append(row)
            
        #loop through every cell in grid and if active cell and not visited start iterative dfs
        for r in range(self.row_len):
            for c in range(self.col_len):
                if self.grid[r][c] > 0 and not visited[r][c]:
                    #stack to hold cells that need to be visited
                    stack = [(r,c)]
                    visited[r][c] = True
                    regCount += 1
                    #iterative dfs using stack
                    while stack:
                        current_r, current_c = stack.pop()
                        #call getadjacent to get all adjacent cells
                        for nei_r, nei_c in self.getAdjacent(self.grid, current_r, current_c):
                            #if cell active and not visited mark as visited and push on stack
                            if self.grid[nei_r][nei_c] > 0 and not visited[nei_r][nei_c]:
                                visited[nei_r][nei_c] = True
                                stack.append((nei_r, nei_c))
        return regCount
            

    
    def numHingers(self):
        hinCount = 0
        currRegion = self.numRegions()
        
        for r in range(self.row_len):
            for c in range(self.col_len):
                if self.grid[r][c] != 1:
                    continue
                else:
                    newGrid = [self.grid[i][:] for i in range(self.row_len)]
                    newGrid[r][c] = 0
                    newState = State(newGrid)
                    newRegion = newState.numRegions()
                    if newRegion > currRegion:
                        hinCount += 1
        return hinCount
                    
    
    

game = State([[1, 1, 0, 0, 1], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])

print(game)

print(game.moves())



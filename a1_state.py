"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

Group Number:
Student ID: 100889423 (Dylan Young)
Partner ID: 100430249 (Robert Soanes)

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
        #create list to hold all valid moves
        moves = []
        count = 0
        for row in range(self.row_len):
            for element in range(self.col_len):
                #if element is valid to make a move on
                if self.grid[row][element] > 0:
                    #get all adjacent cells
                    adjCells = self.getAdjacentVal(self.grid, row, element)
                    #detemine the cost of move = all valid elements adjacent(count) + 1
                    for i in adjCells:
                        if i > 0:
                            count = count + 1
                    #add to list and reset cost and count
                    cost = 1 + count
                    moves.append((row,element,cost))
                    cost = 0
                    count = 0
        return moves
    
    #Func to make a move at a specific coord point
    #Need to debug ln61 to only remove 1 value not turn into -1
    def makeMove(self, posx, posy):
        #Interate through all elements
        if self.grid[posx][posy] > 0:
            self.grid[posx][posy] -= 1
            print(f"move made @ ({posx,posy}) and new value is {self.grid[posx][posy]}")
        else:
            print("the position you specified is invalid")
                
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
            #detects that the neighbour is in the grid and not outside of the border
            if 0 <= r < x and 0 <= c < y:
                adjPos.append((r,c))
        
        #returns the position of all the adjacent cells
        return adjPos
    
    def getAdjacentVal(self, grid, posa, posb):
    #defining the return list and the x and y length of the grid
        adjVal = []
        x = len(grid)
        y = len(grid[0])
        
        #defining the 8 possible neighbours of a specified cell
        neighbours = [(-1,-1), (-1,0,), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        #looping through the neighbours and checking if they are within the grid bounds
        for nx, ny in neighbours:
            #assigning the row and col value of the neighbour
            r = posa + nx
            c = posb + ny
            #detects that the neighbour is in the grid and not outside of the border
            if 0 <= r < x and 0 <= c < y:
                adjVal.append(grid[r][c])
        
        #returns the position of all the adjacent cells
        return adjVal
            
    
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
                        for nei_r, nei_c in self.getAdjacentPos(self.grid, current_r, current_c):
                            #if cell active and not visited mark as visited and push on stack
                            if self.grid[nei_r][nei_c] > 0 and not visited[nei_r][nei_c]:
                                visited[nei_r][nei_c] = True
                                stack.append((nei_r, nei_c))
        return regCount
            

    
    def numHingers(self):
        #initalises the amount of hingers and the current number of regions
        hinCount = 0
        currRegion = self.numRegions()
        
        for r in range(self.row_len):
            for c in range(self.col_len):
                #hinger can only be on a element of value 1
                if self.grid[r][c] != 1:
                    continue
                else:
                    #creates a copy of the grid
                    newGrid = [self.grid[i][:] for i in range(self.row_len)]
                    #changes the specific element to 0
                    newGrid[r][c] = 0
                    newState = State(newGrid)
                    newRegion = newState.numRegions()
                    #if the region count increases then hinger has been found 
                    if newRegion > currRegion:
                        hinCount += 1
        return hinCount
    
    def hingerLoc(self):
        #initalises the amount of hingers and the current number of regions
        hinCount = []
        currRegion = self.numRegions()
        
        for r in range(self.row_len):
            for c in range(self.col_len):
                #hinger can only be on a element of value 1
                if self.grid[r][c] != 1:
                    continue
                else:
                    #creates a copy of the grid
                    newGrid = [self.grid[i][:] for i in range(self.row_len)]
                    #changes the specific element to 0
                    newGrid[r][c] = 0
                    newState = State(newGrid)
                    newRegion = newState.numRegions()
                    #if the region count increases then hinger has been found 
                    if newRegion > currRegion:
                        hinCount.append(newGrid[r][c])
        return hinCount
                    

def tester():
    sa = State([[1, 1, 0, 0, 2], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])
    
    print(sa)
    
    #Test 1 print out all possible moves and compare 
    
    poss_moves = [(0,0,4),(0,1,4),(0,4,1),(1,0,4),(1,1,5), (2,2,4), (2,3,5),(2,4,4),(3,3,5),(3,4,4)]
    detected_moves = sa.moves()
    
    for i, move in enumerate(detected_moves):
        if move != poss_moves[i]:
            print(f"Move {move}is not correct")
        else:
            print(f"Move {move} is correct")

    
    #Test 2: print the number of hingers in the example board and compare
    
    detected_hin = sa.numHingers() 
    
    if detected_hin == 2:
        print("Test 2 Completed Perfectly")
    else:
        print("Test 2 not completed")
        
    #Test 3: Number of regions on example board and compare
    
    detected_reg = sa.numRegions()
    
    if detected_reg == 2:
        print("Test 3 Completed Perfectly")
    else:
        print("Test 3 not completed")
        
    #Test 4: Check that the adjacent position functions are working with example board
    
    example_posx = 2
    example_posy = 3
    
    adj_pos = sa.getAdjacentPos(sa.grid, example_posx, example_posy)
    adj_val = sa.getAdjacentVal(sa.grid, example_posx, example_posy)
    
    adj_pos_true = [(1,2),(1,3),(1,4),(2,2),(2,4),(3,2),(3,3),(3,4)]
    adj_val_true = [0,0,0,1,1,0,1,1]
    
    for i , adj in enumerate(adj_pos):
        if adj != adj_pos_true[i]:
            print(f"the position of neighbour {adj} is incorrect")
        else:
            print(f"the position of neighbour {adj} is correct")
            
    for i , adj in enumerate(adj_val):
        if adj != adj_val_true[i]:
            print(f"the position of neighbour {adj} is incorrect")
        else:
            print(f"the position of neighbour {adj} is correct")
            
            
    #Test 5: makeMove function testing with example element (0,4)
    # move should reduce the value 2 into 1 and then the new grid is printed using __str__ method
    
    move = sa.makeMove(0,4)
    
    print(sa)
    
    
    
    
if __name__ == '__main__':
    tester()



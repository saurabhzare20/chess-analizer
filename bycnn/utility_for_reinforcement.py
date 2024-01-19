import csv
from utility_for_boardmovement import *
from utility_for_network import *

def find_line_number(csv_file, target_element):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for line_number, row in enumerate(reader, start=1):
            if row and row[0] == target_element:
                return line_number
    return None
def cheakmate(board, toggle,moves,white,black):
    if toggle:
       for row in range(8):
          for col in range(8):
            if board[row][col] == 10:  # Black king is represented by -10
                king_position = (row, col)
                #print(king_position)
                break
       if not is_white_king_in_check(moves,king_position):
               return True
       else:   
             if white ==[] and is_white_king_in_check(moves,king_position):
                 return False
             else:
                 return True

               
       
            
    else:   
       for row in range(8):
        for col in range(8):
            if board[row][col] == -10:  # Black king is represented by -10
                king_position = (row, col)
                break
       if not is_black_king_in_check(moves,king_position):
              
              return True
       else:   
             if black == [] and is_black_king_in_check(moves,king_position):
               
                 return False
             else:
                 return True
          
def is_black_king_in_check(moves,black_king_position):
     
    for move in moves:
         for move in moves:
           if len(move)==6:
             piece, start_row, start_col, end_row, end_col,promotion = move
             if piece > 0:  # Checking if the piece is of the opponent
              if (end_row, end_col) == black_king_position:
               
                  return True  # Black king is in check
           elif len(move)==2:
              if move[0][0]> 0:
             
                 continue       
           else:   
             piece, start_row, start_col, end_row, end_col = move
             if piece > 0:  # Checking if the piece is of the opponent
               if (end_row, end_col) == black_king_position:
               
                return True  # Black king is in check 
    return False 
                 # Black king is in check

     
def is_white_king_in_check(moves,white_king_position):
    for move in moves:
       
        if len(move)==6:
          piece, start_row, start_col, end_row, end_col,promotion = move
          if piece < 0:  # Checking if the piece is of the opponent
            if (end_row, end_col) == white_king_position:
               
                return True
                  # Black king is in check
        elif len(move)==2:
            if move[0][0]< 0:
             
                continue

        else:   
             piece, start_row, start_col, end_row, end_col = move
             if piece < 0:  # Checking if the piece is of the opponent
               if (end_row, end_col) == white_king_position:
               
                return True  # Black king is in check 
    return False 
def filter_moves(board, moves,toggle,king_position):

    newMoves=[]
  
   
   
    if toggle:
        
        for move in moves:
           
           
           
            if move[0]>0:
               
               board_copy = [row[:] for row in board]
               
         
               board_copy=performMove(move,board_copy)
               for row in range(8):
                  for col in range(8):
                    if board_copy[row][col] == 10:  # Black king is represented by -10
                       king_position = (row, col)
                       break
               
           
               macc= calculate_all_moves(board_copy)
              
              
                # Simulate the move
               if not is_white_king_in_check( macc,king_position) : 
                       newMoves.append(move)

            else:
                newMoves.append(move)  

               #return True  # Check is preventable
    if not toggle:
         
          for move in moves:
             
             
            
             if move[0]<0:
               
               board_copy = [row[:] for row in board]
          # Create a copy of the board
               board_copy=performMove( move,board_copy)
               
               for row in range(8):
                  for col in range(8):
                    if board_copy[row][col] == -10:  # Black king is represented by -10
                       king_position = (row, col)
                       break  
               
           #print(board_copy)
               macc= calculate_all_moves(board_copy) 
              
               if not is_black_king_in_check(macc,king_position) : 
                        newMoves.append(move)
               #return True  # Check is preventable
               
             else:
                newMoves.append(move) 
    return newMoves


    return newMoves # Check is not preventable
def specialmoves(board,moves,kingmoved,toggle,last):
    if len(last)>1:
       lastboard=last[-2]
    if toggle:
       
            
        kingside_castle = (
        not kingmoved
        and  board[0][4] == 10
        and board[0][7]==5
        and  board[0][5] == board[0][6] == 0  # Check if squares between king and rook are empty
          # Check if squares the king crosses are not under attack
        and not is_white_king_in_check(moves,(0,5))
       
        and not is_white_king_in_check(moves,(0,6))
    )

        queenside_castle = (
        not kingmoved 
        and  
        board[0][4] == 10
        and board[0][0]==5
         
        and  board[0][1] == board[0][2] ==  board[0][3]==0  # Check if squares between king and rook are empty
        
        and not is_white_king_in_check(moves,(0,1))
       
        and not is_white_king_in_check(moves,(0,2))
        and not is_white_king_in_check(moves,(0,3))
        

    )
    
        for x in range(8):
           if board[4][x]==1:
              
              if x ==0 and  lastboard[6][1]==-1  and  board[4][1]==-1:
                  moves.append((1,4,x,5,x+1))
              elif x ==7 and  lastboard[6][x-1]==-1  and  board[4][x-1]==-1:
                  moves.append((1,4,x,5,x-1)) 
              elif 0<x<7:
                 if  lastboard[6][x+1]==-1  and  board[4][x+1]==-1: 
                     moves.append((1,4,x,5,x+1))
                 elif lastboard[6][x-1]==-1  and  board[4][x-1]==-1:  
                       moves.append((1,4,x,5,x-1)) 
                    
             
    
        if  kingside_castle:
            moves.append(((10,0,4,0,6),(5,0,7,0,5)))
        if  queenside_castle:
            moves.append(((10,0,4,0,2),(5,0,0,0,3))) 
        return moves    
   
   
    if not toggle:
          black_kingside_castle = (
               not kingmoved
          and board[7][4] == -10
        and board[7][7]==-5
        and  board[7][5] == board[7][6] == 0  # Check if squares between king and rook are empty
          # Check if squares the king crosses are not under attack
        and not is_black_king_in_check(moves,(7,5))
       
        and not is_black_king_in_check(moves,(7,6))
    )

          black_queenside_castle = (
            not kingmoved
            and   board[7][4] == -10
            and board[7][0]==-5
         
            and  board[7][1] == board[7][2] ==  board[7][3]==0  # Check if squares between king and rook are empty
        
            and not is_black_king_in_check(moves,(7,1))
       
            and not is_black_king_in_check(moves,(7,2))
            and not is_black_king_in_check(moves,(7,3))
        

    )
      
          for x in range(8):
           if board[3][x]==-1:
              if x ==0 and  lastboard[6][1]==1  and  board[3][1]==1:
                  moves.append((-1,3,0,2,1))
              elif x ==7 and  lastboard[6][6]==1  and  board[3][6]==1:
                  moves.append((-1,3,7,2,6)) 
              elif  0<x<7:
                 if  lastboard[6][x+1]==1  and  board[3][x+1]==1: 
                     moves.append((-1,3,x,2,x+1))
                 elif lastboard[6][x-1]==1  and  board[3][x-1]==1:  
                       moves.append((-1,3,x,2,x-1))

          if  black_kingside_castle:
            moves.append(((-10,7,4,7,6),(-5,7,7,7,5)))
          if black_queenside_castle:
             moves.append(((-10,7,4,7,2),(-5,7,0,7,3)))  

          return moves             
def dividebothmoves(allmoves):
    whitemoves=[]
    blackmove=[]
   
    for move in allmoves:
           if len(move)!=2:
             if move[0]>0:
               whitemoves.append(move)
             if move[0]<0:
               blackmove.append(move) 

           else:
               if move[0][0]>0:
                 whitemoves.append(move)
               if move[0][0]<0:
                 blackmove.append(move) 
       

    return whitemoves,blackmove  
def aditional(board,moves,kingmoved,toggle,last):
    lastboard=last[-1]
    if toggle:
       
            
        kingside_castle = (
        not kingmoved
        and  board[0][4] == 10
        and board[0][7]==5
        and  board[0][5] == board[0][6] == 0  # Check if squares between king and rook are empty
          # Check if squares the king crosses are not under attack
        and not is_white_king_in_check(moves,(0,5))
       
        and not is_white_king_in_check(moves,(0,6))
    )

        queenside_castle = (
          not kingmoved   
          and board[0][4] == 10
        and board[0][0]==5
         
        and  board[0][1] == board[0][2] ==  board[0][3]==0  # Check if squares between king and rook are empty
        
        and not is_white_king_in_check(moves,(0,1))
       
        and not is_white_king_in_check(moves,(0,2))
        and not is_white_king_in_check(moves,(0,3))
        

    )
    
        for x in(0,7):
           if board[4][x]==1:
              if x ==0 and  lastboard[6][x+1]==-1  and  board[4][x+1]==-1:
                  moves.append((1,4,x,5,x+1))
              elif x ==7 and  lastboard[6][x-1]==-1  and  board[4][x-1]==-1:
                  moves.append((1,4,x,5,x-1)) 
              else:
                 if  lastboard[6][x+1]==-1  and  board[4][x+1]==-1: 
                     moves.append((1,4,x,5,x+1))
                 elif lastboard[6][x-1]==-1  and  board[4][x-1]==-1:  
                       moves.append((1,4,x,5,x-1)) 
                    
             
    
        if  kingside_castle:
            moves.append(((10,0,4,0,6),(5,0,7,0,5)))
        if  queenside_castle:
            moves.append(((10,0,4,0,2),(5,0,0,0,3))) 
   
   
    if not toggle:
          black_kingside_castle = (
               not kingmoved
          and board[7][4] == -10
        and board[7][7]==-5
        and  board[7][5] == board[7][6] == 0  # Check if squares between king and rook are empty
          # Check if squares the king crosses are not under attack
        and not is_black_king_in_check(moves,(7,5))
       
        and not is_black_king_in_check(moves,(7,6))
    )

          black_queenside_castle = (
             kingmoved
            and   board[7][4] == -10
            and board[7][0]==-5
         
            and  board[7][1] == board[7][2] ==  board[7][3]==0  # Check if squares between king and rook are empty
        
            and not is_black_king_in_check(moves,(0,1))
       
            and not is_black_king_in_check(moves,(0,2))
             and not is_black_king_in_check(moves,(0,3))
        

    )
      
          for x in(0,7):
           if board[3][x]==-1:
              if x ==0 and  lastboard[6][x+1]==1  and  board[3][x+1]==1:
                  moves.append((-1,3,x,2,x+1))
              elif x ==7 and  lastboard[6][x-1]==1  and  board[3][x-1]==1:
                  moves.append((-1,3,x,2,x-1)) 
              else:
                 if  lastboard[6][x+1]==1  and  board[3][x+1]==1: 
                     moves.append((-1,3,x,2,x+1))
                 elif lastboard[6][x-1]==1  and  board[3][x-1]==1:  
                       moves.append((-1,3,x,2,x-1))

          if  black_kingside_castle:
            moves.append(((10,7,4,7,6),(5,7,7,7,5)))
          if black_queenside_castle:
             moves.append(((10,7,4,7,2),(5,7,0,7,3)))            


def isdraw(board,toggle,white,black,lastsix):
    
    pieces = []

    for row in board:
        for piece in row:
            if piece != 0:
                pieces.append(piece)
    conditions = [
        set(pieces) == {10,-10, -3, 3}, 
        set(pieces) == {10, -10},  
        set(pieces) == {10, -10, -4, 4},
        set(pieces) == {10, -10, 4}, 
        set(pieces) == {10, -10, -4},
        set(pieces) == {10, -10, -3},
        set(pieces) == {10, -10, 3},   

        # Example: Only kings and bishops on the board for a specific color
       
    ]            
    if any(conditions):
        return False          
    if len(lastsix)==6:
        if lastsix[0]==lastsix[4] or lastsix[1]==lastsix[5]:
            return False
        
        
    if toggle:
        if white ==[]:
            return False
    else:
        if black ==[]:
            return False   
    return True     
chessboard = [
    [0, 0, 0, 0, 10, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0,0 , 0, 0, -3],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0,0 ,0 ,0 ,0],
    [0, 0, 0, 0, -10, 0, 0, 0]
]

def get_piece_list(chessboard):
    piece_list = []

    for row in chessboard:
        for piece in row:
            if piece != 0:
                piece_list.append(piece)

    return piece_list

pieces = get_piece_list(chessboard)
print(pieces)

def is_insufficient_material(pieces):
    # Count the number of kings and non-pawn pieces for each player
   

    # Define conditions for insufficient material
    conditions = [
        set(pieces) == {10,-10, -3, 3}, 
        set(pieces) == {10, -10},  
        set(pieces) == {10, -10, -4, 4},
        set(pieces) == {10, -10, 4}, 
        set(pieces) == {10, -10, -4},
        set(pieces) == {10, -10, -3},
        set(pieces) == {10, -10, 3},   

        # Example: Only kings and bishops on the board for a specific color
       
    ]

    # If any condition is true, it's insufficient material
    return any(conditions)

# Example usage
pieces_to_check = [10,-10,-3,3]
"""   [10,-10][10,-10,-4,4],[10,-10,-3,3][10,-10,4][10,-10,-4][10,-10,-3][10,-10,3]"""

if is_insufficient_material(pieces_to_check):
    print("The game is a draw due to insufficient material.")
else:
    print("The game continues.")

def thegame(startingposition,toggle,parameters):
    history=[]
    board=copy.deepcopy(startingposition)
    history.append(board)
    movecount=0
    iswhitekingmoved=False
    isblackkingmoved=False
    lastsix=[]
    lastsix.append(startingposition)
   
    moves = calculate_all_moves(board)
    
    if toggle:
       for row in range(8):
        for col in range(8):
            if board[row][col] == 10:  # Black king is represented by -10
                king_position = (row, col)
                break
       
       
       moves=filter_moves(board,moves,  toggle,king_position)
       moves=specialmoves(board,moves,isblackkingmoved,toggle,lastsix)
       white,black=dividebothmoves(moves)
    else:
          for row in range(8):
            for col in range(8):
              if board[row][col] == -10:  # Black king is represented by -10
                king_position = (row, col)
                break
          moves=specialmoves(board,moves,isblackkingmoved,toggle,lastsix)
          moves=filter_moves(board,moves,  toggle,king_position) 
          white,black=dividebothmoves(moves) 
   
    while(movecount<300  and( cheakmate(board,toggle,moves,white,black) and isdraw(board,toggle,white,black,lastsix))):
       if toggle:
          evaluation={}
          for moves in white:
             board_copy = [row[:] for row in board]
             board_copy= performMove(moves,board_copy)
             
             ans,_=L_model_forward(np.reshape(board_copy, (64, 1)),parameters)
             evaluation[moves]=ans
          max_keys = [key for key, value in evaluation.items() if value == max(evaluation.values())]  
          #if len(max_keys)>1:
          bestmove=max_keys[0]
          #elif max_keys==(((10,0,4,0,6),(5,0,7,0,5))) or max_keys== ((10,0,4,0,2),(5,0,0,0,3)) : 
          #     bestmove=max_keys[0]
                
         # else:
          #   bestmove=max_keys[0]
             
          #print(bestmove)     
          board=performMove(bestmove,board)
          movecount=movecount+1
          history.append(copy.deepcopy(board))
          
          
          lastsix.append(copy.deepcopy(board))
          
          if len(lastsix)>6:
              lastsix.pop(0)
          if bestmove[0]==10 or len(bestmove)==2 and bestmove[0][0]==10:
              iswhitekingmoved=True
          toggle=not toggle
          for row in range(8):
            for col in range(8):
                   if board[row][col] == -10:  # Black king is represented by -10
                     king_position = (row, col)
                     break
          moves = calculate_all_moves(board)  
                
          
          moves=filter_moves(board,moves,toggle,king_position)
          
          moves=specialmoves(board,moves,isblackkingmoved,toggle,lastsix) 
         
          white,black=dividebothmoves(moves) 
         
       else:  
           evaluation={}
           for moves in black:
             board_copy = [row[:] for row in board]
             board_copy= performMove(moves,board_copy)
             
             ans,_=L_model_forward(np.reshape(board_copy, (64, 1)),parameters)
             evaluation[moves]=ans
           min_keys = [key for key, value in evaluation.items() if value == min(evaluation.values())]  
          
           bestmove=min_keys[0]
           
             
               
           board=performMove(bestmove,board)
          # print(board)
           lastsix.append(copy.deepcopy(board))
           
           history.append(copy.deepcopy(board))
           if len(lastsix)>6:
              lastsix.pop(0)
           if bestmove[0]==-10 or len(bestmove)==2 and bestmove[0][0]==-10:
              isblackkingmoved=True
           toggle=not toggle
           for row in range(8):
             for col in range(8):
                   if board[row][col] == 10:  # Black king is represented by -10
                     king_position = (row, col)
                     break
           moves = calculate_all_moves(board)
           moves=filter_moves(board,moves,  toggle,king_position)
           moves=specialmoves(board,moves,iswhitekingmoved,toggle,lastsix) 
           white,black=dividebothmoves(moves) 
    return history

def update(X, Y, parameters, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    
    ### END CODE HERE ###
    startingposition = [[5,3,4,9,10,4,3,5],
         [1,1,1,1,1,1,1,1],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [-1,-1,-1,-1,-1,-1,-1,-1],
         [-5,-3,-4,-9,-10,-4,-3,-5]

         
         ]
   
    

        

    for b in X:
         
         
         everyposition=np.reshape(b, (64, 1))
         print(b)
        
      
    # Loop (gradient descent)
         for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = L_model_forward(everyposition, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
            cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
            grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
            parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
            
           
            costs.append(cost)
            # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate for game 1 =" + str(learning_rate))
    plt.show()   
            
    
    
    return parameters







          
          
                
             
          
         
    
        


  
                  
     
      


       
     
               
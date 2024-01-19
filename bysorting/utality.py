def possible_moves( x, y, board):
    # Initialize an empty list to store the possible moves
    moves = []
    piece_value = board[x][y]  # Get the value of the piece at the current position

    # Check the piece type and calculate possible moves accordingly
    if piece_value == -1:  # Pawn
        if x > 1 and board[x - 1][y] == 0:  # Move one square forward
            moves.append((piece_value, x, y, x - 1, y))
        if x == 6 and board[x - 2][y] == 0 and board[x - 1][y] == 0 :  # Initial two-square move
            moves.append((piece_value, x, y, x - 2, y))
        if x > 1 and y > 0 and board[x - 1][y - 1] > 0:  # Capture left
            moves.append((piece_value, x, y, x - 1, y - 1))
        if x > 1 and y < 7 and board[x - 1][y + 1] > 0:  # Capture right
            moves.append((piece_value, x, y, x - 1, y + 1))
        if x==1:
            if  board[0][y] == 0:  # Move one square forward
                 moves.append((piece_value, x, y, x - 1, y,-9))
                 moves.append((piece_value, x, y, x - 1, y,-5))
                 moves.append((piece_value, x, y, x - 1, y,-4))
                 moves.append((piece_value, x, y, x - 1, y,-3))
            if  y > 0 and board[0][y - 1] > 0:  # Capture left
                moves.append((piece_value, x, y, x - 1, y - 1,-9))
                moves.append((piece_value, x, y, x - 1, y - 1,-5))
                moves.append((piece_value, x, y, x - 1, y - 1,-4))
                moves.append((piece_value, x, y, x - 1, y - 1,-3))     
            if   y < 7 and board[0][y + 1] > 0:  # Capture right
                 moves.append((piece_value, x, y, x - 1, y + 1,-9))
                 moves.append((piece_value, x, y, x - 1, y + 1,-5))
                 moves.append((piece_value, x, y, x - 1, y + 1,-4))
                 moves.append((piece_value, x, y, x - 1, y + 1,-3))








    if piece_value == 3:  # Knight
        for dx, dy in [(1, 2), (2, 1), (-1, 2), (2, -1), (-2, 1), (1, -2), (-1, -2), (-2, -1)]:
            newx = x + dx
            newy = y + dy
            if 0 <= newx < 8 and 0 <= newy < 8:
                if board[newx][newy] <= 0:
                    moves.append((piece_value, x, y, newx, newy))

    if piece_value == 4:  # Bishop
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            newx, newy = x + dx, y + dy
            while 0 <= newx < 8 and 0 <= newy < 8:
                if board[newx][newy] == 0:
                    moves.append((piece_value, x, y, newx, newy))
                elif board[newx][newy] < 0:
                    moves.append((piece_value, x, y, newx, newy))
                    break
                else:
                    break
                newx, newy = newx + dx, newy + dy

    if piece_value == 5:  # Rook
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            newx, newy = x + dx, y + dy
            while 0 <= newx < 8 and 0 <= newy < 8:
                if board[newx][newy] == 0:
                    moves.append((piece_value, x, y, newx, newy))
                elif board[newx][newy] < 0:
                    moves.append((piece_value, x, y, newx, newy))
                    break
                else:
                    break
                newx, newy = newx + dx, newy + dy

    if piece_value == 9:  # Queen
        # Combine rook and bishop moves
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            newx, newy = x + dx, y + dy
            while 0 <= newx < 8 and 0 <= newy < 8:
                if board[newx][newy] == 0:
                    moves.append((piece_value, x, y, newx, newy))
                elif board[newx][newy] < 0:
                    moves.append((piece_value, x, y, newx, newy))
                    break
                else:
                    break
                newx, newy = newx + dx, newy + dy

    if piece_value == 10:  # King
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                newx, newy = x + dx, y + dy
                if 0 <= newx < 8 and 0 <= newy < 8 and board[newx][newy] <= 0:
                    moves.append((piece_value, x, y, newx, newy))




    if piece_value == 1:  # Pawn
        if x < 6 and board[x + 1][y] == 0:  # Move one square forward
            moves.append((piece_value, x, y, x + 1, y))
        if x == 1 and board[x + 2][y] == 0 and board[x + 1][y] == 0:  # Initial two-square move
            moves.append((piece_value, x, y, x + 2, y))
        if x < 6 and y > 0 and board[x + 1][y - 1] < 0:  # Capture left
            moves.append((piece_value, x, y, x + 1, y - 1))
        if x < 6 and y < 7 and board[x + 1][y + 1] < 0:  # Capture right
            moves.append((piece_value, x, y, x + 1, y + 1))
        if x==6:
            if board[7][y] == 0:
                 moves.append((piece_value, x, y, x + 1, y,9))
                 moves.append((piece_value, x, y, x + 1, y,5))
                 moves.append((piece_value, x, y, x + 1, y,4))
                 moves.append((piece_value, x, y, x + 1, y,3))

            if y > 0 and board[x + 1][y - 1] < 0:  # Capture left
               moves.append((piece_value, x, y, x + 1, y - 1,9))
               moves.append((piece_value, x, y, x + 1, y - 1,5))
               moves.append((piece_value, x, y, x + 1, y - 1,4))
               moves.append((piece_value, x, y, x + 1, y - 1,3))
            if  y < 7 and board[x + 1][y + 1] < 0:  # Capture right
                moves.append((piece_value, x, y, x + 1, y + 1,9))
                moves.append((piece_value, x, y, x + 1, y + 1,5))
                moves.append((piece_value, x, y, x + 1, y + 1,4))
                moves.append((piece_value, x, y, x + 1, y + 1,3))

           

    if piece_value == -3:  # Knight
        for dx, dy in [(1, 2), (2, 1), (-1, 2), (2, -1), (-2, 1), (1, -2), (-1, -2), (-2, -1)]:
            newx = x + dx
            newy = y + dy
            if 0 <= newx < 8 and 0 <= newy < 8:
                if board[newx][newy] >= 0:
                    moves.append((piece_value, x, y, newx, newy))

    if piece_value == -4:  # Bishop
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            newx, newy = x + dx, y + dy
            while 0 <= newx < 8 and 0 <= newy < 8:
                if board[newx][newy] == 0:
                    moves.append((piece_value, x, y, newx, newy))
                elif board[newx][newy] > 0:
                    moves.append((piece_value, x, y, newx, newy))
                    break
                else:
                    break
                newx, newy = newx + dx, newy + dy

    if piece_value == -5:  # Rook
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            newx, newy = x + dx, y + dy
            while 0 <= newx < 8 and 0 <= newy < 8:
                if board[newx][newy] == 0:
                    moves.append((piece_value, x, y, newx, newy))
                elif board[newx][newy] > 0:
                    moves.append((piece_value, x, y, newx, newy))
                    break
                else:
                    break
                newx, newy = newx + dx, newy + dy

    if piece_value == -9:  # Queen
        # Combine rook and bishop moves
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            newx, newy = x + dx, y + dy
            while 0 <= newx < 8 and 0 <= newy < 8:
                if board[newx][newy] == 0:
                    moves.append((piece_value, x, y, newx, newy))
                elif board[newx][newy] > 0:
                    moves.append((piece_value, x, y, newx, newy))
                    break
                else:
                    break
                newx, newy = newx + dx, newy + dy

    if piece_value == -10:  # King
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                newx, newy = x + dx, y + dy
                if 0 <= newx < 8 and 0 <= newy < 8 and board[newx][newy] >= 0:
                    moves.append((piece_value, x, y, newx, newy))

    return moves

def calculate_all_moves(board):
    all_moves = []

    for x in range(8):
        for y in range(8):
            piece_value = board[x][y]

           
            # Calculate possible moves for the piece at (x, y)
            moves = possible_moves( x, y, board)
            all_moves.extend(moves)

    return all_moves
def algebraic_to_coordinates_modified(algebraic_notation):
    rank_start=-1
    rank_end=-1
    file_start=-1
    file_end=-1
    files = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    

    piece_values = {'K': 10, 'Q': 9, 'R': 5, 'B': 4, 'N': 3}
    if  algebraic_notation=="O-O":
        return((10,0,4,0,6),(5,0,7,0,5))
    if  algebraic_notation=="O-O-O":
        return((10,0,4,0,2),(5,0,0,0,3))
    
    if len(algebraic_notation) == 2:
       file_end = files[algebraic_notation[0]]
       rank_end = int(algebraic_notation[1])-1
       piece_value =1
       
       return piece_value,  rank_start,file_start, rank_end, file_end
    if len(algebraic_notation) == 3:
         
         if algebraic_notation[2]!="+" and algebraic_notation[2]!="#" :
           file_end = files[algebraic_notation[1]]
           rank_end = int(algebraic_notation[2])-1
           piece_value = piece_values.get(algebraic_notation[0])
           return piece_value,  rank_start,file_start, rank_end, file_end
         else:
            file_end = files[algebraic_notation[0]]
            rank_end = int(algebraic_notation[1])-1
            piece_value =1
            return piece_value,  rank_start,file_start, rank_end, file_end

    if len(algebraic_notation) == 4:
        if algebraic_notation[1]=="x":
            if algebraic_notation[0].islower():
             file_end = files[algebraic_notation[2]]
             rank_end = int(algebraic_notation[3])-1
             file_start = files[algebraic_notation[0]]
             piece_value = 1
             return piece_value,  rank_start,file_start, rank_end, file_end
            else:
               file_end = files[algebraic_notation[2]]
               rank_end = int(algebraic_notation[3])-1
               piece_value = piece_values.get(algebraic_notation[0])
               return piece_value,  rank_start,file_start, rank_end, file_end 
               
        if algebraic_notation[3]=="+" or algebraic_notation[3]=="#" : 
            file_end = files[algebraic_notation[1]]
            rank_end = int(algebraic_notation[2])-1
            piece_value = piece_values.get(algebraic_notation[0]) 
            return piece_value,  rank_start,file_start, rank_end, file_end

        if  algebraic_notation[1].isdigit() and algebraic_notation[2]!="=" :
             file_end = files[algebraic_notation[2]]
             rank_end = int(algebraic_notation[3])-1
             
             
             
             rank_start = int(algebraic_notation[1])-1
             piece_value = piece_values.get(algebraic_notation[0])
            
             return piece_value,  rank_start,file_start, rank_end, file_end,
        if  algebraic_notation[1].isalpha():  
            file_end = files[algebraic_notation[2]]
            rank_end = int(algebraic_notation[3])-1
            file_start = files[algebraic_notation[1]]
            
             
            
            piece_value = piece_values.get(algebraic_notation[0]) 
            return piece_value,  rank_start,file_start, rank_end, file_end

        if  algebraic_notation[2]=="=": 
            piece_value =1
            file_end = files[algebraic_notation[0]]
            rank_end =int(algebraic_notation[1])-1
            promotion=  piece_values.get(algebraic_notation[3])
            return piece_value,  rank_start,file_start, rank_end, file_end,promotion

    if len(algebraic_notation) == 5:
        if algebraic_notation[4]=="+" or algebraic_notation[4]=="#" : 
             if algebraic_notation[1]=="x":
               if algebraic_notation[0].islower():
                  file_end = files[algebraic_notation[2]]
                  file_start = files[algebraic_notation[0]]
                  rank_end = int(algebraic_notation[3])-1
                  piece_value = 1
                  return piece_value,  rank_start,file_start, rank_end, file_end
               else:
                 file_end = files[algebraic_notation[2]]
                 rank_end = int(algebraic_notation[3])-1
                 piece_value = piece_values.get(algebraic_notation[0])
                 return piece_value,  rank_start,file_start, rank_end, file_end 
               
             if  algebraic_notation[1].isdigit() and algebraic_notation[2]!="=" :
                file_end = files[algebraic_notation[2]]
                rank_end = int(algebraic_notation[3])-1
                
                rank_start = int(algebraic_notation[1])-1
                piece_value = piece_values.get(algebraic_notation[0]) 
                return piece_value,  rank_start,file_start, rank_end, file_end
             if  algebraic_notation[1].isalpha() and algebraic_notation[1]!="x":  
                file_end = files[algebraic_notation[2]]
                rank_end = int(algebraic_notation[3])-1
                file_start = files[algebraic_notation[1]]
             
            
                piece_value = piece_values.get(algebraic_notation[0]) 
                return piece_value,  rank_start,file_start, rank_end, file_end
             if  algebraic_notation[2]=="=": 
                piece_value =1
                file_end = files[algebraic_notation[0]]
                rank_end =int(algebraic_notation[1])-1   
                promotion=  piece_values.get(algebraic_notation[3])
                return piece_value,  rank_start,file_start, rank_end, file_end,promotion
        if  algebraic_notation[2]=="x":
                piece_value = piece_values.get(algebraic_notation[0]) 
                if algebraic_notation[1].isdigit() :
                     file_end = files[algebraic_notation[3]]
                     rank_end = int(algebraic_notation[4])-1
                
                     rank_start = int(algebraic_notation[1])-1
                     return piece_value,  rank_start,file_start, rank_end, file_end
                else:
                    file_end = files[algebraic_notation[3]]
                    rank_end = int(algebraic_notation[4])-1
                
                    file_start =  files[algebraic_notation[1]]
                    return piece_value,  rank_start,file_start, rank_end, file_end

    if len(algebraic_notation) == 6:
         if algebraic_notation[5]=="+" or algebraic_notation[5]=="#" : 
              piece_value = piece_values.get(algebraic_notation[0]) 
              if algebraic_notation[1].isdigit() :
                     file_end = files[algebraic_notation[3]]
                     rank_end = int(algebraic_notation[4])-1
                
                     rank_start = int(algebraic_notation[1])-1
                     return piece_value,  rank_start,file_start, rank_end, file_end
              else:
                    file_end = files[algebraic_notation[3]]
                    rank_end = int(algebraic_notation[4])-1
                
                    file_start =  files[algebraic_notation[1]]
                    return piece_value,  rank_start,file_start, rank_end, file_end
             
         else:
             file_end = files[algebraic_notation[2]]
             file_start = files[algebraic_notation[0]]
             rank_end = int(algebraic_notation[3])-1
             piece_value = 1
             
             promotion=  piece_values.get(algebraic_notation[5])
             return piece_value,  rank_start,file_start, rank_end, file_end,promotion
    if  len(algebraic_notation) == 7 and ( algebraic_notation[6]=="+" or algebraic_notation[6]=="#"):
        file_end = files[algebraic_notation[2]]
        file_start = files[algebraic_notation[0]]
        rank_end = int(algebraic_notation[3])-1
        piece_value = 1
        promotion=  piece_values.get(algebraic_notation[5])
        return piece_value,  rank_start,file_start, rank_end, file_end,promotion
    
def split_string_into_substrings(input_string):
    # Split the input string by spaces
    substrings = input_string.split()

    # Remove numeric prefixes
    substrings = [substr.split('.', 1)[1] if '.' in substr else substr for substr in substrings]

    return substrings
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
#(10,5,6,7,8)(10,-1,-1,0,0)
def compare (move, board,toggle):
    movelist=calculate_all_moves(board)
    #movelist=filter_moves(board,movelist,toggle,(0,0))
    
   

  
    if not toggle: 
      if len(move)==5 :
          my_list = list(move)
          my_list[0]=-1*my_list[0]
          move= tuple(my_list)
      if len(move) ==6:
          my_list = list(move)
          my_list[0]=-1*my_list[0]
          print(my_list[0])
          my_list[5]=-1*my_list[5]
          print(my_list[5])
          
          move= tuple(my_list)
          
      if len(move)==2:
           my_list = list(move)
           
           
           my_list[0]=(-1 * my_list[0][0],7,my_list[0][2],7,my_list[0][4])

           my_list[1]=(-1 * my_list[1][0],7,my_list[1][2],7,my_list[1][4])
          
           move= tuple(my_list)


        

        
    if len(move)==5:

       piece_value,  rank_start,file_start, rank_end, file_end=move
       if move[0]==1 and move[2]!=-1 and board[move[3]][move[4]] ==0:
           my_list = list(move)
           my_list[1]=4
           move= tuple(my_list)
           return move
       if move[0]==-1 and move[2]!=-1 and board[move[3]][move[4]] ==0:
           my_list = list(move)
           my_list[1]=3
           move= tuple(my_list)
           return move
           
    
       for i in movelist:
          
         if len(move)==5:
          if  move[0]==i[0] :
              #print("task1 done")
              if  move[1]==i[1] or move[1]==-1 :
                  # print("task2 done")
                   if move[2]==i[2] or  move[2]==-1:
                       # print("task3 done")
                        if move[3]==i[3] or move[3]==-1:
                          #   print("task4 done")
                             if move[4]==i[4] or move[4]==-1:
                             #    print("task5 done")
                                 
                                 return i
                             
    if len(move)==6:

       piece_value,  rank_start,file_start, rank_end, file_end,promotion=move
      # (1, -1, -1, 7, 2, 3)
       for i in movelist:
          if   move[0]==i[0] :
              
              if  move[1]==i[1] or move[1]==-1 :
                   
                   if move[2]==i[2] or  move[2]==-1:
                        
                        if move[3]==i[3] or move[3]==-1:
                             
                             if move[4]==i[4] or move[4]==-1:
                                 
                                 if move[5]==i[5] :
                                 

                                   return i                    
    if len(move)==2:
          
            print(move)
            return move   

def performMove(move,board):
    if move[0]==1 and move[2]!=move[4] and board[move[3]][move[4]] ==0:
        piece, start_row, start_col, end_row, end_col = move
        board[end_row][end_col] = piece
        board[start_row][start_col] = 0
        board[start_row][end_col] =0
    if move[0]==-1 and move[2]!=move[4] and board[move[3]][move[4]] ==0:
        print(move)
        
        piece, start_row, start_col, end_row, end_col = move
        board[end_row][end_col] = piece
        board[start_row][start_col] = 0
        board[start_row][end_col] =0    


    if len(move)==6: 
        piece, start_row, start_col, end_row, end_col,promotion = move
        board[end_row][end_col] = promotion
        board[start_row][start_col] = 0  

    
    
    

     

    elif len(move)==2:
        piece, start_row, start_col, end_row, end_col = move[0]
        board[end_row][end_col] = piece
        board[start_row][start_col] = 0
        piece, start_row, start_col, end_row, end_col = move[1]
        board[end_row][end_col] = piece
        board[start_row][start_col] = 0

    else:
        piece, start_row, start_col, end_row, end_col = move
        board[end_row][end_col] = piece
        board[start_row][start_col] = 0    
    return board 
import copy
import numpy as np

def allboards(board ,notation,toggle):
   notations= split_string_into_substrings(notation)
   index=0
   boards=[]
   for i in notations:

      
      move =algebraic_to_coordinates_modified(i)
      #print(move)
      
      move=compare(move,board,toggle)
     
     
      
      board=performMove(move,board)
        
    
      

      
      toggle=not toggle
      index=index+1
      
      print(index)
      
    
      boards.append(copy.deepcopy(board))
     
   return boards   
        

    
            

        




             
             


              
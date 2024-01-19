def possible_moves( x, y, board):
    # Initialize an empty list to store the possible moves
    moves = []
    piece_value = board[x][y]  # Get the value of the piece at the current position

    # Check the piece type and calculate possible moves accordingly
    if piece_value == -1:  # Pawn
        if x > 0 and board[x - 1][y] == 0:  # Move one square forward
            moves.append((piece_value, x, y, x - 1, y))
        if x == 6 and board[x - 2][y] == 0:  # Initial two-square move
            moves.append((piece_value, x, y, x - 2, y))
        if x > 0 and y > 0 and board[x - 1][y - 1] > 0:  # Capture left
            moves.append((piece_value, x, y, x - 1, y - 1))
        if x > 0 and y < 7 and board[x - 1][y + 1] > 0:  # Capture right
            moves.append((piece_value, x, y, x - 1, y + 1))

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
        if x < 7 and board[x + 1][y] == 0:  # Move one square forward
            moves.append((piece_value, x, y, x + 1, y))
        if x == 1 and board[x + 2][y] == 0:  # Initial two-square move
            moves.append((piece_value, x, y, x + 2, y))
        if x < 7 and y > 0 and board[x + 1][y - 1] < 0:  # Capture left
            moves.append((piece_value, x, y, x + 1, y - 1))
        if x < 7 and y < 7 and board[x + 1][y + 1] < 0:  # Capture right
            moves.append((piece_value, x, y, x + 1, y + 1))

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

def count_moves_in_quadrants(moves, quadrants,board):
    counts = {'white': {quadrant: 0 for quadrant in quadrants}, 'black': {quadrant: 0 for quadrant in quadrants}}
    capturing_moves = {'white': 0, 'black': 0}

    for move in moves:
        player, start_row, start_col, end_row, end_col = move
        start_piece = board[start_row][start_col]
        end_piece = board[end_row][end_col]

        if start_piece > 0 and end_piece < 0:
            capturing_moves['white'] += 1
        elif start_piece < 0 and end_piece > 0:
            capturing_moves['black'] += 1
        piece_color = 'white' if player > 0 else 'black'

        for quadrant in quadrants:
            x_condition, y_condition = quadrants[quadrant]
            print("this is,",quadrant)
            if x_condition[0] <= end_row <= x_condition[1] and y_condition[0] <= end_col <= y_condition[1]:
                counts[piece_color][quadrant] += 1
            

    return  capturing_moves ,counts

def simulate_move(board, move):
    piece, start_row, start_col, end_row, end_col = move
    board[end_row][end_col] = piece
    board[start_row][start_col] = 0

def is_white_king_in_check(board, moves):
    white_king_position = None
    for row in range(8):
        for col in range(8):
            if board[row][col] == 10:  # White king is represented by 10
                white_king_position = (row, col)
                break

    if white_king_position is None:
        return False  # White king not found on the board

    for move in moves:
        piece, start_row, start_col, end_row, end_col = move
        if piece < 0:  # Checking if the piece is of the opponent
            if (end_row, end_col) == white_king_position:
                
                return True  # White king is in check

    return False  # White king is not in check

def is_black_king_in_check(board, moves):
    black_king_position = None
    for row in range(8):
        for col in range(8):
            if board[row][col] == -10:  # Black king is represented by -10
                black_king_position = (row, col)
                break

    if black_king_position is None:
        return False  # Black king not found on the board

    for move in moves:
        piece, start_row, start_col, end_row, end_col = move
        if piece > 0:  # Checking if the piece is of the opponent
            if (end_row, end_col) == black_king_position:
                print(move)
                return True  # Black king is in check

    return False 

def is_check_preventable(board, moves):
    for move in moves:
        if move[0]>0:
           board_copy = [row[:] for row in board]
          # Create a copy of the board
           simulate_move(board_copy, move)
           print(board_copy)

           
           macc= calculate_all_moves(board_copy)  # Simulate the move
           update=filter_moves(board_copy,macc,True)
           if not is_white_king_in_check(board_copy, update) : 
            # Check if the check is preventable
               return True  # Check is preventable

    return False  # Check is not preventable
def is_check_preventable_black(board, moves):
    for move in moves:
        if move[0] < 0:
            board_copy = [row[:] for row in board]
            # Create a copy of the board
            simulate_move(board_copy, move)  # Simulate the move
            macc = calculate_all_moves(board_copy)
            update=filter_moves(board_copy,macc,True)
            if not is_black_king_in_check(board_copy, update):
                return True  # Check is preventable

    return False  # Check is not preventable
def filter_moves(board, moves,toggle):
    newMoves=[]
   
    if toggle:
        for move in moves:
            if move[0]>0:
               board_copy = [row[:] for row in board]
          # Create a copy of the board
               simulate_move(board_copy, move)
           #print(board_copy)
               macc= calculate_all_moves(board_copy)  # Simulate the move
               if not is_white_king_in_check(board_copy, macc) : 
                  newMoves.append(move)
               #return True  # Check is preventable
    if not toggle:
          for move in moves:  
             if move[0]<0:
               board_copy = [row[:] for row in board]
          # Create a copy of the board
               simulate_move(board_copy, move)
           #print(board_copy)
               macc= calculate_all_moves(board_copy)  # Simulate the move
               if not is_black_king_in_check(board_copy, macc) : 
                      newMoves.append(move)
               #return True  # Check is preventable
               
      
    return newMoves



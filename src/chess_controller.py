#!/usr/bin/env python3
from std_msgs.msg import *
from chess_bot.srv import *
import numpy as np
import chess.uci
import chess
#import rospy
import time

#---------Globals--------------------------------------------#
game = None 
stockfish = None
old_bg = None
clock = None 

detect_chess_pieces_handle = None
chess_piece_move_handle = None

#---------Debug----------------------------------------------#
def print_(data):
    for i in range(7,-1,-1):
        print(data[i*8:i*8+8])

def send_msg(notation,new_bg,flags=None):
    s,e = convert_notation_to_index(notation)
    new_bg[s[0]+(s[1]-1)*8] = 0
    new_bg[e[0]+(e[1]-1)*8] = 2
    print_(new_bg)
    return new_bg

#---------Utilities------------------------------------------#
def update_board(old_bg,s,e):
    new_bg = old_bg.copy()
    new_bg[s[0]+(s[1]-1)*8] = 0
    new_bg[e[0]+(e[1]-1)*8] = 1
    return new_bg

def convert_index_to_notation(start,end):
    x = ["a","b","c","d","e","f","g","h"]
    start = start[0][0]
    end = end[0][0]
    s = x[start%8] + str(int(start/8)+1)
    e = x[end%8]   + str(int(end/8)+1)
    return s+e

def convert_notation_to_index(notation):
    x = ["a","b","c","d","e","f","g","h"]
    start = notation[:2]
    end = notation[2:]
    start = [x.index(start[0]),int(start[1])-1]
    end   = [x.index(end[0]),  int(end[1])-1]
    return start,end

def convert_data_to_move(old_bg,new_bg):
    new_bg = np.array(new_bg)
    old_bg = np.array(old_bg)
    start = np.where(old_bg & ~new_bg)
    end   = np.where(new_bg & ~old_bg)
    return start,end

def time_tracker(color):
    global clock
    if(color == "w"):
        clock = [[time.time(), clock[0][1]],[None, clock[1][1] - (time.time() - clock[1][0])]]
    else:
        clock = [[None, clock[0][1] - (time.time() - clock[0][0])],[time.time(), clock[1][1]]]
        print("You have",int(clock[1][1]),"seconds remaining")

#-------Start Game Functions---------------------------------#
def call_robit(notation,game):
    s,e = convert_notation_to_index(notation)
    is_cap = game.is_capture(notation)
    is_pass = game.is_en_passant(notation)
    is_castle_left = game.is_kingside_castling(notation)
    is_castle_right = game.is_queenside_castling(notation)
    is_promo = False
    if(notation[-1] == q):
        is_promo = True
        
    chess_piece_move_handle(s[0],s[1],e[0],e[1],
                            is_cap,
                            is_pass,
                            is_promo,
                            is_castle_left,
                            is_castle_right)

def robit_turn(game,stockfish,new_bg,clock):
    stockfish.position(game)
    remb = clock[0][1]
    remw = clock[1][1]
    res = stockfish.go(btime = remb*100, wtime = remw*100)
    call_robit(res.bestmove,game)
    game.push(res.bestmove)
    print(game)
    return game,new_bg

def person_turn(old_bg,new_bg,game):
    s,e = convert_data_to_move(old_bg,new_bg)
    notation = convert_index_to_notation(s,e)
    print("Person Move:",notation)
    try:
        game.push_uci(notation)
    except:
        call_robit(0,0,0,0,0,0,0,0,0,errorCode=1)
    print(game)
    return game,new_bg

def game_loop():
    global game, stockfish, old_bg, clock
    while(not game.is_game_over()):
        new_bg = recieve_msg()
        game, new_bg = person_turn(old_bg,new_bg,game)
        time_tracker("w")
        game, new_bg = robit_turn(game,stockfish,new_bg,clock)
        time_tracker("b")
        old_bg = new_bg.copy()

def recieve_msg():
    global old_bg
    notation = input("Input Move:")
    s,e = convert_notation_to_index(notation)
    new_bg = old_bg.copy()
    new_bg[s[0]+(s[1])*8] = 0
    new_bg[e[0]+(e[1])*8] = 1
    return new_bg

def main():
    global game, stockfish, old_bg, clock
    ros_init_services()
    game = chess.Board()
    stockfish = chess.uci.popen_engine("stockfish")
    stockfish.uci()
    stockfish.setoption({"skill level": 8})
    old_bg = [1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,
              2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2]
    clock = [[None, 300],[time.time(), 300]]
    print("You have 300 seconds remaining")
    game_loop()

def ros_init_services():
    global chess_piece_move_handle, detect,chess_pieces_handle
    rospy.init_node("chess_controller",anonymous=True)
    rospy.wait_for_service('chess_piece_move')
    rospy.wait_for_service('detect_chess_pieces')
    try:
        chess_piece_move_handle = rospy.ServiceProxy('chess_piece_move', ChessPieceMove)
        detect_chess_pieces_handle = rospy.ServiceProxy('detect_chess_pieces', DetectChessPieces)
    except:
        rospy.logerr("Error: Didn't get YO MAMA")

if __name__ == '__main__': main()

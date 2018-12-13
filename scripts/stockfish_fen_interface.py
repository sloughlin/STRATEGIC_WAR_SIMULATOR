#!/usr/bin/env python3
from std_msgs.msg import *
from chess_bot.srv import *
import numpy as np
import chess.uci
import chess
import rospy

#---------Globals--------------------------------------------#
game = None 
stockfish = None
old_bg = None
time = None 

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

#-------Start Game Functions---------------------------------#

def ros_publisher(startx,starty,endx,endy,
                                        is_capture,
                                        is_passant,
                                        is_promotion,
                                        is_casle_left,
                                        is_casle_right, 
                                        errorCode = None):
    pub = rospy.Publisher("chess_piece_move",ChessPieceMove)
    rospy.init_node('chess_piece_node',anonymous=True)
    msg = ChessPieceMove

    if(errorCode):
        msg.error_code = errorCode
        pub.publish(msg)
        return

    msg.start_x = startx
    msg.start_y = starty
    msg.end_x = endx
    msg.endy = endy
    msg.get_extra_queen = is_promotion
    msg.capture_piece = is_capture
    msg.castle_right = is_castle_right
    msg.castle_left = is_castle_left
    msg.enpassend = is_passant

    pub.publish(msg)

def call_robit(notation,game):
    s,e = convert_notation_to_index(notation)
    is_cap = game.is_capture(notation)
    is_pass = game.is_en_passant(notation)
    is_castle_left = game.is_kingside_castling(notation)
    is_castle_right = game.is_queenside_castling(notation)
    is_promo = False
    if(notation[-1] == q):
        is_promo = True
    ros_publisher(s[0],s[1],e[0],e[1],
                    is_cap,
                    is_pass,
                    is_promo,
                    is_castle_left,
                    is_castle_right)


def robit_turn(game,stockfish,new_bg,time):
    stockfish.position(game)
    res = stockfish.go(btime = time[0], wtime = time[1])
    call_robit(res.bestmove,game)
    game.push(res.bestmove)
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

def game_loop(new_bg):
    global game, stockfish, old_bg, time
    if(not game.is_game_over()):
        game, new_bg = person_turn(old_bg,new_bg,game)
        game, new_bg = robit_turn(game,stockfish,new_bg,time)
        old_bg = new_bg.copy()

def recieve_msg(notation):
    global old_bg
    s,e = convert_notation_to_index(notation)
    new_bg = old_bg.copy()
    new_bg[s[0]+(s[1]-1)*8] = 0
    new_bg[e[0]+(e[1]-1)*8] = 1
    game_loop(new_bg)

def globalise_time(data):
    global time
    time = (data.btime,data.wtime)#btime, wtime

def main():
    global game, stockfish, old_bg, time
    game = chess.Board()
    stockfish = chess.uci.popen_engine("stockfish")
    stockfish.uci()
    stockfish.setoption({"skill level": 0})
    old_bg = [1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,
              2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2]
    #time = (30000,30000)
    rospy.wait_for_service("time")
    rospy.wait_for_service("player_move")
    ros_listener()

def ros_listener():
    rospy.init_node("chess_server",anonymous=True)
    s = rospy.Service("chess_fen_interface",ChessNotation.srv.notation, recieve_msg)
    #rospy.ServiceProxy("time",ChessTime,globalise_time)
    #rospy.ServiceProxy("player_move",ChessNotation,recieve_msg)
    rospy.spin()

if __name__ == '__main__': main()

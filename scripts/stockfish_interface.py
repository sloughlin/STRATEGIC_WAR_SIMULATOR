#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16MultiArray
from chess_bot.srv import *
import chess
import chess.uci
import numpy as np

#---------Globals--------------------------------------------#
game = None 
stockfish = None
old_bg = None
time = None #Uninitialized

#---------Debug----------------------------------------------#
def print_(data):
    for i in range(7,-1,-1):
        print(data[i*8:i*8+8])

def recieve_msg(old_bg):
    notation = input("Input Move:")
    s,e = convert_notation_to_index(notation)
    new_bg = old_bg.copy()
    new_bg[s[0]+(s[1]-1)*8] = 0
    new_bg[e[0]+(e[1]-1)*8] = 1
    return new_bg

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
    start = [x.index(start[0]),int(start[1])]
    end   = [x.index(end[0]),  int(end[1])]
    return start,end

def convert_data_to_move(old_bg,new_bg):
    new_bg = np.array(new_bg)
    old_bg = np.array(old_bg)
    start = np.where(old_bg & ~new_bg)
    end   = np.where(new_bg & ~old_bg)
    return start,end

#-------Start Game Functions---------------------------------#

def ros_publisher(startx,starty,endx,endy,
                                        is_capture=False,
                                        is_pessant=False,
                                        is_promotion=False,
                                        is_casle_left=False,
                                        is_casle_right=False, 
                                        errorCode = None):
    pub = rospy.Publisher("chess_piece_move",ChessPieceMove)
    rospy.init_node('chess_piece_node',anonymous=True)
    msg = ChessPieceMove
    msg.whatever = stuff#todo

    pub.publish(msg)

def robit_turn(game,stockfish,new_bg,time):
    stockfish.position(game)
    res = stockfish.go(btime = time[0], wtime = time[1]) #Will time be in M:S or milliseconds. Needs to be passed as ms
    game.push(res.bestmove)
    print("Robit Move:",res.bestmove)
    print(game)
    new_bg = send_msg(res.bestmove.uci(),new_bg)
    return game,new_bg

def person_turn(old_bg,new_bg,game):
    s,e = convert_data_to_move(old_bg,new_bg)
    new_bg = update_board(old_bg,s,e)
    try:
        notation = convert_index_to_notation(s,e)
        print("Person Move:",notation)
    except:
        new_bg = update_board(old_bg)
        game = person_turn(old_bg,new_bg,game)
        return game
    try:
        game.push_uci(notation)
    except:
        send_msg(notation[2:]+notation[:2],old_bg,"Illegal")
    print_(new_bg)
    print(game)
    return game,new_bg

def game_loop(data):
    global game, stockfish, old_bg, time
    if(not game.is_game_over()):
        game, new_bg = person_turn(old_bg,game)
        game, new_bg = robit_turn(game,stockfish,new_bg,time)
        old_bg = new_bg.copy()

def globalise_time(data):
    global time
    time = (data[0],data[1])#btime, wtime

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
    time = (30000,30000)
    ros_listener()

def ros_listener():
    rospy.init_node("ros_recieve_msg",anonymous=True)
    #rospy.Subscriber("time",Int32, globalise_time) #To be determined
    #rospy.Subscriber("board_state",Int16MultiArray,game_loop)
    rospy.spin()

if __name__ == '__main__': main()

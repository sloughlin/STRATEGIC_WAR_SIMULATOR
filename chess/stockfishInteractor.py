import rospy
import chess
import chess.uci
import numpy as np

def print_(data):
    for i in range(7,-1,-1):
        print(data[i*8:i*8+8])
#--------------------------------------------
def recieve_msg(old_bg):
    notation = input("Input Move:")
    s,e = convert_notation_to_index(notation)
    new_bg = old_bg.copy()
    new_bg[s[0]+(s[1]-1)*8] = 0
    new_bg[e[0]+(e[1]-1)*8] = 1
    return new_bg

def listener():
    rospy.init_node('chess_listener')
    rospy.Subscriber('chess_board',bg,callback)

def send_msg(notation,new_bg,flags=None):
    s,e = convert_notation_to_index(notation)
    new_bg[s[0]+(s[1]-1)*8] = 0
    new_bg[e[0]+(e[1]-1)*8] = 2
    print_(new_bg)
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

def robit_turn(game,stockfish,new_bg):
    stockfish.position(game)
    res = stockfish.go()
    game.push(res.bestmove)
    print("Robit Move:",res.bestmove)
    print(game)
    new_bg = send_msg(res.bestmove.uci(),new_bg)
    return game,new_bg

def person_turn(old_bg,new_bg,game):
    s,e = convert_data_to_move(old_bg,new_bg)
    try:
        notation = convert_index_to_notation(s,e)
        print("Person Move:",notation)
    except:
        new_bg = recieve_msg(old_bg)
        game = person_turn(old_bg,new_bg,game)
        return game
    try:
        game.push_uci(notation)
    except:
        send_msg(notation[2:]+notation[:2],old_bg,"Illegal")
    print_(new_bg)
    print(game)
    return game

def start_game():
    game = chess.Board()
    stockfish = chess.uci.popen_engine("stockfish")
    stockfish.uci()
    stockfish.setoption({"skill level": 0})
    return game,stockfish

def main():
    game,stockfish = start_game()
    old_bg = [1,1,1,1,1,1,1,1,
              1,1,1,1,1,1,1,1,
              0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,
              2,2,2,2,2,2,2,2,
              2,2,2,2,2,2,2,2]

    while(not game.is_game_over()):
        bg = recieve_msg(old_bg)
        game = person_turn(old_bg,bg,game)
        game, bg = robit_turn(game,stockfish,bg)
        old_bg = bg.copy()


if __name__ == '__main__': main()

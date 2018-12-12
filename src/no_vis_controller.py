#!/usr/bin/env python
from chess_bot.srv import *
import rospy

# setup local program management
is_running = True

# setup clock service

# setup ros
rospy.init_node('chess_bot_no_vis_controller', anonymous=True)


while(is_running):
    # check for exit conditions: win/lose/draw
    # wait for move from terminal
    move_str = raw_input("Input Move:")
 
    # solve for next move
    rospy.wait_for_service('stockfish_fen_interface')
    try: 
        stockfish_fen_interface = rospy.ServiceProxy('stockfish_fen_interface', )
 
    # notify on illegal move input

    # dispatch move

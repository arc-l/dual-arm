import math
import numpy as np
import random
import copy
import math
import json
import random

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(sys.path)
from util import poly_disk, isCollisionFree



def load_arrangement( numObjs, Density, overlap_rate, Trial):
    file_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'arrangements',
        'D='+str(Density), 
        'n='+str(numObjs),
        'overlap='+str(overlap_rate),
        str(Trial)+'_'+str(numObjs)+'_'+str(Density)+'_'+str(overlap_rate)+'.json'
    )
    with open(file_name, 'r') as f:
        data = json.load(f)
    point_list = data['point_list']
    arr = {
        k:point_list[k] for k in range(numObjs)
    }
    return arr

def load_handoff_arrangement( numObjs, Density, handoff_number, Trial):
    file_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'handoff_arrangements',
        'D='+str(Density), 
        'n='+str(numObjs),
        'Handoff='+str(handoff_number),
        str(Trial),
        'start.json'
    )
    with open(file_name, 'r') as f:
        data = json.load(f)
    point_list = data['point_list']
    start_arr = {
        k:point_list[k] for k in range(numObjs)
    }

    file_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'handoff_arrangements',
        'D='+str(Density), 
        'n='+str(numObjs),
        'Handoff='+str(handoff_number),
        str(Trial),
        'goal.json'
    )
    with open(file_name, 'r') as f:
        data = json.load(f)
    point_list = data['point_list']
    goal_arr = {
        k:point_list[k] for k in range(numObjs)
    }
    return start_arr, goal_arr



if __name__ == '__main__':
    batch_generation()
    # batch_handoff_instances_generation()
    # pass
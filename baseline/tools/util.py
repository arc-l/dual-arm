import math
from math import sqrt, radians, sin, cos
from collections import deque
import numpy as np
from numpy import power, subtract, add, savetxt, array
from numpy.random import choice
from shapely.geometry import Polygon
from random import shuffle
import os
from os import makedirs, mkdir
from os.path import join, abspath, dirname
from ujson import loads, dumps
import sys
import pickle

from baseline.constants import REAL_CENTER, SIM_CENTER, WORKSPACE_SIZE
from baseline.motion_planning.utils import sim_pose_to_env_pose, env_pose_to_sim_pose

Inf = float('inf')

class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

    def red(text):
        return f"{Colors.RED}{text}{Colors.RESET}"
    def green(text):
        return f"{Colors.GREEN}{text}{Colors.RESET}"
    def yellow(text):
        return f"{Colors.YELLOW}{text}{Colors.RESET}"
    def blue(text):
        return f"{Colors.BLUE}{text}{Colors.RESET}"
    def magenta(text):
        return f"{Colors.MAGENTA}{text}{Colors.RESET}"
    def cyan(text):
        return f"{Colors.CYAN}{text}{Colors.RESET}"
    def white(text):
        return f"{Colors.WHITE}{text}{Colors.RESET}"
    

def norm(u):
    return sqrt(math.sum(power(u, 2)))


def dist(a, b):
    return norm(subtract(a, b))


### find the bounding box for a polygon ###
def bbox(poly):
    xmin = Inf
    xmax = -Inf
    ymin = Inf
    ymax = -Inf
    for p in poly:
        xmin = min(xmin, p[0])
        xmax = max(xmax, p[0])
        ymin = min(ymin, p[1])
        ymax = max(ymax, p[1])
    return xmin, xmax, ymin, ymax


### build the shape of the object based on resolution ###
def poly_disk(center, radius, resolution):
    poly = []
    for i in range(360, 0, -resolution):
        rad = radians(i)
        x = center[0] + radius * cos(rad)
        y = center[1] + radius * sin(rad)
        poly.append([x, y])
    return poly

def poly_stick(center, direction, length, width):
    '''
    Input: 
    center: (x,y)
    direction: radian
    length
    width

    Output:
    poly: point list
    '''
    poly = []
    diameter = math.sqrt(length**2+width**2)

    dtheta = math.asin(diameter/width)

    direction_list = [
        direction-dtheta,
        direction+dtheta,
        -direction-dtheta,
        -direction+dtheta
    ]

    for d in direction_list:
        x = center[0] + diameter/2.0*math.cos(d)
        y = center[1] + diameter/2.0*math.sin(d)
        poly.append([x,y])
    return poly


### check if two polygons overlap ###
def polysCollide(poly1, poly2):
    cpoly1 = Polygon(poly1)
    cpoly2 = Polygon(poly2)
    return cpoly1.overlaps(cpoly2)


### collision checker2 ###
def collisionCheck(objects):
    for i in range(len(objects) - 1):
        for poly in objects[i + 1:]:
            if polysCollide(poly, objects[i]):
                return False
    return True


### collision checker ###
def isCollisionFree(object_shape, center, obstacles):
    object_location = add(center, object_shape)
    ### Now check if the new object collides with all other objects (as obstacles)
    for poly in obstacles:
        if polysCollide(poly, object_location):
            return False

    return True


### collision checker with buffers ###
def countNumOverlap(object_shape, center, obstacles, buffer_obs, numOverlapAllowed):
    buffer_polygon = add(center, object_shape)
    ### Now check if the new buffer collides with all other objects (as obstacles)
    numOverlap = 0
    for poly in obstacles:
        if polysCollide(poly, buffer_polygon):
            numOverlap += 1
        if (numOverlap > numOverlapAllowed):
            return numOverlap
    for poly in buffer_obs:
        if polysCollide(poly, buffer_polygon):
            numOverlap += 3
        if (numOverlap > numOverlapAllowed):
            return numOverlap
    ### reach here since numOverlap <= numOverlapAllowed
    return numOverlap


### BFS search on the connectivity graph ###
def bfs(tree, start, goal):
    path = []
    backtrace = {start: start}
    explore = deque([start])
    while goal not in backtrace:
        try:
            p = explore.pop()
        except Exception:
            return []
        for c in tree[p]:
            if c not in backtrace:
                backtrace[c] = p
                explore.append(c)

    path.append(goal)
    while backtrace[path[-1]] != path[-1]:
        path.append(backtrace[path[-1]])
    path.reverse()

    return path


def BFS(graphAdj, start, goal, condition=lambda x: True):
    path = []
    backtrace = {start: start}
    explore = deque([start])
    while goal not in backtrace:
        if len(explore) > 0:
            u = explore.pop()
        else:
            return []
        for v in filter(condition, graphAdj[u]):
            if v not in backtrace:
                backtrace[v] = u
                explore.appendleft(v)

    path.append(goal)
    while backtrace[path[-1]] != path[-1]:
        path.append(backtrace[path[-1]])
    path.reverse()

    return path


def checkBitStatusAtPos(n, k):
    #This code is contributed by Gitanjali (GeeksforGeeks)
    new_num = n >> k
    return (new_num & 1)


def generate_instance(numObjs, Density, IDs):
    # instance_num = 200
    # instances_choice = random.choice(range(instance_num),size=2, replace=False)
    instances_choice = IDs
    # print "arrangement choices: ", instances_choice
    my_path = abspath(dirname(dirname(__file__))) + "/arrangements"
    init_list = range(numObjs)
    final_list = range(numObjs)
    # random.shuffle(init_list)
    # random.shuffle(final_list)

    start_arr = {}
    goal_arr = {}

    ### init arr ###
    f = open( my_path+"/D="+str(round(Density,1))+"/n="+str(numObjs)
    +"/"+str(instances_choice[0])+"_"+str(numObjs)+"_"+str(round(Density,1))
    +".txt", 'rb')
    object_index = 0
    for line in f.readlines():
        line = line.decode('utf-8')
        if line == "objects\n":
            continue
        else:
            pose = line.split()
            start_arr[init_list[object_index]] = (float(pose[0]), float(pose[1]))
            object_index += 1
            # print pose
    f.close()

    ### final arr ###
    f = open( my_path+"/D="+str(round(Density,1))+"/n="+str(numObjs)
    +"/"+str(instances_choice[1])+"_"+str(numObjs)+"_"+str(round(Density,1))
    +".txt", 'rb')
    object_index = 0
    for line in f.readlines():
        line = line.decode('utf-8')
        if line == "objects\n":
            continue
        else:
            pose = line.split()
            goal_arr[final_list[object_index]] = (float(pose[0]), float(pose[1]))
            object_index += 1
            # print pose
    f.close()
    
    return start_arr, goal_arr


def write_a_plan(instance_info, plan, labeltxt):
    folder = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        "plans",
        labeltxt
    )
    if not os.path.exists(folder):
        os.makedirs(folder)
    added_text = ''
    for t in (instance_info[1:5]):
        added_text += str(t) + '_'
    
    with open(
            os.path.join(
                folder, added_text+'.pkl'
            ),'wb'
        ) as output:
        pickle.dump(plan, output, pickle.HIGHEST_PROTOCOL)


def write_a_traj(instance_info, traj, labeltxt):
    folder = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        "trajs",
        labeltxt
    )
    if not os.path.exists(folder):
        os.makedirs(folder)
    added_text = ''
    for t in (instance_info[1:5]):
        added_text += str(t) + '_'
    
    with open(
            os.path.join(
                folder, added_text+'.pkl'
            ),'wb'
        ) as output:
        pickle.dump(traj, output, pickle.HIGHEST_PROTOCOL)


def read_a_traj(instance_info, labeltxt):
    folder = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        "trajs",
        labeltxt
    )
    added_text = ''
    for t in (instance_info[1:5]):
        added_text += str(t) + '_'
    with open(
            os.path.join(
                folder, added_text+'.pkl'
            ), 'rb'
        ) as input_:
        traj = pickle.load(input_)
    return traj


def write_title(experimentID, title, labeltxt):
    folder = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        "logs",
        labeltxt
    )
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    log = os.path.join(folder, experimentID + '.txt')
    with open(log, 'a') as f:
        f.write(title+'\n')

def write_to_log(instance_info, result_data, labeltxt):
    folder = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        "logs",
        labeltxt
    )
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    experimentID = instance_info[0]
    log = os.path.join(folder, experimentID + '.txt')

    added_text = ''
    for t in (instance_info[1:] + result_data):
        added_text += str(t) + ' '
    added_text += '\n'

    with open(log, 'a') as f:
        f.write(added_text)

def set2tuple(s):
    return tuple(sorted(list(s)))


def deepcopy_dict(d):
    k = list(d.keys())[0]
    t = type(d[k])
    assert isinstance(d[k], (float, int, str, list, tuple))
    return { k: loads(dumps(d[k])) for k in d.keys()}


def deepcopy_graph(G):
    new_G = {}
    for k in G.keys():
        new_G[k] = deepcopy_set(G[k])
    return new_G


def deepcopy_set(s):
    return set(list(s))


def construct_DG_cylinder(start_arr, goal_arr, radius):
    '''
    each pose: ( x, y)
    '''
    DG = {}
    for goal_obj, goal_center in goal_arr.items():
        DG[goal_obj] = set()
        for start_obj, start_center in start_arr.items():
            if start_obj == goal_obj:
                continue
            if (math.sqrt(
                (start_center[0]-goal_center[0])**2+
                (start_center[1]-goal_center[1])**2
                ) <= 2*radius):
                DG[goal_obj].add(start_obj)
    return DG


def construct_DG(start_arr, goal_arr, radius):
    '''
    each pose: (x, y)
    square size is 2*radius (side length), so half-width is radius
    '''
    DG = {}
    for goal_obj, goal_center in goal_arr.items():
        DG[goal_obj] = set()
        for start_obj, start_center in start_arr.items():
            if start_obj == goal_obj:
                continue
            # AABB collision check for squares
            if (abs(start_center[0] - goal_center[0]) < 2 * radius and
                abs(start_center[1] - goal_center[1]) < 2 * radius):
                DG[goal_obj].add(start_obj)
    return DG


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def get_real_arrangement(text='test', camera=None):
    '''
    detect objects and save the arrangement as a txt file
    '''
    object_poses = camera.update_object_poses()
    sorted_obj_ids = sorted(object_poses.keys())
    with open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),text+'.txt'),'w') as f:
        for obj_id in sorted_obj_ids:
            f.write(str(obj_id)+' '+str(object_poses[obj_id][0][0])+' '+str(object_poses[obj_id][0][1])+' '+str(object_poses[obj_id][0][2])+'\n')



def load_real_arrangement(text='test'):
    '''
    load the arrangement from a txt file
    '''
    with open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),text+'.txt'),'r') as f:
        lines = f.readlines()
    object_poses = {}
    for line in lines:
        obj_id, x, y, z = line.split(' ')
        object_poses[int(obj_id)] = [float(x), float(y), float(z)]
    return object_poses

def real_env_to_plan_env(r, overlap_rate=1.0):
    '''
    real: centered at REAL_CENTER in the coordinate of robot 0
    '''
    # real->sim
    s = real_env_to_sim_env(r, overlap_rate)
    # sim->planning
    return sim_pose_to_env_pose(s, overlap_rate)
    
def plan_env_to_real_env(p, overlap_rate=1.0):
    '''
    p: centered at REAL_CENTER in the coordinate of robot 0
    '''
    # planning->sim
    s = env_pose_to_sim_pose(p, overlap_rate)
    # sim->real
    return sim_env_to_real_env(s, overlap_rate)

    
def sim_env_to_real_env(s, overlap_rate=1.0):
    '''
    s: sim env point in sim world coordinate
    '''
    s = np.array(s[:2]).reshape(2)    
    # robot 0 center in sim env
    robot0_position = np.array((
        0.25-WORKSPACE_SIZE/(1+overlap_rate),
        0.0, 0.0
    ))
    # s in sim robot 0 coordinate
    s = s - robot0_position[:2]
    
    # compute transformation
    # sim env center in sim robot 0 coordinate
    s_env = np.array([
        SIM_CENTER[0]-robot0_position[0],
        SIM_CENTER[1]-robot0_position[1]
    ])
    rotation_mat = np.array([
        [0, 1],
        [-1, 0]
    ])
    translation_vec = np.array([
        s_env[1]+REAL_CENTER[0],
        -s_env[0]+REAL_CENTER[1]
    ])
    # s in real env in real robot 0 coordinate
    r = np.matmul(rotation_mat, s) + translation_vec
    return (r[0], r[1], 0.355)


def real_env_to_sim_env(r, overlap_rate=1.0):
    '''
    r: real env point in real robot 0 coordinate,
    it can be [x,y] or [x,y,z]
    real -> sim: rotate -pi/2
    '''
    # real env center in real robot 0 coordinate
    r = np.array(r[:2]).reshape(2)
    # robot 0 center in sim env
    robot0_position = (
        0.25-WORKSPACE_SIZE/(1+overlap_rate),
        0.0, 0.0
    )
    # compute transformation
    # sim env center in sim robot 0 coordinate
    s_env = np.array([
        SIM_CENTER[0]-robot0_position[0],
        SIM_CENTER[1]-robot0_position[1]
    ])
    rotation_mat = np.array([
        [0, 1],
        [-1, 0]
    ])
    # dual to the rotation
    translation_vec = np.array([
        s_env[0]-REAL_CENTER[1],
        s_env[1]+REAL_CENTER[0]
    ])
    # pt in sim env in sim robot 0 coordinate   
    s = np.matmul(rotation_mat, r) + translation_vec
    # pt in sim env in sim world coordinate
    s = s + robot0_position[:2]
    return (s[0], s[1], 0.023000000044703484)
    
    
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', printEnd="\r"):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: 
        print()  


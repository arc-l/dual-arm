import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import Polygon
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tools.util import poly_disk

offset = 50
robot_speed = 100
Timeout = 1.0



class robot_arm(object):
    def __init__(self, arm_id, Width, Height, start_arr, goal_arr, radius, task_list, speed = 100.0) -> None:
        self.arm_id = arm_id
        self.start_arr = start_arr
        self.goal_arr = goal_arr
        self.Width = Width
        self.Height = Height
        self.radius = radius
        self.current_object = None
        self.rest = regional_rest(arm_id, Width,Height, radius)
        self.buffer = self.rest
        self.current_position = self.rest
        self.speed = speed
        self.task_list = task_list
        self.current_goal = self.rest
        self.stop = False # finished or handoff
        self.wait = False # execution time of grasps and releases
        self.start = None
        self.obstacles = set()

        self.task_list = [ t for t in self.task_list if t != None]
        

        # prepare for position list
        self.construct_path_list()

    def update_obstacles(self, obstacles):
        self.obstacles = obstacles

    def construct_path_list(self):
        self.path_list = []
        for obj, p1, p2 in self.task_list:
            places = []
            for p in [p1, p2]:
                if p == 'H':
                    places.append(
                        (self.Width/2, self.Height/2)
                    )
                else:
                    places.append(p)
            self.path_list.append((None, places[0]))
            self.path_list.append((obj, places[1]))
        self.path_list.append((None, self.rest))

    def start_waiting(self):
        self.wait = True
        self.start = time.time()

    def check_waiting(self):
        if (time.time()-self.start)>Timeout:
            self.wait = False
            self.start = None
            return True
        return False

    def update(self):
        if self.wait: # grasp and release time
            if self.check_waiting():
                self.current_position = self.update_position()
            return
        if self.stop: # finished or waiting for handoffs
            return
        if np.linalg.norm(
            np.array(self.current_position) - np.array(self.current_goal)
        ) >= 0.0001: # need to move
            self.current_position = self.update_position()
        elif len(self.path_list) == 0: # finished
            self.stop = True
            return
        else: # move to the destination
            if self.check_reachability():
                self.current_object, self.current_goal = self.path_list.pop(0)
                self.start_waiting()
            else:
                return # yield_to_obstacles
        return 

    def check_reachability(self):
        if self.current_object == None: # transitting
            return True
        if (self.current_goal[0]<0) or (self.current_goal[0]>self.Width) or (self.current_goal[1]<0) or (self.current_goal[1]>self.Height):
            # momved to a buffer
            return True
        if tuple(self.current_goal) == (self.Width/2, self.Height/2):
            return True
        CollisionFree = True
        for obs in self.obstacles:
            if np.linalg.norm(
                np.array(self.current_goal) - np.array(obs)
                ) <= 2 * self.radius:
                CollisionFree = False
                print('robot ' + str(self.arm_id) + ' waiting for removal')
                return False
        if CollisionFree:
            return True

    def update_position(self):
        dist = np.linalg.norm(
            np.array(self.current_goal) - np.array(self.current_position)
        )
        if dist <= self.speed:
            return self.current_goal
        else:
            return (
                self.speed/dist*(self.current_goal[0]-self.current_position[0])+self.current_position[0],
                self.speed/dist*(self.current_goal[1]-self.current_position[1])+self.current_position[1]
                )
    
def setupPlot(HEIGHT, WIDTH, num_arms):
    # fig = plt.figure(num=None, figsize=(int(5 * WIDTH / HEIGHT)+1, 5+1), dpi=120, facecolor='w', edgecolor='k')
    fig = plt.figure( figsize=(10,7), facecolor='w', edgecolor='k')
    ax = fig.subplots()
    return fig, ax

def createPolygonPatch(polygon, color, zorder=1):
    verts = []
    codes = []
    for v in range(0, len(polygon)):
        verts.append(polygon[v])
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=color, lw=0.5, zorder=zorder)
    return patch


def createPolygonPatch_distinct(polygon, color, isGoal, zorder=1, alpha=1):
    verts = []
    codes = []
    for v in range(0, len(polygon)):
        verts.append(polygon[v])
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)

    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    if isGoal:
        patch = patches.PathPatch(path, linestyle='--', edgecolor=color, facecolor="white", lw=2, zorder=zorder, alpha = alpha)
    else:
        patch = patches.PathPatch(path, facecolor=color, lw=2, zorder=zorder, alpha=alpha)

    return patch


def getColorMap(obj_list):
    # set colormap
    gist_ncar = plt.get_cmap('gist_ncar')
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=gist_ncar)

    color_values = np.linspace(0.1, 0.9, len(obj_list))
    color_pool = {} 
    for ii, obj in enumerate(obj_list):
        color_pool[obj] = scalarMap.to_rgba(color_values[ii])

    return color_pool


def plan_animation(HEIGHT, WIDTH, start_arr, goal_arr, radius, action_list, num_arms=2, overlap_rate=0.3):
    ### polygon shape
    objectShape = np.array(poly_disk([0, 0], radius, 3))

    obj_list = sorted(list(start_arr.keys()))
    current_pts = {
        k:p for k,p in start_arr.items()}
    goal_pts = {
        k:p for k,p in goal_arr.items()}
    current_obstacles = set([tuple(p) for k,p in start_arr.items()])

    
    ### get some colors from colormap
    color_pool = getColorMap(obj_list)

    ### set the canvas
    fig, ax = setupPlot(HEIGHT, WIDTH, num_arms)
    wallx1 = [0, WIDTH, WIDTH, 0, 0]
    wally1 = [0, 0, HEIGHT, HEIGHT, 0]

    xmin = -350
    xmax = WIDTH+350
    ymin = -50
    ymax = HEIGHT+50

    # offset = 100
    # plt.xlim([ -2*radius-offset, WIDTH+2*radius+offset])
    # plt.ylim([ -2*radius-offset, HEIGHT+2*radius+offset])
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    ax.plot(wallx1, wally1, 'blue')
    ax.plot([0.5*WIDTH*(1-overlap_rate), 0.5*WIDTH*(1-overlap_rate)], [0,HEIGHT], 'black', ls='--')
    ax.plot([0.5*WIDTH*(1+overlap_rate), 0.5*WIDTH*(1+overlap_rate)], [0,HEIGHT], 'black', ls='--')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show(block=False)
    # plt.show()
    ### give a prompt to start animation
    # raw_input("Press <ENTER> to start animation")
    input("Press <ENTER> to start animation")
    # return

    arms = {k:robot_arm(k, WIDTH, HEIGHT, start_arr, goal_arr, radius, action_list[k], speed = robot_speed) for k in range(num_arms)}

    for id, arm in arms.items():
        print(id, arm.path_list)
    # return


    while is_executing(arms):
        handoff_list = []
        manipulating_objs = []
        for arm in arms.values():
            arm.update()
            obj = arm.current_object
            if obj != None:
                current_pts[obj] = arm.current_position
                manipulating_objs.append(obj)
            if is_handoff(arm.current_position, WIDTH, HEIGHT, radius):
                arm.stop = True
                handoff_list.append(arm.arm_id) # wait for the other one
        current_obstacles = set([tuple(p) for k,p in current_pts.items() if k not in manipulating_objs])
        for arm in arms.values():
            arm.update_obstacles(current_obstacles)
        if len(handoff_list) >= 2:  # handoff succeed
            for arm in arms.values():
                arm.stop = False
        
        ax.cla()
        ax.plot(wallx1, wally1, 'blue')
        ax.plot([0.5*WIDTH*(1-overlap_rate), 0.5*WIDTH*(1-overlap_rate)], [0,HEIGHT], 'black', ls='--')
        ax.plot([0.5*WIDTH*(1+overlap_rate), 0.5*WIDTH*(1+overlap_rate)], [0,HEIGHT], 'black', ls='--')
    

        ### plot the gripper
        for arm in arms.values():
            polygon = Polygon(objectShape + np.array(arm.current_position))
            patch = createPolygonPatch_distinct(list(polygon.exterior.coords), 'darkgray', isGoal=False, zorder=30, alpha = 0.7)
            ax.add_patch(patch)
            ax.text(arm.current_position[0], arm.current_position[1], 'R'+str(arm.arm_id), fontweight='bold', fontsize=16, zorder=100)

        ### plot the objects
        for obj in current_pts.keys():
            if obj in manipulating_objs:
                current_bonus = 10
            else:
                current_bonus = 0
            ### current
            polygon = Polygon(objectShape + current_pts[obj])
            patch = createPolygonPatch_distinct(list(polygon.exterior.coords), color_pool[obj], isGoal=False, zorder=3+current_bonus)
            ax.add_patch(patch)
            ax.text(current_pts[obj][0], current_pts[obj][1], str(obj), fontweight='bold', fontsize=16, zorder=3+current_bonus)
            ### goal
            polygon = Polygon(objectShape + goal_pts[obj])
            patch = createPolygonPatch_distinct(
                list(polygon.exterior.coords), color_pool[obj], isGoal=True, zorder=1
            )
            ax.add_patch(patch)
            ax.text(goal_pts[obj][0], goal_pts[obj][1], str(obj), fontweight='bold', fontsize=16, zorder=1)
        # plt.xlim([ -2*radius-offset, WIDTH+2*radius+offset])
        # plt.ylim([ -2*radius-offset, HEIGHT+2*radius+offset])
        ax.set_xlim((xmin,xmax))
        ax.set_ylim((ymin,ymax))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(0.0005)
    
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    return

def is_handoff(pose, Width, Height, radius):
    return (np.linalg.norm(
        np.array(pose)-np.array([Width/2, Height/2])
    ) <= 0.0001)

def is_executing(arms):
    for arm in arms.values():
        if not arm.stop:
            return True
    else:
        return False

def regional_rest(arm_id, Width, Height, radius):
    if arm_id == 0:
        return (-offset-radius, Height/2)
    else:
        return (Width+offset+radius, Height/2)

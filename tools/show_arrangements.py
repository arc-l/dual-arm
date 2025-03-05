import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.path import Path
import matplotlib.patches as patches
from shapely.geometry import Polygon
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tools.util import poly_disk

offset = 50

def setupPlot(HEIGHT, WIDTH, num_arms):
    # fig = plt.figure(num=None, figsize=(int(5 * WIDTH / HEIGHT)+1, 5+1), dpi=120, facecolor='w', edgecolor='k')
    fig = plt.figure(figsize=(10,7), facecolor='w', edgecolor='k')
    ax = fig.subplots()
    plt.gca().set_aspect('equal', adjustable='box')
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


def show_arrangements(HEIGHT, WIDTH, start_arr, goal_arr, radius, num_arms=2):
    # TODO only for two arms
    ### polygon shape
    objectShape = np.array(poly_disk([0, 0], radius, 3))

    obj_list = sorted(list(start_arr.keys()))
    current_pts = {
        k:p for k,p in start_arr.items()}
    goal_pts = {
        k:p for k,p in goal_arr.items()}
    
    ### get some colors from colormap
    color_pool = getColorMap(obj_list)

    ### set the canvas
    fig, ax = setupPlot(HEIGHT, WIDTH, num_arms)
    wallx1 = [0, WIDTH, WIDTH, 0, 0]
    wally1 = [0, 0, HEIGHT, HEIGHT, 0]
    ax.plot(wallx1, wally1, 'blue')
    # offset = 100
    # plt.xlim([ -2*radius-offset, WIDTH+2*radius+offset])
    # plt.ylim([ -2*radius-offset, HEIGHT+2*radius+offset])

    ### give a prompt to start animation
    # raw_input("Press <ENTER> to start animation")

    ### plot the objects
    for obj in current_pts.keys():
        ### current
        polygon = Polygon(objectShape + current_pts[obj])
        patch = createPolygonPatch_distinct(list(polygon.exterior.coords), color_pool[obj], isGoal=False, zorder=3)
        ax.add_patch(patch)
        # ax.text(current_pts[obj][0], current_pts[obj][1], str(obj), fontweight='bold', fontsize=16, zorder=3)
        ### goal
        polygon = Polygon(objectShape + goal_pts[obj])
        patch = createPolygonPatch_distinct(
            list(polygon.exterior.coords), color_pool[obj], isGoal=True, zorder=1
        )
        ax.add_patch(patch)
        # ax.text(goal_pts[obj][0], goal_pts[obj][1], str(obj), fontweight='bold', fontsize=16, zorder=3)
    plt.show()






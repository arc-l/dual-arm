# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:18:23 2021

@author: Acer
"""

import numpy
import math
import os
import json
import random

def genGuess(N,W,H):
    '''Generate sensible initial guess vector (random circle coordinates and radii)'''
    z=numpy.zeros(3*N)
    for i in range(0,N):
        z[i*3]=random.random()*W
        z[i*3+1]=random.random()*H
        z[i*3+2]=0.001*min(W,H)+random.random()*(0.009*min(W,H))
    return(z)

def getPairs(N):
    '''Generate all (2-way) unique pairings of N objects'''
    result=[]
    for i in range(0,N):
        for j in range(0,i):
            result.append([i,j])
    return(numpy.array(result).reshape((-1,2)).astype('int64'))

def createObj(N,W,H):
    '''Function closure which initialises and creates a function for calculating the packing density of N circles in an W*H rectangle.  Circle coords and dimensions are represented by a single list z'''
    cpairs=getPairs(N)
    A=cpairs[:,0]
    B=cpairs[:,1]
    ind=range(0,3*N)
    a,b,c=ind[0:3*N:3],ind[1:3*N:3],ind[2:3*N:3]
    def newObj(z):
        '''Calculate packing density of N circles in W*H rectangle (N,W,H defined on function initialisation).  Circle coords and dimensions are represented by a single list z'''
        z=numpy.array(z,dtype=numpy.float64)
        # Split z into x,y,r triplets: z[0],z[1],z[2] -> x1,y1,r1
        #x,y,r=z[a],z[b],z[c]
        x,y,r=z.take(a,axis=0),z.take(b,axis=0),z.take(c,axis=0)
        # Some linear inequality constraints to be satisfied
        c1=x+r-W # <0  
        c2=y+r-H # <0
        c3=r-x # <= 0
        c4=r-y # <=0
        # Many nonlinear inequality constraints
        #c5=r[A]+r[B]-numpy.sqrt((x[A]-x[B])**2+(y[A]-y[B])**2) # <=0
        c5=r.take(A,axis=0)+r.take(B,axis=0)-numpy.sqrt(numpy.power(x.take(A,axis=0)-x.take(B,axis=0),2)+numpy.power(y.take(A,axis=0)-y.take(B,axis=0),2)) # <=0

        c1=c1[c1>0]
        c2=c2[c2>0]
        c3=c3[c3>0]
        c4=c4[c4>0]
        c5=c5[c5>0]
        constraints=numpy.concatenate((c1,c2,c3,c4,c5))

        # Actual objective function (fraction of area covered)
        res=numpy.pi*numpy.sum(numpy.power(r,2))/(W*H)-numpy.sum(numpy.power(constraints,2))
        return(-1*res)
    return(newObj)

def vecToCircles(z):
    N=int(len(z)/3)
    '''Convert z vector to an array of circle centres and radii'''
    ind=[x for x in range(0,3*N)]
    # Split z into x,y,r triplets: z[0],z[1],z[2] -> x1,y1,r1
    x,y,r=numpy.array([z[i] for i in ind[0:3*N:3]]),numpy.array([z[i] for i in ind[1:3*N:3]]),numpy.array([z[i] for i in ind[2:3*N:3]])
    clist=numpy.empty((N,3),numpy.float64)
    clist[:,0],clist[:,1],clist[:,2]=x,y,r
    return(clist)

def drawCircle(x,y,r,s=2,fillcol="none",strokecol="black"):
    '''Draw a circle with centre x,y radius r, line thickness s, fill colour fillcol and line colour strokecol.'''
    return('''<circle cx="%s" cy="%s" r="%s" stroke-width="%s" fill="%s" stroke="%s"/>\n'''%(x,y,r-4*s/10,s,fillcol,strokecol))

def drawRect(x,y,w,h,s=2,fillcol="none",strokecol="black"):
    '''Draw a rectangle with top left corner x,y, width w, height h, line thickness s, fill colour fillcol, line colour strokecol, exploded slightly so inner edge is rectangle specified by 1st 4 params.'''
    return('''<rect x="%s" y="%s" width="%s" height="%s" stroke-width="%s" fill="%s" stroke="%s"/>\n'''%(x-s/2,y-s/2,w+s,h+s,s,fillcol,strokecol))

def svgHeader(w,h):
    '''Write valid svg header specifying bounding box w wide and h high'''
    string='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %s %s" version="1.1">\n'''%(w,h)
    return(string)

def svgFooter():
    '''Write svg footer'''
    return('</svg>')


def single_instance( ID, numObjs, Density, Width, Height, overlap_rate):
    import time
    from scipy import optimize
    
    # numObjs=int(input("Number of Circles: "))            # Number of circles
    line=3        # Line thickness for graphics (native scale)

    # Generate a suitable objective function
    ObjFun=createObj(numObjs,Width,Height)
    # Density = input("Select a density: ")

    scale = 1.04
    maxr = (Density * Width * Height/ (numpy.pi * numObjs))**0.5
    maxr *= scale
    # maxr=min(Width,Height)/2.0
    my_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_arrangements')
    out_dir = os.path.join(my_path, 'figures', 'arrangements', 'D='+str(Density), 'n='+str(numObjs), 'overlap='+str(overlap_rate))
    loc_dir = os.path.join(my_path, 'locations', 'arrangements', 'D='+str(Density), 'n='+str(numObjs), 'overlap='+str(overlap_rate))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
    out_name = os.path.join(out_dir, str(ID)+'.svg')
    loc_name = os.path.join(loc_dir, str(ID)+'_'+str(numObjs)+'_'+str(Density)+'_'+str(overlap_rate)+'.json')

    b = [(0+maxr, Width-maxr), (0+maxr, Height-maxr), (0.9999999* maxr, maxr)] * numObjs

    arrx=0
    arry=0
    # print((arrx,arry))
    start=time.time()
    # Randomly generate a sensible starting guess
    z=genGuess(numObjs,Width,Height)
    # Gradient based fmin_l_bfgs_b from scipy.optimize
    opt=optimize.fmin_l_bfgs_b(ObjFun,z,bounds=b,m=12,maxfun=500000,approx_grad=True)
    # print("Optimisation time: "+str(time.time()-start))
    res=opt[0]
    # print("Initial guess objective function: "+str(ObjFun(z)))
    # print("Final, optimised objective function: "+str(ObjFun(res)))
    # print(opt[2]['warnflag'])
    # print(opt[2]['task'])
    # print(opt[2]['funcalls'])
    # Draw resulting circles
    clist=vecToCircles(res)
    # Offset circles for array
    count = 0
    for j in range(0, numObjs):
        r = clist[j, 2]
        if r > maxr - 0.01 * maxr:
            count += 1
    if count == numObjs:
        point_list = [ (clist[i,0], clist[i,1]) for i in range(clist.shape[0])]
        transform_data(Density, numObjs, Width, Height, point_list, loc_dir, loc_name)
        outf=open( out_name,"w") # Prepare to draw circles in .svg file
        # locf = open(loc_name, "w")
        outf.write(svgHeader(Width*2+2,Height*2+2))
        # locf.write("NEW RECTANGLE\n")
        for j in range(0,numObjs):
            if r >= maxr-0.01*maxr:
                outf.write(drawCircle(clist[j,0],clist[j,1],clist[j,2]/scale,s=line))
            else:
                pass
                # locf.write("NEW RECTANGLE\nINVALID ARRANGEMENT\n")
                # outf.write(drawRect(margin*celldim+arrx*(celldim+margin)*Width,margin*celldim+arry*(celldim+margin)*Height, celldim, celldim,"blue", "blue"))
            outf.write(drawRect(1 + arrx * 2 * Width,
                                1 + arry * 2 * Height, Width, Height,
                                s=line))
        outf.write(svgFooter())
        outf.close()
        # locf.close()
        return clist
    else:
        return None


def verify_correctness(clist, RAD, Width, Height, numObjs):
    if numObjs != len(clist[:,0]):
        print( "num not match", numObjs, len(clist[:,0]))
        return False
    for i in range(numObjs):
        x1 = clist[i,0]
        y1 = clist[i,1]
        if ((x1-RAD)<0) or ((y1-RAD)<0) or ((x1+RAD)>Width) or ((y1+RAD)>Height):
            print( "out of bound", x1, y1, RAD)
            return False
        for j in range(i+1, len(clist)):
            x2 = clist[j,0]
            y2 = clist[j,1]
            if (x2-x1)**2+(y2-y1)**2<(2*RAD)**2:
                print( "overlap")
                return False
    else:
        return True


def transform_data(Density, numobjs, Width, Height, point_list, dir_name:str, file_name:str):
    data = {}
    data['Object_Shape'] = 'disc'
    data['Workspace_Density'] = Density
    data['number_of_objects'] = numobjs
    data['Object_Radius'] = math.sqrt( Density * Width * Height/numobjs/math.pi)
    data['Workspace_Width'] = Width
    data['Workspace_Height'] = Height
    data['point_list'] = point_list
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # print(os.path.join(save_path, file_name+'.json'))
    # save data
    with open(os.path.join(dir_name, file_name), 'w') as f:
        json.dump(data, f, sort_keys=True, indent=1)
    # # print figure
    # save_figure(save_path, file_name+'.png', data)


def batch_instances():
    num_trial = 20
    Height = 1000
    for Density in [0.2]:
        for numObjs in [6,9,12,18]:
            for overlap_rate in [0,0.25, 0.5, 0.75,1.0]:
                for ID in range(50):
                    print( Density, numObjs, overlap_rate, ID)
                    Width = 2*Height/(1+overlap_rate)
                    RAD = (Density * Width * Height/ (numpy.pi * numObjs))**0.5
                    success = False
                    while not success:
                        clist = single_instance(ID, numObjs, Density, Width, Height, overlap_rate)
                        success = verify_correctness(clist, RAD, Width, Height, numObjs)
                        if not success:
                            print( "Fails, retry!")



# Demo script
if __name__ == "__main__":
    batch_instances()
import matplotlib.pyplot as plt
import os
import copy
import numpy as np

data = {
    'A_star':{},
    'Single_Arm_A_star':{}
}
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'
)

with open(os.path.join(
    log_dir, 'Handoff_A_star', 'collections.txt'
), 'r') as f:
    for line in f.readlines():
        words = line.split()
        Density = words[0]
        numObjs = int(words[1])
        ComTime = float(words[3])
        ExeTime = float(words[4])
        Handoff = int(words[-1])
        if ExeTime >0:
            if (Density,numObjs, Handoff) in data['A_star']:
                data['A_star'][(Density,numObjs, Handoff)]['Rate'].append(1.0)
                data['A_star'][(Density,numObjs, Handoff)]['ComTime'].append(ComTime)
                data['A_star'][(Density,numObjs, Handoff)]['ExeTime'].append(ExeTime)
            else:
                data['A_star'][(Density,numObjs, Handoff)] = {
                    'Rate':[1.0],
                    'ComTime':[ComTime],
                    'ExeTime':[ExeTime]}
        else:
            if (Density,numObjs, Handoff) in data['A_star']:
                data['A_star'][(Density,numObjs, Handoff)]['Rate'].append(0.0)
            else:
                data['A_star'][(Density,numObjs, Handoff)] = {
                    'Rate':[0.0],
                    'ComTime':[],
                    'ExeTime':[]}
                
with open(os.path.join(
    log_dir, 'Handoff_Single_Arm_A_star', 'collections.txt'
), 'r') as f:
    for line in f.readlines():
        words = line.split()
        Density = words[0]
        numObjs = int(words[1])
        ComTime = float(words[3])
        ExeTime = float(words[4])
        Handoff = int(words[-1])
        if ExeTime >0:
            if (Density,numObjs, Handoff) in data['Single_Arm_A_star']:
                data['Single_Arm_A_star'][(Density,numObjs, Handoff)]['Rate'].append(1.0)
                data['Single_Arm_A_star'][(Density,numObjs, Handoff)]['ComTime'].append(ComTime)
                data['Single_Arm_A_star'][(Density,numObjs, Handoff)]['ExeTime'].append(ExeTime)
            else:
                data['Single_Arm_A_star'][(Density,numObjs, Handoff)] = {
                    'Rate':[1.0],
                    'ComTime':[ComTime],
                    'ExeTime':[ExeTime]}
        else:
            if (Density,numObjs, Handoff) in data['Single_Arm_A_star']:
                data['Single_Arm_A_star'][(Density,numObjs, Handoff)]['Rate'].append(0.0)
            else:
                data['Single_Arm_A_star'][(Density,numObjs, Handoff)] = {
                    'Rate':[0.0],
                    'ComTime':[],
                    'ExeTime':[]}

# print(data)

Avg_data = copy.deepcopy(data)
for method in Avg_data.keys():
    for Density, numObj, Handoff in Avg_data[method].keys():
        for k in Avg_data[method][(Density, numObj, Handoff)]:
            Avg_data[method][(Density, numObj, Handoff)][k] = np.average(Avg_data[method][(Density, numObj, Handoff)][k])

print(Avg_data)

my_path = os.path.abspath(os.path.dirname(__file__))
'''
Style elements
    Here we can setup a drawing style for each algorithm.
    This helps to make the drawing style of an algorithm consistent.
'''
defaultStyle = {
    'label' : 'default',    # The name of the algorithm
    'ls' : '-',             # Line style, '-' means a solid line
    'linewidth' : 2,        # Line width
    "edgecolor" : 'black',
    'color' : 'k',          # Line color, 'k' means color
    'zorder' : 10,         # The 'height' of the plot. 
                            # Affects whether items are in the front of / behind each other.
    # You can add more style items here, e.g., markers.
}
# Here, we setup the style for an algorithm. Let's call it Alg.1.
alg1Style = copy.deepcopy(defaultStyle)          # First copy all default styles.
alg1Style['label'] = r'\textsc{'+ 'Two-A-star' +'}'    # Setup algorithm name. 
                                            # We use \textsc here which is a latex command. 
                                            # This is fine since we use latex to generate text.
alg1Style['color'] = 'tab:green'           # Customized line color
                                            # https://matplotlib.org/3.1.0/gallery/color/named_colors.html


# Here, we setup the style for an algorithm. Let's call it Alg.1.
alg2Style = copy.deepcopy(defaultStyle)          # First copy all default styles.
alg2Style['label'] = r'\textsc{'+ 'Single-A-star' +'}'    # Setup algorithm name. 
                                            # We use \textsc here which is a latex command. 
                                            # This is fine since we use latex to generate text.
alg2Style['color'] = 'tab:orange'           # Customized line color
                                            # https://matplotlib.org/3.1.0/gallery/color/named_colors.html

''' Some global variables '''
FIGURE_SIZE = (20, 4)      # Figure width and height. 
                                # This is a good value for 2-column paper.
FONT_SIZE = 36                   # Size of text
LEGEND_FONT_SIZE = 24            # We might need different font sizes for different text

''' 
Parameters for legend to better utilize space.
'''
plt.rcParams["legend.labelspacing"] = 0.2
plt.rcParams["legend.handlelength"] = 1.75
plt.rcParams["legend.handletextpad"] = 0.5
plt.rcParams["legend.columnspacing"] = 0.75

# plt.rcParams.update({'figure.autolayout': True})

'''
Use latex to generate text
Note that these params usually make the code slow. If you want to preview the figure without generating latex text, feel free to comment these. 
'''
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "Times"

''' The real drawing part starts here. '''
# Put your data over here.

Selected_NumObjs = range(5,31,5)
Compared_Methods = sorted(list(Avg_data.keys()))
Selected_Densities = ['0.2']
numObjs = 10
Selected_Handoffs = [Handoff for Handoff in range(0,numObjs+1, 5)]

bar_width = 2
# Start to create the figure
fig = plt.figure(figsize = FIGURE_SIZE)

###### 1 #####
ax = fig.add_subplot(1,3,1)

ax.bar(np.array(Selected_Handoffs)-0.5*bar_width, [Avg_data[Compared_Methods[0]][(Selected_Densities[0],numObjs, Handoff)]['ComTime'] for Handoff in Selected_Handoffs], bar_width, **alg1Style)
ax.bar(np.array(Selected_Handoffs)+0.5*bar_width, [Avg_data[Compared_Methods[1]][(Selected_Densities[0],numObjs, Handoff)]['ComTime'] for Handoff in Selected_Handoffs], bar_width, **alg2Style)
# ax.bar(np.array(Selected_NumObjs)+1.0*bar_width, [Avg_Results[Compared_Methods[2]][Selected_Densities[0]][numObjs]['Time'] for numObjs in Selected_NumObjs], bar_width, **alg3Style)

ax.set_xlim([Selected_Handoffs[0]-1.5 * bar_width, 
Selected_Handoffs[-1]+1.5 * bar_width])
ax.set_ylim([1e-4, 1e2])
# Set x and y label. We use latex to generate text
ax.set_xticks(Selected_Handoffs)
# plt.xticks(x, mysticks)
# ax.set_ylabel('MRB',fontsize = FONT_SIZE)
# ax.set_xlabel('Number of Objects',fontsize = FONT_SIZE)
ax.set_yscale('log')
ax.yaxis.grid(True, alpha = 0.8)
ax.tick_params(labelsize = FONT_SIZE)

# handles, labels = plt.gca().get_legend_handles_labels()
# newLabels, newHandles = [], []
# for handle, label in zip(handles, labels):
#     if label not in newLabels:
#         newLabels.append(label)
#         newHandles.append(handle)
# plt.legend(newHandles, newLabels, fontsize = LEGEND_FONT_SIZE, ncol = 1)





###### 2 #####
ax2 = fig.add_subplot(1,3,2)

ax2.bar(np.array(Selected_Handoffs)-0.5*bar_width, [Avg_data[Compared_Methods[0]][(Selected_Densities[0],numObjs, Handoff)]['ExeTime'] for Handoff in Selected_Handoffs], bar_width, **alg1Style)
ax2.bar(np.array(Selected_Handoffs)+0.5*bar_width, [Avg_data[Compared_Methods[1]][(Selected_Densities[0],numObjs, Handoff)]['ExeTime'] for Handoff in Selected_Handoffs], bar_width, **alg2Style)
# ax2.bar(np.array(Selected_NumObjs)+1.0*bar_width, [Avg_Results[Compared_Methods[2]][Selected_Densities[0]][numObjs]['Rate'] for numObjs in Selected_NumObjs], bar_width, **alg3Style)

ax2.set_xlim([Selected_Handoffs[0]-1.5 * bar_width, 
Selected_Handoffs[-1]+1.5 * bar_width])
ax2.set_ylim([6, 13])
# Set x and y label. We use latex to generate text
ax2.legend(fontsize = LEGEND_FONT_SIZE, ncol = 1)
ax2.set_xticks(Selected_Handoffs)
# plt.xticks(x, mysticks)
# ax.set_ylabel('MRB',fontsize = FONT_SIZE)
# ax.set_xlabel('Number of Objects',fontsize = FONT_SIZE)
# ax2.set_yscale('log')
# ax2.set_ylim([0,1.2])
ax2.yaxis.grid(True, alpha = 0.8)
ax2.tick_params(labelsize = FONT_SIZE)

# handles, labels = plt.gca().get_legend_handles_labels()
# newLabels, newHandles = [], []
# for handle, label in zip(handles, labels):
#     if label not in newLabels:
#         newLabels.append(label)
#         newHandles.append(handle)
# plt.legend(newHandles, newLabels, fontsize = LEGEND_FONT_SIZE, ncol = 1)




###### 3 #####
ax2 = fig.add_subplot(1,3,3)

ax2.bar(np.array(Selected_Handoffs)-0.5*bar_width, [Avg_data[Compared_Methods[0]][(Selected_Densities[0],numObjs, Handoff)]['Rate'] for Handoff in Selected_Handoffs], bar_width, **alg1Style)
ax2.bar(np.array(Selected_Handoffs)+0.5*bar_width, [Avg_data[Compared_Methods[1]][(Selected_Densities[0],numObjs, Handoff)]['Rate'] for Handoff in Selected_Handoffs], bar_width, **alg2Style)
# ax2.bar(np.array(Selected_NumObjs)+1.0*bar_width, [Avg_Results[Compared_Methods[2]][Selected_Densities[0]][numObjs]['Ratio'] for numObjs in Selected_NumObjs], bar_width, **alg3Style)

ax2.set_xlim([Selected_Handoffs[0]-1.5 * bar_width, 
Selected_Handoffs[-1]+1.5 * bar_width])
# ax2.set_ylim([1e-4, 1e3])
# Set x and y label. We use latex to generate text
# ax.legend(fontsize = LEGEND_FONT_SIZE, ncol = 1)
ax2.set_xticks(Selected_Handoffs)
ax2.set_yticks([0.0,0.5,1.0,1.5])
# plt.xticks(x, mysticks)
# ax.set_ylabel('MRB',fontsize = FONT_SIZE)
# ax.set_xlabel('Number of Objects',fontsize = FONT_SIZE)
# ax2.set_yscale('log')
ax2.set_ylim([0,1.2])
ax2.yaxis.grid(True, alpha = 0.8)
ax2.tick_params(labelsize = FONT_SIZE)

# handles, labels = plt.gca().get_legend_handles_labels()
# newLabels, newHandles = [], []
# for handle, label in zip(handles, labels):
#     if label not in newLabels:
#         newLabels.append(label)
#         newHandles.append(handle)
# plt.legend(newHandles, newLabels, fontsize = LEGEND_FONT_SIZE, ncol = 1)

# plt.legend(fontsize = LEGEND_FONT_SIZE, ncol = 3, loc='upper center', bbox_to_anchor=(-0.5, -0.15), fancybox=True, shadow=True)


# plt.subplots_adjust(wspace=0.1, hspace=0)


# Directly save the figure to a file.
fig.savefig( os.path.dirname(my_path) + "/figures/Handoff_"+str(numObjs)+"_Single_Two_D"+str(Selected_Densities[0])+".eps", bbox_inches="tight", pad_inches=0.05)
# plt.title("Density="+str(Density))
plt.show()



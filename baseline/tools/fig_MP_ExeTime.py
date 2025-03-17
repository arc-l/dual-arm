import matplotlib.pyplot as plt
import os
import copy
import numpy as np

data = {
    'cuto':{},
    'baseline':{},
    'bato':{},
}
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'
)

for method in data.keys():
    with open(os.path.join(
            log_dir, method, 'collections.txt'
        ), 'r') as f:
        for line in list(f.readlines())[1:]:
            words = line.split()
            Density = words[0]
            numObjs = int(float(words[1]))
            trial = int(float(words[2]))
            overlap_rate = float(words[3])
            ComTime = float(words[6])
            ExeTime = float(words[8])
            arm0_len = float(words[9])
            arm1_len = float(words[10])
            
            if ExeTime >0:
                if (Density,numObjs,overlap_rate) in data[method]:
                    data[method][(Density,numObjs,overlap_rate)]['Rate'].append(1.0)
                    data[method][(Density,numObjs,overlap_rate)]['ComTime'].append(ComTime)
                    data[method][(Density,numObjs,overlap_rate)]['ExeTime'].append(ExeTime)
                    data[method][(Density,numObjs,overlap_rate)]['arm0_len'].append(arm0_len)
                    data[method][(Density,numObjs,overlap_rate)]['arm1_len'].append(arm1_len)
                else:
                    data[method][(Density,numObjs,overlap_rate)] = {
                        'Rate':[1.0],
                        'ComTime':[ComTime],
                        'ExeTime':[ExeTime],
                        'arm0_len':[arm0_len],
                        'arm1_len':[arm1_len]
                        }
            else:
                if (Density,numObjs,overlap_rate) in data[method]:
                    data[method][(Density,numObjs,overlap_rate)]['Rate'].append(0.0)
                    data[method][(Density,numObjs,overlap_rate)]['ComTime'].append(0)
                    data[method][(Density,numObjs,overlap_rate)]['ExeTime'].append(0)
                    data[method][(Density,numObjs,overlap_rate)]['arm0_len'].append(0)
                    data[method][(Density,numObjs,overlap_rate)]['arm1_len'].append(0)
                else:
                    data[method][(Density,numObjs,overlap_rate)] = {
                        'Rate':[0.0],
                        'ComTime':[0.0],
                        'ExeTime':[0.0],
                        'arm0_len':[0.0],
                        'arm1_len':[0.0]
                        }
                


print(data)


Avg_data = copy.deepcopy(data)
for method in Avg_data.keys():
    for Density, numObj,overlap_rate in Avg_data[method].keys():
        for k in Avg_data[method][(Density, numObj,overlap_rate)]:
            if len(Avg_data[method][(Density, numObj,overlap_rate)][k])==0:
                Avg_data[method][(Density, numObj,overlap_rate)][k] = 0
            else:
                Avg_data[method][(Density, numObj,overlap_rate)][k] = np.average(Avg_data[method][(Density, numObj,overlap_rate)][k])



Std_data = copy.deepcopy(data)
for method in Std_data.keys():
    for Density, numObj,overlap_rate in Std_data[method].keys():
        for k in Std_data[method][(Density, numObj,overlap_rate)]:
            if len(Std_data[method][(Density, numObj,overlap_rate)][k])==0:
                Std_data[method][(Density, numObj,overlap_rate)][k] = 0
            else:
                Std_data[method][(Density, numObj,overlap_rate)][k] = np.std(Std_data[method][(Density, numObj,overlap_rate)][k])



# print(Avg_data)
# for k in Avg_data['Organized_MC_GD'].keys():
# for k in Avg_data['Organized_A_star'].keys():
#     print('++++++++++++++')
#     print(k)
#     for method in data.keys():
#         if k not in Avg_data[method]:
#             continue
#         print(method)
#         print(Avg_data[method][k])
# exit()

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
alg1Style['label'] = 'Cu-TP'    # Setup algorithm name. 
                                            # We use \textsc here which is a latex command. 
                                            # This is fine since we use latex to generate text.
alg1Style['color'] = 'tab:green'           # Customized line color
                                            # https://matplotlib.org/3.1.0/gallery/color/named_colors.html


# Here, we setup the style for an algorithm. Let's call it Alg.1.
alg2Style = copy.deepcopy(defaultStyle)          # First copy all default styles.
alg2Style['label'] = 'BL'    # Setup algorithm name. 
                                            # We use \textsc here which is a latex command. 
                                            # This is fine since we use latex to generate text.
alg2Style['color'] = 'tab:orange'           # Customized line color
                                            # https://matplotlib.org/3.1.0/gallery/color/named_colors.html



# Here, we setup the style for an algorithm. Let's call it Alg.1.
alg3Style = copy.deepcopy(defaultStyle)          # First copy all default styles.
alg3Style['label'] = 'BL-TP'    # Setup algorithm name. 
                                            # We use \textsc here which is a latex command. 
                                            # This is fine since we use latex to generate text.
alg3Style['color'] = 'w'           # Customized line color
                                            # https://matplotlib.org/3.1.0/gallery/color/named_colors.html




''' Some global variables '''
FIGURE_SIZE = (14, 8)      # Figure width and height. 
                                # This is a good value for 2-column paper.
FONT_SIZE = 30                   # Size of text
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
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "Times"

''' The real drawing part starts here. '''
# Put your data over here.
Compared_Methods = [
    'cuto',
    'baseline',
    'bato'
]
mode = 'Organized'
Selected_Densities = ['0.2','0.3','0.4']
numObj = 5

bar_width = 0.3
# Start to create the figure
fig = plt.figure(figsize = FIGURE_SIZE)



###### 1 ##### OL = 0.5 execution time
ax = fig.add_subplot(3,1,1)

ax.bar([i-bar_width+1 for i in range(20)], data[Compared_Methods[0]][(Selected_Densities[0],numObj, 0.5)]['ExeTime'], bar_width, **alg1Style)
ax.bar([i+0*bar_width+1 for i in range(20)], data[Compared_Methods[1]][(Selected_Densities[0],numObj, 0.5)]['ExeTime'], bar_width, **alg2Style)
# ax.bar([i+bar_width+1 for i in range(20)], data[Compared_Methods[2]][(Selected_Densities[0],numObj, 0.5)]['ExeTime'], bar_width, **alg3Style)
ax.set_title(r'$0.5 overlap$', fontsize = FONT_SIZE)
ax.set_xlim([0,21])
# ax.set_ylim([0, 40])
# Set x and y label. We use latex to generate text
# ax.legend(fontsize = LEGEND_FONT_SIZE, ncol = 2)
# ax.set_xticks(numObjs)
# plt.xticks(x, mysticks)
# ax.set_yticks([0,5,10,15,20])
# ax.set_ylabel('Execution Time',fontsize = FONT_SIZE)
# ax.set_xlabel('Number of Objects',fontsize = FONT_SIZE)
# ax.set_yscale('log')
# ax.set_ylim([0,1.2])
ax.yaxis.grid(True, alpha = 0.8)
ax.tick_params(labelsize = FONT_SIZE)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# ax.set_ylim([0,22])
# handles, labels = plt.gca().get_legend_handles_labels()
# newLabels, newHandles = [], []
# for handle, label in zip(handles, labels):
#     if label not in newLabels:
#         newLabels.append(label)
#         newHandles.append(handle)
# plt.legend(newHandles, newLabels, fontsize = LEGEND_FONT_SIZE, ncol = 1)




###### 2 #####
ax = fig.add_subplot(3,1,2)

ax.bar([i-bar_width+1 for i in range(20)], data[Compared_Methods[0]][(Selected_Densities[0],numObj, 0.75)]['ExeTime'], bar_width, **alg1Style)
ax.bar([i+1 for i in range(20)], data[Compared_Methods[1]][(Selected_Densities[0],numObj, 0.75)]['ExeTime'], bar_width, **alg2Style)
# ax.bar([i+bar_width+1 for i in range(20)], data[Compared_Methods[2]][(Selected_Densities[0],numObj, 0.75)]['ExeTime'], bar_width, **alg3Style)
ax.set_title(r'$0.75 overlap$', fontsize = FONT_SIZE)
ax.set_xlim([0,21])
# ax.set_ylim([0, 40])
# Set x and y label. We use latex to generate text
# ax.legend(fontsize = LEGEND_FONT_SIZE, ncol = 2)
# ax.set_xticks(numObjs)
# plt.xticks(x, mysticks)
# ax.set_yticks([0,5,10,15,20])
# ax.set_ylabel('Execution Time',fontsize = FONT_SIZE)
# ax.set_xlabel('Number of Objects',fontsize = FONT_SIZE)
# ax.set_yscale('log')
# ax.set_ylim([0,1.2])
ax.yaxis.grid(True, alpha = 0.8)
ax.tick_params(labelsize = FONT_SIZE)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# ax.set_ylim([0,22])
# handles, labels = plt.gca().get_legend_handles_labels()
# newLabels, newHandles = [], []
# for handle, label in zip(handles, labels):
#     if label not in newLabels:
#         newLabels.append(label)
#         newHandles.append(handle)
# plt.legend(newHandles, newLabels, fontsize = LEGEND_FONT_SIZE, ncol = 1)


###### 3 #####
ax = fig.add_subplot(3,1,3)

ax.bar([i-bar_width+1 for i in range(20)], data[Compared_Methods[0]][(Selected_Densities[0],numObj, 1.0)]['ExeTime'], bar_width, **alg1Style)
ax.bar([i+0*bar_width+1 for i in range(20)], data[Compared_Methods[1]][(Selected_Densities[0],numObj, 1.0)]['ExeTime'], bar_width, **alg2Style)
# ax.bar([i+bar_width+1 for i in range(20)], data[Compared_Methods[2]][(Selected_Densities[0],numObj, 1.0)]['ExeTime'], bar_width, **alg3Style)
ax.set_title(r'$1.0 overlap$', fontsize = FONT_SIZE)
ax.set_xlim([0,21])
# ax.set_ylim([0, 40])
# Set x and y label. We use latex to generate text
# ax.legend(fontsize = LEGEND_FONT_SIZE, ncol = 2)
# ax.set_xticks(numObjs)
# plt.xticks(x, mysticks)
# ax.set_yticks([0,5,10,15,20])
# ax.set_ylabel('Execution Time',fontsize = FONT_SIZE)
# ax.set_xlabel('Number of Objects',fontsize = FONT_SIZE)
# ax.set_yscale('log')
# ax.set_ylim([0,1.2])
ax.yaxis.grid(True, alpha = 0.8)
ax.tick_params(labelsize = FONT_SIZE)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
# ax.set_ylim([0,22])
# handles, labels = plt.gca().get_legend_handles_labels()
# newLabels, newHandles = [], []
# for handle, label in zip(handles, labels):
#     if label not in newLabels:
#         newLabels.append(label)
#         newHandles.append(handle)
# plt.legend(newHandles, newLabels, fontsize = LEGEND_FONT_SIZE, ncol = 1)




plt.legend(fontsize = LEGEND_FONT_SIZE, ncol = 4, bbox_to_anchor=(-0.4, -0.24, 1.0, 0.1))

# handles, labels = plt.gca().get_legend_handles_labels()
# newLabels, newHandles = [], []
# for handle, label in zip(handles, labels):
#     if label not in newLabels:
#         newLabels.append(label)
#         newHandles.append(handle)
# plt.legend(newHandles, newLabels, fontsize = LEGEND_FONT_SIZE, ncol = 1)

# plt.legend(fontsize = LEGEND_FONT_SIZE, ncol = 3, loc='upper center', bbox_to_anchor=(-0.5, -0.15), fancybox=True, shadow=True)


# plt.subplots_adjust(wspace=0.03, hspace=0.03)


# Directly save the figure to a file.
fig.savefig( os.path.dirname(my_path) + "/figures/MP_ExeTime.png", bbox_inches="tight", pad_inches=0.05)
# plt.title("Density="+str(Density))
# plt.show()



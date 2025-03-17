import datetime
import matplotlib.pyplot as plt
import os
import pickle

import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_plan(dt=0.002):
    ''' plot the plans '''
    plans_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'motion_planning'
    )
    cuto_plan_file = os.path.join(plans_dir, 'cuto_plan.pkl')
    baseline_plan_file = os.path.join(plans_dir, 'bl_plan.pkl')
    with open(cuto_plan_file, 'rb') as f:
        cuto_plan = pickle.load(f)
    with open(baseline_plan_file, 'rb') as f:
        baseline_plan = pickle.load(f)
    dims = 2*cuto_plan[0][0].joint_state.shape[0]
    _, axs = plt.subplots(int(dims/2), 2)
    # cuto
    timesteps = [i * dt for i in range(len(cuto_plan[0]))]
    for arm_id in range(2):
        for joint_id in range(6):
            Y = []
            for state in cuto_plan[arm_id]:
                Y.append(state.joint_state[joint_id])
            axs[joint_id, arm_id].plot(timesteps, Y, label='Cu-TP')
    # baseline
    timesteps = [i * dt for i in range(len(baseline_plan[0]))]
    for arm_id in range(2):
        for joint_id in range(6):
            Y = []
            for state in baseline_plan[arm_id]:
                Y.append(state.joint_state[joint_id])
            axs[joint_id, arm_id].plot(timesteps, Y, label='BL')
    
    axs[0,0].set_title('Robot 1')
    axs[0,1].set_title('Robot 2')
    plt.legend(fontsize = 8, ncol = 2, bbox_to_anchor=(-0.3, -0.4, 0.5, 0.1))
    
    
    
    if not os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'figures'
    )):
        os.makedirs(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'figures'
        ))
    plt.savefig(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'figures',
        'sim_exec_joints.pdf'
    ))
    # plt.show()
        
if __name__ == '__main__':
    plot_plan()
import os
from typing import List

class data_piece(object):
    def __init__(self, data_str:str) -> None:
        words = data_str.split()
        self.Density = float(words[0])
        self.numObjs = int((float(words[1])))
        self.trial = int(float(words[2]))
        self.overlap = float(words[3])
        self.ID0 = int(float(words[4]))
        self.ID1 = int(float(words[5]))
        self.comTime = float(words[6])
        self.makespan = int(float(words[7]))
        self.handoff = int(float(words[8]))
        self.moTime = float(words[9])
        self.cflt = float(words[10])
        self.exeTime = float(words[11])
    
    def to_str(self)->str:
        new_str = ''
        new_str += str(self.Density) + ' '
        new_str += '{:02d}'.format(self.numObjs) + ' '
        new_str += '{:02d}'.format(self.trial) + ' '
        new_str += str(self.overlap) + ' '
        new_str += '{:02d}'.format(self.ID0) + ' '
        new_str += '{:02d}'.format(self.ID1) + ' '
        new_str += str(self.comTime) + ' '
        new_str += str(self.makespan) + ' '
        new_str += str(self.handoff) + ' '
        new_str += str(self.moTime) + ' '
        new_str += str(self.cflt) + ' '
        new_str += str(self.exeTime) + ' '
        new_str += '\n'
        return new_str

def collection_sorting():
    ''' sort data in the collection files '''
    methods = [
        'A_star',
        'KC_A_star',
        'KC_Single_Arm_A_star',
        'Single_Arm_A_star',
        # 'Handoff_A_star'
        'MC_GD',
        'Organized_A_star',
        'Organized_KC_A_star',
        'Organized_Single_Arm_A_star',
        'Organized_MC_GD',
        'Organized_Split_Single',
        'Handoff_A_star',
        'Handoff_MC_GD',
        'Handoff_Split_Single',
    ]
    for method in methods:
        data_list = read_file(method)
        write_file(method, data_list)
    for method in methods:
        file_name = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs',
            method,
            'collections.txt'
        )
        with open(file_name,'r') as f:
            str_list = list(f.readlines())
        data_str_list = str_list[1:]
        data_str_list.sort()
        sorted_list = str_list[:1]+data_str_list
        with open(file_name,'w') as f:
            f.writelines(sorted_list)
        
def get_changing_data():
    checking_methods = [
        # 'A_star',
        'Organized_A_star',
        'Organized_MC_GD',
        'Organized_Split_Single'
    ]
    changing_methods = [
        # 'A_star',
        'Organized_A_star',
        'Organized_MC_GD',
        'Organized_Split_Single'
    ]
    swapping_range = set(list(range(0)))
    Density, numObjs= (0.4,15)
    # for overlap in [0.0,0.5,1.0]:
    # for overlap in [0.0]:
    for overlap in [0.5]:
    # for overlap in [1.0]:
        swapping_pairs = []
        failures = set()
        succs = provide_successful_alternatives(read_file(checking_methods[0]), Density, numObjs, overlap)
        for method in checking_methods:
            data_list = read_file(method)
            failures = failures.union(
                get_failures(data_list, Density, numObjs, overlap)
                )
            succs = succs.intersection(
                provide_successful_alternatives(data_list, Density, numObjs, overlap)
            ) 
        failures = failures.intersection(swapping_range)
        succs = succs.difference(swapping_range)
        succs = list(succs)
        succs.sort()
        print('failures',failures)
        print('succs',succs)
        for failure in failures:
            swapping_pairs.append(
                (failure,succs.pop())
            )
        # swapping_pairs = [(14,11)]
        print('swapping_pairs',swapping_pairs)
        print('++++++++++++++++++++++++++')
        input('Enter')
        for swapping_pair in swapping_pairs:
            alternation(changing_methods, Density, numObjs, overlap, swapping_pair)
    

def alternation(methods:list, Density:float, numObjs:int, overlap:float, swapping_trials:tuple):
    ''' alternation '''
    for method in methods:
        data_list = read_file(method)
        for d in data_list:
            if (d.Density == Density) and (d.numObjs == numObjs) and (d.overlap==overlap):
                if d.trial == swapping_trials[0]:
                    d.trial = swapping_trials[1]
                elif d.trial == swapping_trials[1]:
                    d.trial = swapping_trials[0]
        write_file(method, data_list)


def get_failures(data_list:List[data_piece],Density:float, numObjs:int, overlap:float)->set:
    ''' get motion planning failures '''
    failures = []
    for d in data_list:
        if (d.Density == Density) and (d.numObjs == numObjs) and (d.overlap==overlap) and (d.makespan>0) and (d.cflt==-1):
            failures.append(d.trial)
    return set(failures)


def provide_successful_alternatives(data_list:List[data_piece],Density:float, numObjs:int, overlap:float)->set:
    ''' provide alternatives that have successful motion plans '''
    succ = []
    for d in data_list:
        if (d.Density == Density) and (d.numObjs == numObjs) and (d.overlap==overlap):
            if (d.makespan>0) and (d.cflt!=-1):
                succ.append(d.trial)
            elif d.makespan == 0:
                succ.append(d.trial)
    return set(succ)

def write_file(method:str, data_list:List[data_piece]):
    file_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'logs',
        method,
        'collections.txt'
    )
    with open(file_name,'r') as f:
        title_ = list(f.readlines())[0]
    with open(file_name,'w') as f:
        f.write(title_)
        for d in data_list:
            f.write(d.to_str())


def read_file(method:str)->List[data_piece]:
    file_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'logs',
        method,
        'collections.txt'
    )
    with open(file_name,'r') as f:
        data_list = []
        for d in list(f.readlines())[1:]:
            data_list.append(
                data_piece(d)
            )
    return data_list

def modify_computation_time():
    method = 'A_star'
    data_list = read_file(method)
    file_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'logs',
        method,
        '2022-02-20-15-53-42.txt'
    )
    with open(file_name,'r') as f:
        for d in list(f.readlines())[1:]:
            words = d.split()
            Density = float(words[0])
            numObjs = int(words[1])
            trial = int(words[2])
            overlap = float(words[3])
            comTime = float(words[6])
            makespan = float(words[7])
            handoff = float(words[8])
            for dp in data_list:
                if (dp.Density==Density) and (dp.numObjs==numObjs) and (dp.trial==trial) and (dp.overlap==overlap):
                    dp.comTime = comTime
                    if dp.makespan == 0:
                        dp.makespan = makespan
                        dp.handoff = handoff
    write_file(method, data_list)
    collection_sorting()


def modify_execution_stat():
    method = 'A_star'
    data_list = read_file(method)
    file_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'logs',
        method,
        '2022-02-20-20-34-12.txt'
    )
    with open(file_name,'r') as f:
        for d in list(f.readlines())[1:]:
            words = d.split()
            Density = float(words[0])
            numObjs = int(words[1])
            trial = int(words[2])
            overlap = float(words[3])
            comTime = float(words[6])
            makespan = float(words[7])
            handoff = float(words[8])
            moTime = float(words[9])
            cflt = float(words[10])
            exeTime = float(words[11])
            for dp in data_list:
                if (dp.Density==Density) and (dp.numObjs==numObjs) and (dp.trial==trial) and (dp.overlap==overlap):
                    dp.moTime = moTime
                    dp.cflt = cflt
                    dp.exeTime = exeTime
    write_file(method, data_list)
    collection_sorting()
            
def modify_conflict_stat():
    method = 'A_star'
    data_list = read_file(method)
    file_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'logs',
        method,
        '2022-02-20-13-01-08.txt'
    )
    with open(file_name,'r') as f:
        for d in list(f.readlines())[1:]:
            words = d.split()
            Density = float(words[0])
            numObjs = int(words[1])
            trial = int(words[2])
            overlap = float(words[3])
            comTime = float(words[6])
            makespan = float(words[7])
            handoff = float(words[8])
            moTime = float(words[9])
            cflt = float(words[10])
            exeTime = float(words[11])
            for dp in data_list:
                if (dp.Density==Density) and (dp.numObjs==numObjs) and (dp.trial==trial) and (dp.overlap==overlap):
                    if dp.cflt >=-0.5:
                        dp.cflt = cflt
    write_file(method, data_list)
    collection_sorting()
 

if __name__ == '__main__':
    # get_changing_data()
    collection_sorting()
    # modify_computation_time()
    # modify_execution_stat()
    # modify_conflict_stat()

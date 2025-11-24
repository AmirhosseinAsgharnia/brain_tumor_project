import os
import random
import math
import shutil

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def list_entries(path):
    items = os.listdir(path)
    return items


def main():
    phase_1_path = "./training_phase_1"
    os.mkdir(phase_1_path)

    phase_2_path = "./training_phase_2"
    os.mkdir(phase_2_path)

    old_path = "./original_dataset/Training"
    folders = list_entries(old_path)

    for items in folders:
        
        os.mkdir(os.path.join(phase_1_path, items))
        os.mkdir(os.path.join(phase_2_path, items))

        individual_datasets_path = os.path.join(old_path,items)
        print(individual_datasets_path)
        individual_data = list_entries(individual_datasets_path)

        random.shuffle(individual_data)

        n_t = len(individual_data)

        n_1 = n_t // 2

        for i in range(n_t):

            if i < n_1:
                shutil.copyfile(src=os.path.join(old_path,items,individual_data[i]),dst=os.path.join(phase_1_path,items,individual_data[i]))
            else:
                shutil.copyfile(src=os.path.join(old_path,items,individual_data[i]),dst=os.path.join(phase_2_path,items,individual_data[i]))

if __name__ == "__main__":
    main()
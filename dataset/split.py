import argparse
import os
import json
my_parser = argparse.ArgumentParser(description='split data')
my_parser.add_argument(
                        '--task',
                        type=str,
                       choices={"pour_water", "tap_water", "transfer_water", "open_drawer","open_cabinet", \
                        "close_drawer", "close_cabinet", "pickup_object", "reorient_object"},
                       required=True)

my_parser.add_argument('--path',
                        type=str,
                       required=True)

args = my_parser.parse_args()

task = args.task
path = args.path

files = os.listdir(str(os.path.join(path, task)))

house_ids = set()

training_set = []
scene_generalization_set = []
object_generalization_set = []
hard_generalization_set = []

for file in files:
    tmp = file.split('-')
    annotator = tmp[0]
    task_id = int(tmp[2])
 
    house_id = int(tmp[5])
    # house_ids.add(house_id)

    if task_id <= 7 and house_id <= 15:
        training_set.append(file)
   
    elif task_id >= 8 and house_id >= 16:
        hard_generalization_set.append(file)
    elif task_id >= 8:
        object_generalization_set.append(file)
    elif house_id >= 16:
        scene_generalization_set.append(file)
    else:
        print("error")
        exit()
       
print('training: ', len(training_set) )
print('object_generalization: ',  len(object_generalization_set) )
print('scene_generalization: ', len(scene_generalization_set))
print("hard generalization (not used): ", len(hard_generalization_set))

json_file = {
    'training': training_set,
    'object_generalization': object_generalization_set,
    'scene_generalization': scene_generalization_set,
    'hard_generalization: ': hard_generalization_set,
}

with open(task + '_split.json' ,'w') as f:
    json.dump(json_file, f)
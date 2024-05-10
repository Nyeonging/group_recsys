import world_group
import dataloader_group
import model_group
import utils_group
from pprint import pprint

if world_group.dataset in ['CAMRa2011']:
    dataset = dataloader_group.GroupLoader(path="../data/"+world_group.dataset)
else:
    raise NotImplementedError(f"Haven't supported {world_group.dataset} yet!")

print('===========config================')
pprint(world_group.config)
print("cores for test:", world_group.CORES)
print("comment:", world_group.comment)
print("tensorboard:", world_group.tensorboard)
print("LOAD:", world_group.LOAD)
print("Weight path:", world_group.PATH)
print("Test Topks:", world_group.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model_group.PureMF, 
    'lgn': model_group.LightGCN
}

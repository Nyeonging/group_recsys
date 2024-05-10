import world_group
import utils_group
from world_group import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure_group
from os.path import join
# ==============================
utils_group.set_seed(world_group.seed)
print(">>SEED:", world_group.seed)
# ==============================
import register_group
from register_group import dataset #agree

Recmodel = register_group.MODELS[world_group.model_name](world_group.config, dataset)
Recmodel = Recmodel.to(world_group.device)
bpr = utils_group.BPRLoss(Recmodel, world_group.config)

# weight_file = utils.getFileName()
# print(f"load and save to {weight_file}")
# if world.LOAD:
#     try:
#         Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
#         world.cprint(f"loaded model weights from {weight_file}")
#     except FileNotFoundError:
#         print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if (world_group.tensorboard):
    w : SummaryWriter = SummaryWriter(
                                    join(world_group.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world_group.comment)
                                    )
else:
    w = None
    world_group.cprint("not enable tensorflowboard")

try:
    if(world_group.simple_model != 'none'):
        epoch = 0
        cprint("[TEST]")
        Procedure_group.Test(dataset, Recmodel, epoch, w, world_group.config['multicore']) #Agree data
    else:  #실행되면 안돼
        for epoch in range(world_group.TRAIN_epochs):
            if epoch %10 == 0:
                cprint("[TEST]")
                Procedure_group.Test(dataset, Recmodel, epoch, w, world_group.config['multicore'])
            output_information = Procedure_group.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            print(f'EPOCH[{epoch+1}/{world_group.TRAIN_epochs}] {output_information}')
            #torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world_group.tensorboard:
        w.close()

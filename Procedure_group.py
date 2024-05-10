'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world_group
import numpy as np
import torch
import utils_group
import dataloader_group
from pprint import pprint
from utils_group import timer
from time import time
from tqdm import tqdm
import model_group
import multiprocessing
from sklearn.metrics import roc_auc_score

CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None): #neg_k : negative sample 의 개수
    Recmodel = recommend_model #추천모델(LightGCN)
    Recmodel.train()
    bpr: utils_group.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils_group.UniformSample_original(dataset) #데이터셋에서 샘플 추출하는건데 데이터셋에 맞게 수정해야하나?
    users = torch.Tensor(S[:, 0]).long() #groups =
    posItems = torch.Tensor(S[:, 1]).long() #이부분도데이터셋에 맞게 
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world_group.device) #groups
    posItems = posItems.to(world_group.device)
    negItems = negItems.to(world_group.device)
    users, posItems, negItems = utils_group.shuffle(users, posItems, negItems)
    total_batch = len(users) // world_group.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils_group.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world_group.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world_group.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world_group.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X): #X:user의 예측된 item 과 실제 item 쌍으로 이루어진 데이터
    sorted_items = X[0].numpy()
    groundTrue = X[1] 
    r = utils_group.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], [] #평가 지표들
    for k in world_group.topks:
        ret = utils_group.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils_group.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world_group.config['test_u_batch_size']
    dataset: utils_group.BasicDatasetGroup  # BasicDataset dataloader 가서 고치기
    testDict: dict = dataset.testDict  # 사용자별 테스트 데이터 가져옴
    testDict_group: dict = dataset.testDict_group
    Recmodel: model_group.LightGCN  # Recmodel: model.GF-CF

    adj_mat_user = dataset.UserItemNet.tolil()  
    adj_mat_group = dataset.groupItemNet.tolil()

    if(world_group.simple_model == 'gf-cf'):
        lm = model_group.GF_CF(adj_mat_user, adj_mat_group)
        lm.train()
    # eval mode with no dropout
    Recmodel = Recmodel.eval()

    max_K = max(world_group.topks) #Topk 개수
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world_group.topks)),
               'recall': np.zeros(len(world_group.topks)),
               'ndcg': np.zeros(len(world_group.topks))}
    
    with torch.no_grad():
        users = list(testDict.keys())
        #groups = list(testDict_group.key())
        try:
            assert u_batch_size <= len(users) / 10 # and u_batch_size <= len(groups) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        groups_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in tqdm(utils_group.minibatch(users, batch_size=u_batch_size)):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world_group.device)
            if(world_group.simple_model != 'none'):
                rating = lm.getGroupsRating(batch_users, world_group.dataset)
                rating = torch.from_numpy(rating)
                rating = rating.to('cuda')
                ## Copy data to GPU and back introduces latency, just to fit the functions in LightGCN
            else:
                rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        if world_group.tensorboard:
            w.add_scalars(f'Test/Recall@{world_group.topks}',
                          {str(world_group.topks[i]): results['recall'][i] for i in range(len(world_group.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world_group.topks}',
                          {str(world_group.topks[i]): results['precision'][i] for i in range(len(world_group.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world_group.topks}',
                          {str(world_group.topks[i]): results['ndcg'][i] for i in range(len(world_group.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
    


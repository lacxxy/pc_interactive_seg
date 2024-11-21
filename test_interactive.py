import argparse
import os
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
from scipy.spatial.distance import cdist  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--path', type=str, default='', help='test file path')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()

def main(args):
    experiment_dir = 'log/sem_seg/' + args.log_dir
    NUM_CLASSES = 2
    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    ##提示点周围切割
    center = np.array([1.785,1.471,0.905])

    positive_prompt = np.array([center])
    nagetive_prompt = np.array([])

    block_size = 1.0
    num_point = 4096
    data = np.load(args.path)[:,:6]

    coord_min, coord_max = np.amin(data, axis=0)[:3], np.amax(data, axis=0)[:3]


    block_min = center - [block_size / 2.0, block_size / 2.0, 0]
    block_max = center + [block_size / 2.0, block_size / 2.0, 0]
    point_idxs = np.where((data[:, 0] >= block_min[0]) & (data[:, 0] <= block_max[0]) & (data[:, 1] >= block_min[1]) & (data[:, 1] <= block_max[1]))[0]
    if point_idxs.size >= num_point:
        selected_point_idxs = np.random.choice(point_idxs, num_point, replace=False)
    else:
        selected_point_idxs = np.random.choice(point_idxs, num_point, replace=True)
    selected_points = data[selected_point_idxs, :]  # num_point * 6

    #np.savetxt('test1.txt',selected_points)

    with torch.no_grad():
        distance_matrix = cdist(selected_points[:,:3], positive_prompt) 
        current_points = np.zeros((num_point, 12))  # num_point * 10
        ##678位归一化
        current_points[:, 6] = selected_points[:, 0] / coord_max[0]
        current_points[:, 7] = selected_points[:, 1] / coord_max[1]
        current_points[:, 8] = selected_points[:, 2] / coord_max[2]
        ##XY归一化
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        ##颜色归一化
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        ##计算每个点到prompt点的距离，赋值给第9位
        current_points[:, 9] =  np.min(distance_matrix, axis=1)
        ##第10、11位置0
        current_points[:, 10] = np.zeros((num_point))
        current_points[:, 11] = np.zeros((num_point))
        ##输入模型
        points = current_points.reshape((1,num_point,-1))

        points = torch.Tensor(points)
        points = points.float().cuda()
        points = points.transpose(2, 1)
        seg_pred, trans_feat = classifier(points)
        pred_val = seg_pred.contiguous().cpu().data.numpy()
        pred_val = np.argmax(pred_val, 2)
        pred_val = np.reshape(pred_val,[num_point,-1])
        ##结果concat给切割过的data
        print(np.shape(selected_points),np.shape(pred_val))
        result = np.concatenate((selected_points,pred_val),axis=-1)
        ##保存
        np.savetxt('test0.txt',result)

        round = 1
        while True:
            positive_new_click = eval(input("请输入正点击点: "))
            nagetive_new_click = eval(input("请输入负点击点："))
            print(positive_new_click,nagetive_new_click)
            if positive_new_click:
                positive_prompt = np.concatenate((positive_prompt,np.array([positive_new_click])),axis=0)
            elif nagetive_new_click:
                nagetive_prompt = np.concatenate((nagetive_prompt,np.array([nagetive_new_click])),axis=1)
            else:
                print("无法都为空")
                sys.exit(1)

            ##计算每个点到prompt点的距离，赋值给第9位
            distances = cdist(selected_points[:,:3], positive_prompt, 'euclidean')
            current_points[:, 9] =  np.min(distances, axis=1)
            ##第10、11位置0
            if nagetive_prompt:
                distances = cdist(selected_points[:,:3], nagetive_prompt, 'euclidean')
                current_points[:, 10] = np.min(distances, axis=1)
            current_points[:,11] = pred_val.reshape((num_point))

            ##输入模型
            points = current_points.reshape((1,num_point,-1))

            points = torch.Tensor(points)
            points = points.float().cuda()
            points = points.transpose(2, 1)
            seg_pred, trans_feat = classifier(points)

            pred_val = seg_pred.contiguous().cpu().data.numpy()
            pred_val = np.argmax(pred_val, 2)
            pred_val = np.reshape(pred_val,[num_point,-1])
            ##结果concat给切割过的data
            print(np.shape(selected_points),np.shape(pred_val))
            result = np.concatenate((selected_points,pred_val),axis=-1)
            ##保存
            np.savetxt('test%f.txt'%(round),result)
            round += 1

if __name__ == '__main__':   
    args = parse_args()
    main(args)

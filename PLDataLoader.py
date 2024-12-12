import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist  

def save_points_to_random_file(points, base_dir='./', prefix='points_', suffix='.txt'):
    # 生成一个随机整数作为文件名的一部分
    random_number = random.randint(1, 1000000)  # 生成 1 到 1000000 之间的随机整数
    unique_filename = f"{prefix}{random_number}{suffix}"
    
    # 确保保存路径存在
    os.makedirs(base_dir, exist_ok=True)
    
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, unique_filename)
    
    # 保存点云数据到文件
    np.savetxt(file_path, points)
    
    print(f"Points saved to: {file_path}")
    return file_path
class PLDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=65536, block_size=50.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        self.train_dir = 'train'
        self.test_dir = 'test'

        if split == 'train':
            complete_path = os.path.join(data_root, self.train_dir)
        else:
            complete_path = os.path.join(data_root, self.test_dir)

        scene_names = os.listdir(complete_path)
        # print(scene_names)
        
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        # labelweights = np.ones(13)

        for scene_name in tqdm(scene_names, total=len(scene_names)):
            print(scene_name)
            pc_path = os.path.join(complete_path, scene_name)
            room_data = np.load(pc_path)  # xyzl, N*4
            points, labels = room_data[:, 0:3], room_data[:, 3]  # xyzl, N*3; l, N
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        self.labelweights = np.ones(2)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(scene_names)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 3
        labels = self.room_labels[room_idx]   # N
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels!=0]
        #instance_num = np.random.choice(unique_labels)
        N_points = points.shape[0]
        
        while (True):
            choose_instance = np.random.choice(unique_labels)
            instance_point_idx = np.where(labels== choose_instance)[0]
            center_idx = np.random.choice(instance_point_idx)
            center = points[center_idx][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        click_set = np.array([[0,0,center[2]]])#N*3

        selected_points = points[selected_point_idxs, :]  # num_point * 4

        current_points = np.zeros((self.num_point, 12))  # num_point * 10
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        # 点击编码
        distance_matrix = cdist(selected_points[:,:3], click_set) 

        current_points[:, 0:3] = selected_points
        current_points[:, 3:6] = selected_points

        current_points[:, 9] =  np.min(distance_matrix, axis=1)
        current_points[:, 10] = np.zeros((self.num_point))
        current_points[:, 11] = np.zeros((self.num_point))

        current_labels = labels[selected_point_idxs]
        instance_labels = (current_labels == choose_instance).astype(int) 
        if self.transform is not None:
            current_points, instance_labels = self.transform(current_points, instance_labels)
        return current_points, instance_labels,click_set

    def __len__(self):
        return len(self.room_idxs)

# if __name__ == '__main__':
#     data_root = '/public/home/lh/lh/czh/powerline'
#     num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

#     point_data = PLDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
#     print('point data size:', point_data.__len__())
#     print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
#     print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
#     import torch, time, random
#     manual_seed = 123
#     random.seed(manual_seed)
#     np.random.seed(manual_seed)
#     torch.manual_seed(manual_seed)
#     torch.cuda.manual_seed_all(manual_seed)
#     def worker_init_fn(worker_id):
#         random.seed(manual_seed + worker_id)
#     train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
#     for idx in range(4):
#         end = time.time()
#         for i, (input, target) in enumerate(train_loader):
#             print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
#             end = time.time()
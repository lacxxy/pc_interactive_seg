import os
import sys
import numpy as np
import laspy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '/public/home/lh/lh/czh'
sys.path.append(BASE_DIR)


output_folder = os.path.join(ROOT_DIR, 'powerline_data/Z/test')
data_path = '/public/home/lh/lh/czh/powerline/Z/test'

file_names = os.listdir(data_path)
def load_pc_las(filename):
    las=laspy.read(filename)
    las_x=np.array(las.x)
    las_y=np.array(las.y)
    las_z=np.array(las.z)
    pt=np.stack([las_x,las_y,las_z],axis=1)#N*3
    las_ins=np.reshape(np.array(las.ins),[-1,1])
    pt=np.hstack([pt,las_ins])
    return pt.astype(np.float64)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for file_name in file_names:
    out_filename = file_name.split('.')[0]+'.npy'
    comlete_path = os.path.join(data_path,file_name)
    save_path = os.path.join(output_folder,out_filename)
    print(save_path,comlete_path)
    pc_data = load_pc_las(comlete_path)#xyz实例标签
    ##归一化
    xyz_min = np.amin(pc_data, axis=0)[0:3]

    pc_data[:, 0:3] -= xyz_min
    np.save(save_path,pc_data)




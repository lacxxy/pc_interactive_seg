import numpy as np
data = np.load('/public/home/lh/lh/czh/powerline_data/Z/train/zouchuan_3_4_1.npy')
print(data)
np.savetxt('./test.txt',data)
import numpy as np
data = np.load('/public/home/lh/lh/czh/stanford_indoor3d/Area_1_office_21.npy')
print(data)
np.savetxt('./test.txt',data)
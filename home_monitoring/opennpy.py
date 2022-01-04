import numpy as np
import glob

# Write npy
#test_arr = [0,1,2,3,4,5,6,7,8,9]
#np.save(str("./npy/test.npy"), np.array(test_arr), allow_pickle=True)

# Read npy
path = './npy'
for np_name in glob.glob(path + '/*.npy'):
    data = np.load(np_name, allow_pickle=True)
    print("Reading", np_name[len(path)+1:], "shape", data.shape)
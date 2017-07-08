from scipy import ndimage
import numpy as np

def pics2array(path, pic_names):
    X = list()
    
    for pic_name in pic_names:
        pic = ndimage.imread(path + pic_name)
        pic = pic.transpose(2, 0, 1)
        
        X.append(pic)
        
    return np.asarray(X)
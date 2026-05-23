import numpy as np 
def stictch_patches(patches: np.ndarray , image_size = 256 , patch_size = 64):
    reconstructed = np.zeros((image_size , image_size) , dtype = np.float32)
    idx = 0 
    for i in range(0  , image_size , patch_size):
        for j in range(0 , image_size , patch_size):
            reconstructed[i:i+patch_size , j:j+patch_size] = patches[idx , 0 ]
            idx = idx+1
    
    return reconstructed 


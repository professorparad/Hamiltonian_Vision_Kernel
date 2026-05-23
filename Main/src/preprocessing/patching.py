import numpy as np
def extract_patches(image: np.ndarray , patchsize = 64 ):
    height , width = image.shape 
    patches = []
    positions = []
    for i in range(0 , height , patchsize):
        for j in  range(0 , width , patchsize):
            patch = image[i: i+patchsize , j: j+width]
            patches= patches.append(patch)
            positions.append([i /height , j / width ])
    
    patches = np.array(patches , dtype=np.float32)
    positions = np.array(positions , dtype= np.float32)
    return patches , positions 
    
            
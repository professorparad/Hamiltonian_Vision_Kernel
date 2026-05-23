import numpy as np 
def blend_seams(image:np.ndarray , patch_size: int = 64 , blend_width : int = 10):
    original = image.copy()
    weight = np.ones_like(original , dtype= np.float32)
    for seam in range(patch_size , image,shape[0] , patch_size):
        for d in range(blend_width):
            alpha = d /blend_width
            for pos in [seam - d , seam+d]:
                if 0 <= pos < image.shape[0]:
                    weight[pos , :] = np.minimum(weight[pos , :] , alpha )
                    weight[: , pos] = np.minimum(weight[: ,pos] ,alpha )
    blurred = box_blur(original)
    return np.clip(weight*original + (1.0 -weight)*blurred , 0.0 , 1.0 )

def box_blur(image: np.ndarray , radius : int = 4 ):
    kernel = np.ones(2*radius +1 , dtype = np.float32)
    kernel /= kernel.size 
    return np.apply_along_axis(lambda row: np.convolve(row,kernel,mode="same"),axis=1,
                               arr=np.apply_along_axis( lambda col: np.convolve(col, kernel, mode="same"),
                                                       axis=0, arr=image))
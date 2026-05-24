import cv2 
import numpy as np 
def load_image_grayscale(img_path:str , size : tuple =(256 , 256)):
    img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE)
    if img is None :
        raise FileNotFoundError(f"Image not found  at :{img_path}")
    img = cv2.resize(img , size).astype(np.float32)
    img = img/255.0
    return img 
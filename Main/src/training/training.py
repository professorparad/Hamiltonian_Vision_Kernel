import torch 
import torch.optim as optim 
import numpy as np 
from src.preprocessing.image_loader import(load_image_grayscale)
from src.preprocessing.patching import(extract_patches)
from src.preprocessing.positional_encoding import(sinusodial_encoding)
from src.tensornetworks.mps_features import(extract_mps_features)
from src.quantum.circuit import(observable_dim)
from src.quantum.quantum_model import(QuantumModel)
from src.decoder.patch_decoder import(PatchDecoder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = "/home/adminpc/Desktop/HVK/Script/Hamiltonian_Vision_Kernel/Main/data/monalisa.jpg"
def build_dataset():
    image = load_image_grayscale(img_path)
    patches, positions = extract_patches(image , patchsize=64)
    features = np.array([extract_mps_features(p) for p in patches])
    
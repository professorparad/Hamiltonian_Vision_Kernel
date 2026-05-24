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
    features = torch.tensor(features,dtype = np.float32)
    features = (features-features.mean(dim=0))/(features.std(dim=0)+1e-8)
    postions = sinusodial_encoding(positions , d_model = 8)
    targets = torch.tensor(patches , dtype = np.float32).unsqueeze(1)
    return(features.to(device) , positions.to(device) , targets.to(device))

def train():
    features , positions , targets = build_dataset()
    model = QuantumModel(feature_dim=features.shape[1] , positional_dim= positions.shape[1]).to(device)
    decoder = PatchDecoder(observable_dim = observable_dim , positional_dim =positions.shape[1] ,patch_size = 64 ).to(device)
    optimizer = optim.Adam(list(model.parameters())+ list(decoder.parameters()) , lr = 0.003 )
    for step in range(120):
        model.train()
        optimizer.zero_grad()
        observables , energies = model(features , positions)
        output = decoder(observables , positions)
        reconstruction_loss = torch.mean((output -targets )**2)
        energy_loss = torch.mean(energies)
        loss = (reconstruction_loss+ 0.01*energy_loss)
        loss.backward()
        optimizer.step()
        if step%20 == 0 :
            print(
                f"step:{step:> 4d}|"
                f"loss :{loss.item():.6f} |"
                f"recon : {reconstruction_loss.item():.6f}|"
                f"Energy:{energy_loss.item():.6f}"
            )
    return model , decoder 


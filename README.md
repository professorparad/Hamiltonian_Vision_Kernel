# Hamiltonian_Vision_Kernel
A spatially aware hybrid autoencoder framework responsible for reconstruction of high quality images using tensor network contraction and field peturbation 

*NOTE:*
This is a research documenetary mainly used for Research Analysis and record my current ongoing project it is still not ready for open source and viable usage.

The concept essentially comes by thinking of a spatially aware , autoencoding framework which uses hamiltonian cost function for training the model.
The hamiltonina used here is the Hisenberg Hamiltonian which acts as primary filter training the model in specifc weights instead of random weights giving accurate results than an usually used quantumn autoencoding framework
Before feeding into the Variational Model the Model First goes to 4 layers:
1. The Patching layer: Here the image is broken into patches for easier computation
2. The MPS layer: this layers uses hard coded SVD to flatten each patches and compress the data to a specififc dimensional tensor 
3. The positional encoded layer : this layer essentially provides and encoded each patch with spatial information using Fourier analysis
4. The Feature Extracter: from the mps layer we compute observables like expectation values of X , Z and interpatch correlation and generate the mps feature array then this Mps feature array is added with positional encoded array to make the overall feature space which is then feed into the variational circuit.

this the overall architecture of the proposed kernel

*/NOVELITY AND CREDIBILITY CHECK/* 
1. The individual component like MPS , Hamiltonain or positional Encoding are not novel in the sense cause the MPS is pretty common in tensor networks and Hamiltonian based Energy functions are used in every day quantum algorthims specifically the Ising Hamiltonian and the Heisenberg Hamiltonian for algorithms like auto-optimization and error mitigiation moreoverthe positional encoding framework is mostly used in the Classical Vision transformers.
2. the Novelity Lies in the intersection of all these and the concept of using peturbations for image reconstruction using the Quantum ML.
3. Recent studies have shown that the most of the models mainly focus on the Building models used for image classification of particular dataset using hard encoded Hybrid Models but do not answer the fundamental question that can we define or construct a hybrid framework which works as closely as a classical autoencoder using Quantum latent features or is it to stretch of an question for current quantumn models.
4. My model essentially uses the concept of spin movements. See think of each positional encoded pixel as spins. Each spin can have values 1 . -1. Now when the positonal Dependance is added and the rotations are viewed in the variantional Circuit the circuit learns different structures based on the energy and the realtion poositonals of each pixels its more like a spins state in a external field which slightly changes the overal hamiltonian depening upon the position of the spins.

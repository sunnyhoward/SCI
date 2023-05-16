# Snapshot Compressive Imaging. 

In this repo we work on the task of going from the image captured on the sensor (possibly 9 copies of the beam with differing dispersions), to the cube. 

For each copy, we can potentially try and first un-integrate and un-disperse that copy, but maintaining the blur from dispersion. Then we will put the cubes together and unblur later. 

## Models
The ML models and the helper functions
- Implement Wiener deconvolution instead of just division. I also like the idea of making the fourier-kernel trainable.
- Implement CoordGate to 
- Maybe make a model to analyse each copy separately, before bringing them together. 

## Forward
The forward models.
- Try and get padding working with the Fourier method.
- Get normalization working in calc_psiT_g. 


## Other Qs

#### Training
- It should be easy to overfit the training data position. Right now we are not. 
- I suspect there is some CUDA torch memory inefficiency. 


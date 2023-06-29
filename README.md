# Snapshot Compressive Imaging. 

In this repo we work on the task of going from the image captured on the sensor (possibly 9 copies of the beam with differing dispersions), to the cube. 

For each copy, we can potentially try and first un-integrate and un-disperse that copy, but maintaining the blur from dispersion. Then we will put the cubes together and unblur later. 


We have worked extensively on learning the forward model in position and spectrum. This should hopefully make it alot easier to train on fts data now. See in `models/notebooks/kernel_learner.ipynb`


## Models
The ML models and the helper functions.
- Right now, the undispersed cube contains some spatially varying spectral modulation already. 


## Forward
The forward models.
- Try and get padding working with the Fourier method.
- Get normalization working in calc_psiT_g. 
- The Mask needs to be made correctly.


## Other Qs


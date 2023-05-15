## Snapshot Compressive Imaging. 

In this repo we work on the task of going from the image captured on the sensor (possibly 9 copies of the beam with differing dispersions), to the cube. 

For each copy, we can potentially try and first un-integrate and un-disperse that copy, but maintaining the blur from dispersion. Then we will put the cubes together and unblur later. 

### models
The ML models and the helper functions

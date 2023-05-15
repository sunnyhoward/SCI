## Forward Models

This section will contain the forward models (and their inverses). Currently there are two: Rolling, and a Fourier method. 

There is a big difference in the implementations right now however... The rolling just disperses cubes around the center of the copy (ie, if you have 9 copies in space, the 'undispersed' will still have 9 separated copies but the mask spots will overlap). In fourier, the 'undispersed' copies all come back to the center of one cube.
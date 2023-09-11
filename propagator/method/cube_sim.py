import torch
import numpy as np
import torch.nn.functional as F



def generate_cube(nx, rotation, supergaussian_order = 1):

    mask_cube = create_mask(pixel_size=1,  pixel_spacing=8, nx=nx)

    hyp_intensity = create_spatiotemporal_intensity(mask_cube.shape, n_zernike = 15)

    data_cube = mask_cube * hyp_intensity

    data_cube = rotate_mask(data_cube, rotation=rotation)

    data_cube = data_cube * gaussian2d(nx, [0,0], [0.6,0.6], factor=supergaussian_order)

    data_cube = blur_mask(data_cube, kernel_size=5, sigma=0.1)

    return data_cube



def blur_mask(mask, kernel_size=5, sigma=1):
    
    weight = gaussian2d(kernel_size,[0,0],[sigma,sigma]).tile(mask.shape[1],mask.shape[1],1,1)
    mask = torch.nn.functional.conv2d(mask, weight,  stride=1, padding='same')

    return mask


def create_mask(pixel_size=1,  pixel_spacing=8, nx=640):
    
    mask = torch.zeros((1,21,nx, nx))

    for i in range(0, nx, pixel_size + pixel_spacing):
        for j in range(0, nx, pixel_size + pixel_spacing):
            mask[0,:,i:i+pixel_size, j:j+pixel_size] = 1

    return mask


def rotate_mask(mask, rotation=0):


    rotation = torch.tensor(rotation)
    theta = torch.zeros((1,2,3))

    theta[:,0,0] = torch.cos(rotation)
    theta[:,0,1] = -torch.sin(rotation)
    theta[:,1,0] = torch.sin(rotation)
    theta[:,1,1] = torch.cos(rotation)

    grid = F.affine_grid(theta, mask.size(), align_corners=False)
    rot_mask = F.grid_sample(mask, grid, align_corners=False) 

    return rot_mask


def gaussian_mixture(num_components, nx):
    
    # Generate random mean values for each component
    mean_values = torch.rand(num_components, 2)
    std_values = torch.rand(num_components, 2)*0.2
    
    # Generate the Gaussian mixture
    mixture = np.zeros((nx,nx))
    for i in range(num_components):
        gaussian = gaussian2d(nx, mean_values[i], std_values[i])
        mixture += gaussian

    gaussian *= gaussian2d(nx, [0.5,0.5], [0.2,0.2])
    
    return mixture


def gaussian2d(nx, mean, sigma, factor=2):
    
    x, y = torch.meshgrid(torch.linspace(-1,1,nx), torch.linspace(-1,1,nx))

    # return torch.exp(-((x-mean[0])**2/(2*sigma[0]**2) ** factor + (y-mean[1])**2/(2*sigma[1]**2) ** factor)).unsqueeze(0).unsqueeze(0) #supergaussian definition from wiki
    return torch.exp(-(((x-mean[0])/(np.sqrt(2)*sigma[0])) ** factor + ((y-mean[1])/(np.sqrt(2)*sigma[1])) ** factor)).unsqueeze(0).unsqueeze(0) #supergaussian definition from wiki



def gaussian1d(nc, mean, sigma, factor=2):
    
    c = torch.linspace(-1,1,nc)

    # return torch.exp(-((x-mean[0])**2/(2*sigma[0]**2) ** factor + (y-mean[1])**2/(2*sigma[1]**2) ** factor)).unsqueeze(0).unsqueeze(0) #supergaussian definition from wiki
    return torch.exp(-2*(((c-mean)/sigma) ** factor)).unsqueeze(1).unsqueeze(1).unsqueeze(0) #supergaussian definition from wiki




def create_spatiotemporal_wavefront(size, n_zernike = 15):
    '''
    Create a spatiotemporal intensity map with zernike polynomials that have a relationship in frequency.
    Make a random set of zernikes, and then for one order we make a linear function with wl

    no_examples: number of examples to create
    size: size of the intensity map
    '''

    no_examples,nλ,nx,ny = size

    wavefront = torch.zeros(size)


    for i in range(no_examples):

        zernikes = (np.random.rand(nλ,n_zernike) - 0.5) * 0.03

        for _ in range(2):
            ind = np.random.randint(n_zernike)
            fac = np.random.rand(1) * 2 +1
            fac = np.linspace(0,1,nλ) ** fac - np.random.rand() + (np.random.rand(nλ) -0.5)*0.01
            val = np.random.randint(2) #up or down
            zernikes[:,ind ] = val * fac + (1-val) * np.flip(fac)



        zernikes = torch.tensor(zernikes).float()

        for n in range(nλ):
            wavefront[i,n] = expand_zernikes(15,zernikes[n],shape = size[2:], normalize=True)

    return wavefront



def create_spatiotemporal_intensity(size, n_gaussian = 15):
    '''
    This will just be a superposition between two gaussians.

    no_examples: number of examples to create
    size: size of the intensity map
    '''

    no_examples,nλ,nx,ny = size

    intensities = torch.zeros(size)

    
    for i in range(no_examples):

        intensity1= gaussian2d(nx, [0,0], [0.3,0.3], factor=2) #(1,1,nx,nx)

        intensity1 = intensity1.tile(1,nλ,1,1) #(1,nλ,nx,nx)

        spectralgaussian = gaussian1d(nλ, 0.5, 0.2, factor=2) #(1,nλ,1,1)


        intensities[i] = (intensity1 * spectralgaussian)[0]

    intensities = (intensities - torch.amin(intensities,dim=(1,2,3),keepdim=True)) / (torch.amax(intensities,dim=(1,2,3),keepdim=True) - torch.amin(intensities,dim=(1,2,3),keepdim=True))

    return intensities



def expand_zernikes(orders, weights, shape, normalize=True):
    """
    Function to generate zernike polynomials, normalized between -2pi and 2pi

    Parameters
    ----------
    orders : 2d array of pairs of m and n
        m = azimuthal degree, n = radial degree.
    
    dim : int
        dimensions of phase map
    
    weights : 1d array
        how to weight each zernike pair.

    Returns
    -------
    normalized : mesh
        phase map.

    """
    
    xx = torch.linspace(-1,1,shape[0])
    yy = torch.linspace(-1,1,shape[1])

    xx,yy = torch.meshgrid(xx,yy,indexing='ij')

    r = np.sqrt(xx**2+yy**2)
    phi = np.arctan2(yy,xx)

    n,m = [],[]
    for i in range(1,orders+1):
        n.append(noll_to_zern(i)[0])
        m.append(noll_to_zern(i)[1])
    
    n = torch.tensor(n)
    m = torch.tensor(m)

    output = torch.zeros((len(m),phi.shape[0],phi.shape[1]))

    

    m_mod = np.abs(m)


    ends = (n-m_mod)/2+1


    for i in range(len(m)):
        k=torch.arange(int(ends[i]))
        output[i] += torch.sum(r**(n[i]-2.0*k)[:,None,None] * ((-1.0)**k *factorial(n[i]-k)/(factorial(k) * factorial((n[i]+m_mod[i])/2.0 - k) * factorial((n[i]-m_mod[i])/2.0 - k)))[:,None,None],axis=0)


    output[m>=0] *= torch.cos(m[m>=0][:,None,None] * phi)
    output[m<0] *= torch.sin(-m[m<0][:,None,None] * phi)


    norm = torch.ones_like(weights)

    if normalize:
        norm[:] = torch.sqrt(2*(n+1))
        norm[m==0] = torch.sqrt(n[m==0]+1)


    output*= (weights*norm)[:,None,None]

    output = torch.sum(output,dim=0)
    
    return output



def factorial(x):
    if type(x) == int:
        if x == 0:
            fac = 1
        else:
            fac = torch.prod(torch.arange(1,x))
    else:   
        fac = torch.zeros_like(x)
        for i in range(len(x)):
            if x[i] == 0:
                fac[i] = 1
            else:
                fac[i]=torch.prod(torch.arange(1,x[i]))
    return fac


def noll_to_zern(j): ###STOLEN FROM LIGHTPIPES
    """
    *Convert linear Noll index to tuple of Zernike indices.*
    
    :param j: the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike index.
    :type j: int, float
    :return: name of Noll Zernike term
    :rtype: string (n, m) tuple of Zernike indices
    
    .. seealso::
    
        * :ref:`Manual: Zernike polynomials.<Zernike polynomials.>`
        * `https://oeis.org <https://oeis.org/A176988>`_
        * `Tim van Werkhoven, https://github.com/tvwerkhoven <https://github.com/tvwerkhoven>`_
        * :ref:`Examples: Zernike aberration.<Zernike aberration.>`
    """

    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")

    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n

    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
    return (n, m)
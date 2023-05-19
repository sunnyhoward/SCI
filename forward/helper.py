import torch
import numpy as np


def create_spatiotemporal_intensity(no_examples, size, n_zernike = 15, nλ = 21):
    '''
    Create a spatiotemporal intensity map with zernike polynomials that have a relationship in frequency.
    Make a random set of zernikes, and then for one order we make a linear function with wl

    no_examples: number of examples to create
    size: size of the intensity map
    '''

    intensities = torch.zeros(no_examples, nλ, size[1], size[2])



    weights = np.flip(np.arange(1,n_zernike,1))**3
    p = weights/np.sum(weights)

    for i in range(no_examples):

        main_aberration = np.random.choice(np.arange(1,n_zernike), p = p)
        zernikes = np.random.rand(nλ,n_zernike)/((1+np.abs(np.arange(0,n_zernike, 1)-main_aberration))**(2*(np.random.rand(1)+0.2)))[None,:]
        

        main_spec = np.linspace(0.5*np.mean(zernikes[:,main_aberration]),1.5*np.mean(zernikes[:,main_aberration]),nλ) + (np.random.rand(nλ) -0.5)*0.01

        
        val = np.random.randint(2) #up or down
        zernikes[:,main_aberration] = val * main_spec + (1-val) * np.flip(main_spec)

        zernikes = torch.tensor(zernikes).float()

        for n in range(nλ):
            intensities[i,n] = expand_zernikes(15,zernikes[n],dim = size[1], normalize=True)

    return intensities



def expand_zernikes(orders, weights, dim = 1024, normalize=True):
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
    
    xx = torch.linspace(-1,1,dim)
    yy = torch.linspace(-1,1,dim)

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
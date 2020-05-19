import odl
import numpy as np
import torch

def random_shapes(interior=False):
    if interior:
        x_0 = np.random.rand() - 0.5
        y_0 = np.random.rand() - 0.5
        
    else:
        x_0 = 2 * np.random.rand() - 1.0
        y_0 = 2 * np.random.rand() - 1.0
        

    return ((np.random.rand() - 0.5) * np.random.exponential(0.4),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            x_0, y_0,
            np.random.rand() * 2 * np.pi)

def random_phantom(spc, n_ellipse=100, interior=True, form='ellipse'):
    n = np.random.poisson(n_ellipse)
    shapes = [random_shapes(interior=interior) for _ in range(n)]
    if form == 'ellipse':
        result = odl.phantom.ellipsoid_phantom(spc, shapes)
    elif form == 'rectangle':
        result = odl.phantom.cuboid_phantom(spc, shapes)
    else:
        raise Exception('unknown form')
        
    result = np.clip(result, 0, 1)
    result /= np.std(result) + 1e-5
    return result

def summary_image_impl(writer, name, tensor, it):
    image = tensor[0, 0]
    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
    writer.add_image(name, image, it, dataformats='HW')



    

def summary_image(writer, name, tensor, it, window=False):
    summary_image_impl(writer, name + '/full', tensor, it)
    if window:
        summary_image_impl(writer, name + '/window', (tensor), it)        
   
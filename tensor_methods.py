import sympy as sp
import numpy as np 

from Tensor import *

def transpose(array, components):
    """ just a wrapper """
    return np.transpose(array, components)


def get_transposition_structure(old_indices, new_indices):

    idx_map = list(map(lambda x : old_indices.index(x), new_indices))
    return idx_map

def indexed_transpose(array, old_indices, new_indices):
    idx_map = get_transposition_structure(new_indices, old_indices)
    return transpose(array, idx_map)
    
    
def contract(array, contractions):
    """ 
        in general case if we need to contract dimensions i and j:
        ---> array.diagonal(axis1=i, axis2=j).sum(axis=-1)
        gives needed answer
    """ 
    # contractions_left is either a tuple or a list of tuples axes to contract
    # contractions_left just a copy of contractions, 'cause we're going original
               
    if isinstance(contractions, list):
        contractions_left = contractions[:]
        contr_number = len(contractions_left)
        for i in range(contr_number):
            idc = contractions_left.pop()
            array = array.diagonal(axis1=idc[0], axis2=idc[1]).sum(axis=-1)
            contractions_left = list(map(lambda x, y: (x - 1*(x>idc[0]) - 1*(x>idc[1]), \
                                y- 1*(y>idc[0]) - 1*(y>idc[1])), contractions_left))
    else:
        array = array.diagonal(axis1=contractions[0], axis2=contractions[1]).sum(axis=-1)
    
    return array


def tensor_product(first_array, second_array):

    # this hack, because tensor product of 2 numbers, which are tensors of rank (1,)
    # is again a number (1,) not a (1, 1)
    a = first_array.shape
    b = second_array.shape
    if a == (1,): 
        a = ()
    elif b == (1,):
        b = ()
    return np.outer(first_array, second_array).reshape(a + b)
          

def get_slice(array, slice_range):
    """
    slice_range is a list of integers or indices:
    for example slice_range = [not int, not int, 1] will return
    slice [:, :, 1] 
    """

    rank       = sum(map((lambda x: not isinstance(x, int)), slice_range))
    components = [(i, slice_range[i]) for i in range(len(slice_range))]
    extract    = list(filter(lambda x : isinstance(x[1], int), components))
    while extract != []:
        elem = extract.pop()
        array = np.take(array, [elem[1]], axis=elem[0])
        extract = list(map(lambda x: (x[0] - 1*(x[0] > elem[0]) , x[1]), extract))
    array = np.squeeze(array)
    return (rank, array)


def eliminate_contractions(array, indices):
    
    contractions = [(i,j) for i in range(len(indices)) for j in range(i+1, len(indices))\
                        if indices[i].is_conjugate(indices[j])] 
    if contractions:
        array = contract(array, contractions)
        dummies = [x for (x,y) in contractions] + [y for (x,y) in contractions]
        indices = [indices[i] for i in range(len(indices)) if i not in dummies]
        
    return (array, indices)


def get_transformation_from_index(old_idx, new_idx):
    
    transformation = None
    
    if old_idx.chart != new_idx.chart:
        print(old_idx, new_idx)
        transformation = old_idx.chart.manifold.get_transformation(\
                                 old_idx.chart, new_idx.chart)
       
    if new_idx.covar != old_idx.covar:
        if new_idx.covar > 0:
            g = new_idx.chart.metric.guu
        else:
            g = new_idx.chart.metric.gdd
            
        if transformation is None:
            transformation = g
        else:
            transformation = tensor_product(g, transformation)
            
    return transformation
        


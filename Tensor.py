import sympy as sp
import numpy as np
from collections import Counter
from sympy.core.compatibility import is_sequence

import tensor_methods as t_meth

covar_dict     = {"u" : 1, "d" : -1}
inv_covarar_dict = {1 : "u", -1 : "d"}


class IndexException(Exception):
    pass


class SumIdx(sp.Basic):
    
    def __init__(self, label, cov = 1, chart = None):
        super(SumIdx, self).__init__()
        if isinstance(label, str):
            label = sp.Symbol(label, integer=True)
        self._label = label
        self._covar = cov
        self._chart = chart
        
    @property
    def label(self):
        return self._label
    
    @property
    def covar(self):
        return self._covar
    
    @property
    def chart(self):
        return self._chart
    
    def is_conjugate(self, index):
        if self._label == index.label and self._covar * index.covar < 0:
            return True
        else:
            return False
    
    def __str__(self):
        if self._covar > 0:
            return self._label.name
        else:
            return "-" + self._label.name
    
    def __neg__(self):
        return SumIdx(self._label, self._covar * (-1), self._chart)
    
    def __eq__(self, other):
        return self._chart == other.chart and self._label == other.label \
                                          and self._covar == other.covar
        
    
class Metric(object):
    
    def __init__(self, shape, chart, array):
        super(Metric, self).__init__()

        if isinstance(shape, str):
            shape = list(map(covar_dict.get, shape))
        
        if len(shape) != 2 or shape[0]*shape[1] < 0:
            raise IndexException("Metric error 1.")
            
        tmp = sp.Matrix(array)
        inv_tmp = tmp.inv()
        
        self._det = tmp.det()
        
        if shape == [-1, -1]:
            guu = inv_tmp.tolist()
            gdd = tmp.tolist()
        else:
            guu = tmp.tolist()
            gdd = inv_tmp.tolist()
            
        self._uu = np.array(guu, dtype = object)
        self._dd = np.array(gdd, dtype = object)
        self._ud = np.array(sp.eye(chart.dim).tolist(), dtype = object)
        
        self._chart = chart
        chart.set_metric(self)
        
    @property
    def manifold(self):
        return self._chart._manifold
    
    @property
    def chart(self):
        return self._chart
    
    @property
    def dim(self):
        return self._chart.dim
    
    @property 
    def guu(self):
        return self._uu
    
    @property
    def gdd(self):
        return self._dd
    
    @property
    def det(self):
        return self._det
    
    def __getitem__(self, indices):
        
        if indices[0].covar == -1:
            if indices[1].covar == -1:
                return Tensor(list(indices), self._dd)
            else:
                return Tensor(list(indices), self._ud)
        else:
            if indices[1].covar == -1:
                return Tensor(list(indices), self._ud)
            else:
                return Tensor(list(indices), self._uu) 


class CoordinateChart(object):
    
    def __init__(self, label, coords):
        super(CoordinateChart, self).__init__()
        self._label = label   
        self._metric = None
        self._coords = coords
        self._dim = len(coords)
        
        
    @property 
    def label(self):
        return self._label
    
    @property
    def dim(self):
        return self._dim
        
    @property
    def metric(self):
        return self._metric
    
    def coords(self, index):
        return Tensor([get_dummy_idx(1, index.chart)], \
                       np.array(self._coords))[index]
             
    def set_metric(self, metric):
        self._metric = metric
    
    @property
    def raw_coords(self):
        return self._coords
    
    def assign_indices(self, indices):
        for i in indices:
            i._chart = self



class Tensor(object):
    
    def __init__(self, indices, array=None):
        
        
        if [(i,j) for i in range(len(indices)) \
                    for j in range(i+1, len(indices)) \
                        if indices[i] == indices[j]]:
            raise IndexException("Same index appears at least twice in tensor.")
        
        (array, indices) = t_meth.eliminate_contractions(array, indices)
        
        self._indices = indices
        self._covar = [x.covar for x in indices]

        self.set_components(array)

    
    @property
    def indices(self):
        return self._indices
    
    @property
    def chart(self):
        return zip(self._indices, self._indices.chart)
    
    @property
    def covar(self):
        return self._covar   
          
    @property
    def shape(self):
        return self._shape
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def as_array(self):
        """ needs a better name ???"""
        return self._array
    
    def __str__(self):
        return "Tensor, rank: %s, shape: %s, indices: %s, components: %s" \
                %(self.rank, list(map(inv_covarar_dict.get, self._covar)), \
                  self._indices, self.as_array)
                
    def __eq__(self, other):
        return self.shape == other.shape and \
         np.all(self._array == other[tuple(self.indices)].as_array)  
      

    def set_components(self, array, components_range = None):
        
        if isinstance(array, np.ndarray):
            array = array.squeeze()
            shape = array.shape  
        elif not array is None:
            array = np.array(array, dtype=object)
            shape = array.shape
                      
        self._shape = shape
        self._rank = len(shape)
        
        if components_range is not None:
            self._array[components_range] = array
        else:       
            if array is None:
                self._array = np.empty(self.shape, dtype = object)
            else:
                if isinstance(array, np.ndarray):
                    self._array = array
                else:
                    self._array = np.array(array, dtype=object)    
 
          
    def transpose(self, new_indices):
        new_array = t_meth.indexed_transpose(self._array, self.indices, \
                                                          new_indices)
        return Tensor(new_indices, new_array)



    def _get_contraction_args(self, contractions):
        dummyes = [x[0] for x in contractions] + [x[1] for x in contractions]
        indices = [self._indices[i] for i in range(len(self._indices)) \
                                        if i not in dummyes]
        return indices

    def contract(self, contractions):
        new_array   = t_meth.contract(self._array, contractions)
        new_indices = self._get_contraction_args(contractions)
        return Tensor(new_indices, new_array)
    
    
    def tensor_product(self, other):
        new_array   = t_meth.tensor_product(self._array, other.as_array)
        new_indices = self.indices + other.indices
        return Tensor(new_indices, new_array)
    

    
    def apply_function(self, f):
        return Tensor(self.indices, f(self._array)) 
    
    def __neg__(self):
        return Tensor(self.indices, -self._array)
    
   
    def _check_conformance(self, other):
        
        if isinstance(other, type(self)):
            if self.rank != other.rank:
                raise IndexException("Tensors dimensions are different.")
        if isinstance(other, Tensor):
            if self.indices and any(filter(lambda x: x not in self.indices, \
                                               other.indices)):
                raise IndexException("Tensors indices mismatch.")
        else:
            if self.rank > 0:
                raise IndexException("Tensors dimensions are different.")
            
    def _adjust_term(self, other):
        if isinstance(other, Tensor):
            array = t_meth.indexed_transpose(other.as_array, other.indices, \
                                              self.indices)
        else:
            array = other
        return array
 
    
    def __add__(self, other):
        self._check_conformance(other)
        c = self._adjust_term(other)
        return Tensor(self.indices, self._array + c)
    
    def __sub__(self, other):
        self._check_conformance(other)
        c = self._adjust_term(other)
        return Tensor(self.indices, self._array - c)
    
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            return Tensor(self.indices, self._array * other)
        else:
            return self.tensor_product(other)
    
    def __rmul__(self, other):
        if not isinstance(other, Tensor):
            return Tensor(self.indices, self._array * other)
        else:
            return other.tensor_product(self)
        
    def __truediv__(self, divisor):
        return Tensor(self.indices, self._array/divisor)    
       
          
    def _prepare_getslice(self, indices):
        
        if len(indices) != len(self._indices):
            raise IndexException("Invalid slice.")
        
        new_array = self._array
        
        transpositions = [i for i in range(len(indices))]
        
        for idx in range(len(indices)):
            if not isinstance(indices[idx], SumIdx):
                continue
            
            transformation = t_meth.get_transformation_from_index(          \
                                    self._indices[idx], indices[idx])
            
            if transformation is not None:
                    
                tmp = transpositions[idx]   
                transpositions = [x - 1*(x < tmp) for x in transpositions]
                transpositions[idx] = 0
                    
                new_array = t_meth.tensor_product(transformation, new_array)
                new_array = t_meth.contract(new_array, (1, idx + 2))

        new_array = t_meth.transpose(new_array, transpositions)
        new_indices = list(filter(lambda x : isinstance(x, SumIdx), indices))
        return (new_array, new_indices)  
    
    
    def get_slice(self, slice_range):
        """
        form is a list of integers or other types:
        for example form = [not int, not int, 1] return
        slice [:, :, 1] because i don't see any needs of obtaining
        tensors with lower dimensions
        """
        (array, args) = self._prepare_getslice(slice_range)
        (rank, array) = t_meth.get_slice(array, slice_range)
        return Tensor(args, array)
    
    
    def __getitem__(self, indices):
        if not is_sequence(indices):
            indices = [indices]
        else:
            indices = list(indices)
        return self.get_slice(indices)           
       

    def __call__(self, args):
        return self.transpose(args)
    
              
       
def get_dummy_idx(covar, chart = None):
    return SumIdx(sp.Dummy("x"), covar, chart)



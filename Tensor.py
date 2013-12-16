import sympy as sp
import numpy as np
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
    
    
    
class FreeTensor(object):
    
    def __init__(self, array, shape = None):
        """
        an abstract object with `rank` - dimensions, each with `dim` elements
        for example matrix is rank 2 object
        
        """
        #super(FreeTensor, self).__init__()

        if isinstance(array, np.ndarray):
            array = array.squeeze()
            shape = array.shape  
        elif not array is None:
            array = np.array(array, dtype=object)
            shape = array.shape
                      
        self._shape = shape
        self._rank = len(shape)
        self.set_components(array)
        
    
    def set_components(self, array, components_range = None):
        
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
        
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def _base_arg(self):
        return self._shape
        
    @property
    def components(self):
        """ needs a better name ???"""
        return self._array
    
    
    ## This methods should be redefined for every tensor

    def _return_tensor(self, args, array):
        if not isinstance(array, np.ndarray):
            array = np.array(array, dtype = object)
        return FreeTensor(array) 
    
    def _check_conformance(self, other):
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise IndexException("Tensors dimensions are different.")
        else:
            if self.rank > 0:
                raise IndexException("Tensors dimensions are different.")      
    
    def _get_trasnpositions(self, components):
        return (components, self._shape)
       
    def _get_contraction_args(self, contractions, array):
        return array.shape
              
    def _prepare_getslice(self, slice_args):
        if len(slice_args) != self.rank:
            raise IndexException("Invalid slice")
        return (self._array, None)
    
    def _adjust_term(self, other):
        if isinstance(other, type(self)):
            return other.components
        else:
            return other   

    ### End      
          
    def transpose(self, components):
        (transpositions, args) = self._get_trasnpositions(components)
        return self._return_tensor(args, t_meth.transpose(self._array, \
                                                                transpositions))

    def contract(self, contractions):
        array = t_meth.contract(self._array, contractions)
        return self._return_tensor(self._get_contraction_args(contractions, \
                                            array), array)
    
    def tensor_product(self, other):
        array = t_meth.tensor_product(self._array, other.components)
        return self._return_tensor(self._base_arg + other._base_arg, array)
    
    def get_slice(self, slice_range):
        """
        form is a list of integers or other types:
        for example form = [not int, not int, 1] return
        slice [:, :, 1] because i don't see any needs of obtaining
        tensors with lower dimensions
        """
        (array, args) = self._prepare_getslice(slice_range)
        (rank, array) = t_meth.get_slice(array, slice_range)
        return self._return_tensor(args, array)
    
    
    def apply_function(self, f):
        return self._return_tensor(self._base_arg, f(self._array))
    
    
    def __neg__(self):
        return self._return_tensor(self._base_arg, -self._array)
    
    def __add__(self, other):
        self._check_conformance(other)
        c = self._adjust_term(other)
        return self._return_tensor(self._base_arg, self._array + c)
    
    def __sub__(self, other):
        self._check_conformance(other)
        c = self._adjust_term(other)
        return self._return_tensor(self._base_arg, self._array - c)
    
    
    def __mul__(self, other):
        if not isinstance(other, type(self)):
            return self._return_tensor(self._base_arg, self._array * other)
        else:
            return self.tensor_product(other)
    
    def __rmul__(self, other):
        if not isinstance(other, type(self)):
            return self._return_tensor(self._base_arg, self._array * other)
        else:
            return other.tensor_product(self)
        
    def __truediv__(self, divisor):
        return self._return_tensor(self._base_arg, self._array/divisor)    
       
    def __getitem__(self, indices):
        if not is_sequence(indices):
            indices = [indices]
        else:
            #### Bullshit ??????????????
            indices = list(indices)
        return self.get_slice(indices)

    def __call__(self, args):
        return self.transpose(args)
 
    def __eq__(self, other):
        return self.shape == other.shape \
               and np.all(self._array == other.components)
               
    def __str__(self):
        return "FreeTensor, rank: %s, shape: %s, components:\n %s" \
                %(self._rank, self._shape, self._array)
       

class IndexedTensor(FreeTensor):
    
    def __init__(self, indices, array, shape = None):
                    
        if [(i,j) for i in range(len(indices)) for j in range(i+1, len(indices))\
                        if indices[i] == indices[j]]:
            raise IndexException("Same index appears at least twice in tensor.")
        
        (array, indices) = t_meth.eliminate_contractions(array, indices)

        self._covar = list(map(lambda x: x.covar, indices))
        super(IndexedTensor, self).__init__(array, shape)
        
        self._indices = indices
        
   
    @property
    def covar(self):
        return self._covar   
    
    @property
    def indices(self):
        return self._indices
    
    @property
    def _base_arg(self):
        return self._indices
    
    def _return_tensor(self, args, array):
        return IndexedTensor(args, array)
    
      
    def _check_conformance(self, other):
        super(IndexedTensor, self)._check_conformance(other)
        if self.indices and any(filter(lambda x: x not in self._indices, other.indices)):
            raise IndexException("Tensors indices mismatch.")
           
    def _get_trasnpositions(self, indices):
        return (t_meth.get_transposition_structure(self.indices, indices), indices)
    
    def _get_contraction_args(self, contractions, contr_number):
        dummyes = [x[0] for x in contractions] + [x[1] for x in contractions]
        indices = [self._indices[i] for i in range(len(indices)) \
                                                if i not in dummyes]
        return indices
          
    def _get_TP_args(self, other):
        return self.indices + other.indices
    
    def _prepare_getslice(self, slice_args):
        if len(slice_args) != len(self._indices):
            raise IndexException("Invalid slice.")
        indices = list(filter(lambda x : isinstance(x, SumIdx), slice_args))

        for i in range(len(slice_args)):
            if isinstance(slice_args[i], SumIdx):
                if not self.indices[i].covar * slice_args[i].covar > 0:
                    raise IndexException("Invalid slice.")
   
        return (self._array, indices)

    def _adjust_term(self, other):
        if isinstance(other, type(self)):
            array = t_meth.indexed_transpose(other.components, self.indices, other.indices)
        else:
            array = other
        return array
    
    def __eq__(self, other):
        return self.covar == other.covar and super(IndexedTensor, self).__eq__(other)
    
    def __str__(self):
        return "IndexedTensor, rank: %s, shape: %s, indices: %s, components: %s" \
                %(self._rank, list(map(inv_covarar_dict.get, self._covar)), self._indices, self._array)
    
    
               
class Manifold(sp.Basic):
    
    def __init__(self, label, dim, charts=[]):
        super(Manifold, self).__init__()
        self._label = label
        self._dim = dim
        self._charts = charts
        
    @property
    def label(self):
        return self._label
    
    @property
    def dim(self):
        return self._dim
    
    def add_chart(self, chart):
        self._charts.append(chart)
        
    def get_transformation(self, old_chart, new_chart):
        ### TODO
        return None
        
        
        
class CoordinateChart(sp.Basic):
    
    def __init__(self, label, manifold, coords):
        super(CoordinateChart, self).__init__()
        self._label = label   
        self._manifold = manifold
        self._metric = None
        self._coords = coords
        
        
    @property 
    def label(self):
        return self._label
    
    @property
    def dim(self):
        return self._manifold.dim
        
    @property
    def metric(self):
        return self._metric
    
    def coords(self, index):
        return MTensor([get_dummy_idx(1, index.chart)], np.array(self._coords))[index]
             
    def set_metric(self, metric):
        self._metric = metric
    
    @property
    def raw_coords(self):
        return self._coords
    
    def assign_indices(self, indices):
        for i in indices:
            i._chart = self
        
        
        
        
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
                return MTensor(list(indices), self._dd)
            else:
                return MTensor(list(indices), self._ud)
        else:
            if indices[1].covar == -1:
                return MTensor(list(indices), self._ud)
            else:
                return MTensor(list(indices), self._uu) 

  
  
  
class MTensor(IndexedTensor):
    
    def __init__(self, indices, array=None):
        super(MTensor, self).__init__(indices, array)  
    
    @property
    def indices(self):
        return self._indices
    
    @property
    def chart(self):
        return zip(self._indices, self._indices.chart)
       
    def _return_tensor(self, args, array):
        return MTensor(args, array)
   
    def _check_conformance(self, other):
        if isinstance(other, type(self)):
            if self.rank != other.rank:
                raise IndexException("Tensors dimensions are different.")
            if self.indices and any(filter(lambda x: x not in self._indices, other.indices)):
                raise IndexException("Tensors indices mismatch.")
        else:
            if self.rank > 0:
                raise IndexException("Tensors dimensions are different.")  
        
   
   
    def _prepare_getslice(self, indices):
        if len(indices) != len(self._indices):
            raise IndexException("Invalid slice.")
        
        array = self._array
        
        transpositions = [i for i in range(len(indices))]
        
        for idx in range(len(indices)):
            if not isinstance(indices[idx], SumIdx):
                continue
            
            transformation = t_meth.get_transformation_from_index(\
                                    self._indices[idx], indices[idx])
            
            if transformation is not None:
                    
                tmp = transpositions[idx]   
                transpositions = list(map(lambda x : x - 1*(x < tmp), transpositions))
                transpositions[idx] = 0
                    
                array = t_meth.tensor_product(transformation, array)
                array = t_meth.contract(array, (1, idx + 2))
                
        array = t_meth.transpose(array, transpositions)
        new_indices = list(filter(lambda x : isinstance(x, SumIdx), indices))
        return (array, new_indices)


    def __str__(self):
        return "Tensor, rank: %s, shape: %s, indices: %s, components: %s" \
                %(self.rank, list(map(inv_covarar_dict.get, self._covar)), self._indices, \
                self.components)
                
    def __eq__(self, other):
        return self.shape == other.shape and \
         np.all(self._array == other[tuple(self.indices)].components)
              
              
       
def get_dummy_idx(covar, chart = None):
    return SumIdx(sp.Dummy("x"), covar, chart)



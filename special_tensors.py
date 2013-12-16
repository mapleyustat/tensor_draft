from Tensor import *
from tensor_methods import *
import sympy as sp
import numpy as np 
 

class DerOp(object):
    
    def __init__(self, coord):
        self._coord = coord
        
    def __mul__(self, b):
        return b.diff(self._coord)
    

class CoordinateDerivative(object):
    
    def __init__(self, index=None):
        self._index = index
        
    def __getitem__(self, index):
        return CoordinateDerivative(index)
    
    def __call__(self, tensor):
        if isinstance(tensor, MTensor):
            array = tensor.components
            new_array = np.array([v_diff(array, x)                       \
                                  for x in self._index.chart.raw_coords], \
                                  dtype = object)
            return tensor._return_tensor([get_dummy_idx(-1, self._index.chart)]\
                    + tensor.indices, new_array)[[self._index] + tensor.indices]
        else:
            array = [sp.diff(tensor, x) for x in self._index.chart.raw_coords]
            return MTensor([get_dummy_idx(-1, self._index.chart)], array)[self._index]


def get_Cristoffel_symbol(chart, indices):
    l, mu, nu, rho = map(get_dummy_idx, [1, 1, 1, 1])
    chart.assign_indices([l, mu, nu, rho])
    
    CD = CoordinateDerivative()
    metric_deriv = CD[-mu](chart.metric[-rho, -nu])
    gamma_udd = (chart.metric[l, rho] * (metric_deriv[-mu, -rho, -nu] \
                + metric_deriv[-nu, -mu, -rho] \
                - metric_deriv[-rho, -mu, -nu]))/2
    return gamma_udd.transpose([l, -mu, -nu])[indices]



def kronecker_delta(indices):
    dim = indices[0].chart.dim
    if not all(map(lambda x : x.chart.dim == dim, indices)):
        raise IndexException("Dimensions mismatch")
    return MTensor(indices, sp.eye(dim).tolist())


def levi_civita(indices):
    v_lc = np.vectorize(sp.LeviCivita)
    def_indices = [get_dummy_idx(1, i.chart) for i in indices]
    dimensions = [x.chart.dim for x in indices]
    return MTensor(def_indices, np.fromfunction(v_lc, \
                   dimensions, dtype = object))[indices]

 
def get_Riemann( chart, indices ):
    l, mu, nu, rho, kappa = sp.symbols("l mu nu rho kappa")
    l, mu, nu, rho, kappa = map(SumIdx, [l, mu, nu, rho, kappa])
    chart.assign_indices([l, mu, nu, rho, kappa])
    
    Gamma_u = get_Cristoffel_symbol(chart, [l, -mu, -nu])
    CD = CoordinateDerivative()
    
    R1 = CD[-nu](Gamma_u[l, -mu, -kappa])
    R2 = R1[-kappa, l, -mu, -nu]
    R3 = Gamma_u[l, -nu, -rho] * Gamma_u[rho, -mu, -kappa]
    R4 = R3[l, -kappa, -mu, -nu]

    Riemann = (R1 - R2 + R3 - R4)
    
    return Riemann.transpose([l, -mu, -nu, -kappa])[indices]
    
    
def get_Ricci( chart, indices ):
    l, mu, nu = sp.symbols("l mu nu")
    l, mu, nu = map(SumIdx, [l, mu, nu])
    chart.assign_indices([l, mu, nu])
    
    return get_Riemann(chart, [l, -mu, -l, -nu])[indices]

def get_Curvature( chart ):
    l = get_dummy_idx(1, chart)
    return get_Ricci(chart, [l, -l])
    
def get_Einstein( chart, indices ):
    l = get_dummy_idx(1, chart)
    R = get_Ricci( chart, indices )
    curv = sp.simplify(R[-l, l].components.tolist())
    return (R - sp.Rational(1,2) * chart.metric[indices] * curv)
    
def get_dummy_vector( label, index, *args ):
    array = [sp.symbols(label + str(i), cls=sp.Function)(*args) \
             for i in range(index.chart.dim)]
    return MTensor([index], array)
    
### Wrappers around sympy functions

def v_diff(expr, *symbols, **kwargs):
    return np.vectorize(lambda exp : sp.diff(exp, *symbols, **kwargs))(expr)

def t_diff(tensor, *symbols, **kwargs):
    expr = symbols[0]
    if isinstance(expr, type(tensor)):
        derivative = expr.apply_function(np.vectorize(DerOp))
        derivative._indices = map(lambda x : -x, derivative.indices)
        return derivative*tensor
    else:
        v_diff = np.vectorize(lambda expr : sp.diff(expr, *symbols, **kwargs))
        return tensor.apply_function(v_diff)
 
def t_integrate(tensor, *symbols, **kwargs):
    v_integrate = np.vectorize(lambda expr : sp.integrate(expr, *symbols, **kwargs))
    return tensor.apply_function(v_integrate) 

def t_simplify(tensor, *args, **kwargs):
    v_simplify = np.vectorize(lambda expr : sp.simplify(expr, *args, **kwargs))
    return tensor.apply_function(v_simplify) 



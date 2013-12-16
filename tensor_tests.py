from Tensor import *
from special_tensors import *
import sympy as sp
import numpy as np

### Index tests
idx1 = SumIdx("i")
assert(-idx1 == SumIdx("i", -1))
idx2 = SumIdx("j", -1)
assert(-idx2 == SumIdx("j"))

### Free Tensors tests
i = SumIdx("i")
a = np.array([[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
ft = FreeTensor(a)
assert(ft  == FreeTensor(a))
assert(-ft == FreeTensor(-a))
assert(ft.contract([(0,1)]) == FreeTensor(34))
assert(ft.transpose((1,0))  == FreeTensor(a.transpose()))
assert(ft[1,i]          == FreeTensor(a[1,:]))
assert(ft[0,i]*ft[0,i]  == FreeTensor([[1,2,3,4],[2,4,6,8], \
                                       [3,6,9,12],[4,8,12,16]]))
assert(2 * ft == ft * 2 == FreeTensor(2*a))
             
### Indexed Tensors tests
i, j, k = map(SumIdx, ["i", "j", "k"])
s = np.array([[1, 2],[2, 4]])
a = np.array([[0, -8],[8, 0]])
it1 = IndexedTensor([i,-j], a + s)
it2 = IndexedTensor([-j,i], a + s)
assert((it1 + it2)/2 == IndexedTensor([i, -j], s))
assert((it1 - it2)/2 == IndexedTensor([i, -j], a))
assert(it1 - it2 == it1 + (-it2))
assert(it1[i,-j]*it1[j, -k] == IndexedTensor([i, -j], np.dot(a+s, a+s)))
assert(it1[i,-j]*it2[-i, k] == IndexedTensor([-j, k], \
                                             np.dot((a+s).transpose(), a+s)))
assert(it1[0,0] == IndexedTensor([], 1))

### MTensors tests
i, j, k = map(SumIdx, ['i', 'j', 'k'])
t, x = sp.symbols("t x")

M2 = Manifold("M2", 2)

decart = CoordinateChart("orth", M2, [t, x])
decart.assign_indices([i, j, k])

g = Metric([-1, -1], decart, np.diag([-1, 1]))

assert(g[i, j] == MTensor([i, j], np.diag([-1, 1])))

mt = MTensor([i, j], [[1, 2], [3, 4]])

assert(mt[ i,  j] == MTensor([ i,  j], [[ 1,  2], [ 3,  4]]))
assert(mt[ i, -j] == MTensor([ i, -j], [[-1,  2], [-3,  4]]))
assert(mt[-i,  j] == MTensor([-i,  j], [[-1, -2], [ 3,  4]]))
assert(mt[-i, -j] == MTensor([-i, -j], [[ 1, -2], [-3,  4]]))

### Curvature related Tensors tests

M2 = Manifold("M2", 2)
a, theta, phi = sp.symbols("a theta phi")
i, j, k, l = map(SumIdx, ["i", "j", "k", "l"])
a_gdd = [
    [a**2, 0],
    [0, a**2*sp.sin(theta)**2]
    ]
    
sphere = CoordinateChart("sph", M2, [theta, phi])
sphere.assign_indices([i,j,k,l])

g_sph = Metric([-1, -1], sphere, np.array(a_gdd))

assert(g_sph[i, j] == MTensor([i, j], [[1/a**2, 0],\
                                       [0, 1/(a*sp.sin(theta))**2]]))
                                       
assert(get_Cristoffel_symbol(sphere, [i, -j, -k]) ==               \
       MTensor([i, -j, -k], [ [[0, 0],                             \
                               [0, -sp.sin(theta)*sp.cos(theta)]], \
                              [[0, sp.cos(theta)/sp.sin(theta)], 
                               [sp.cos(theta)/sp.sin(theta), 0]] ]))

assert(get_Curvature(sphere) == MTensor([], 2/a**2))

### Special Tensors tests    


print("Passed.")


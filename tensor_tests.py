from Tensor import *
from special_tensors import *
import sympy as sp
import numpy as np

### Index tests
idx1 = SumIdx("i")
assert(-idx1 == SumIdx("i", -1))
idx2 = SumIdx("j", -1)
assert(-idx2 == SumIdx("j"))


### Algebraic properties
i, j = map(SumIdx,["i", "j"])
a = np.array([[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
ft = Tensor([i, j], a)
assert(ft                   == Tensor([j, i], a))
assert(-ft                  == Tensor([i, j], -a))
assert(ft.contract([(0,1)]) == Tensor([], 34))
assert(ft.transpose([j,i])  == Tensor([i, j], a.transpose()))
assert(ft[1,i]              == Tensor([i], a[1,:]))
assert(ft[0,i]*ft[0,j]      == Tensor([i,j], [[1,2,3,4],[2,4,6,8], \
                                              [3,6,9,12],[4,8,12,16]]))
assert(2 * ft == ft * 2 == Tensor([i,j], 2*a))
        
i, j, k = map(SumIdx, ["i", "j", "k"])
s = np.array([[1, 2],[2, 4]])
a = np.array([[0, -8],[8, 0]])
it1 = Tensor([i,-j], a + s)
it2 = Tensor([-j,i], a + s)
assert((it1 + it2)/2        == Tensor([i, -j], s))
assert((it1 - it2)/2        == Tensor([i, -j], a))
assert(it1[i,-j]*it1[j, -k] == Tensor([i, -j], np.dot(a+s, a+s)))
assert(it1[i,-j]*it2[-i, k] == Tensor([-j, k], np.dot((a+s).transpose(), a+s)))
assert(it1[0,0]             == Tensor([], 1))   

### Tensors on manifolds tests
i, j, k = map(SumIdx, ['i', 'j', 'k'])
t, x = sp.symbols("t x")

minkoswki = CoordinateChart("orth", [t, x])
minkoswki.assign_indices([i, j, k])

g = Metric([-1, -1], minkoswki, np.diag([-1, 1]))

assert(g[i, j] == Tensor([i, j], np.diag([-1, 1])))

mt = Tensor([i, j], [[1, 2], [3, 4]])

assert(mt[ i,  j] == Tensor([ i,  j], [[ 1,  2], [ 3,  4]]))
assert(mt[ i, -j] == Tensor([ i, -j], [[-1,  2], [-3,  4]]))
assert(mt[-i,  j] == Tensor([-i,  j], [[-1, -2], [ 3,  4]]))
assert(mt[-i, -j] == Tensor([-i, -j], [[ 1, -2], [-3,  4]]))


### Curvature related Tensors tests

a, theta, phi = sp.symbols("a theta phi")
i, j, k, l = map(SumIdx, ["i", "j", "k", "l"])
    
sphere = CoordinateChart("sph", [theta, phi])
sphere.assign_indices([i,j,k,l])

a_gdd = [ [a**2, 0], [0, a**2*sp.sin(theta)**2] ]
g_sph = Metric([-1, -1], sphere, np.array(a_gdd))

assert(g_sph[i, j] == Tensor([i, j], [[1/a**2, 0],\
                                      [0, 1/(a*sp.sin(theta))**2]]))

assert(get_Cristoffel_symbol(sphere, [i, -j, -k]) ==               \
        Tensor([i, -j, -k], [ [[0, 0],                             \
                               [0, -sp.sin(theta)*sp.cos(theta)]], \
                              [[0, sp.cos(theta)/sp.sin(theta)], 
                               [sp.cos(theta)/sp.sin(theta), 0]] ]))

assert(get_Curvature(sphere) == Tensor([], 2/a**2))

### Special Tensors tests    


print("Passed.")  



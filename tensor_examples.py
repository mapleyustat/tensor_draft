from Tensor import *
from special_tensors import *
from sympy import *
import numpy as np


### Schwarzschild solution
print("----> I). Finding Schwarzschild solution:")
nu, lam          = symbols('nu lambda', cls=Function)
i, j, k          = map(SumIdx, ["i", "j", "k"])
t, r, theta, phi = symbols("t r theta phi")

a_gdd = np.diag([-exp(nu(r)), exp(-nu(r)), r**2, 
                 r**2*sin(theta)**2])
    
spherical = CoordinateChart("sph", [t, r, theta, phi])
spherical.assign_indices([i, j, k])

g         = Metric([-1, -1], spherical, a_gdd)

e = get_Einstein(spherical, [i, j]).as_array.tolist()
print("Einstein tensor's diagonal components: ")
for l in range(0, 4):
    pprint(Eq(Symbol('G^%i%i' % (l, l)), e[l][l]))
print("Solution:")
pprint(dsolve(e[2][2]))

### Curvature of the conformally flat 2D metric

print("----> II). Curvature of the conformally flat 2D manifold:")
omega            = symbols('omega', cls=Function)
i, j, k          = map(SumIdx, ["i", "j", "k"])
x, y             = symbols("x y")
  
cf_flat   = CoordinateChart("sph", [x, y])
cf_flat.assign_indices([i, j, k])

g         = Metric([-1, -1], cf_flat, np.diag([exp(-omega(x, y)), \
                                                exp(-omega(x, y))]))
R = get_Curvature( cf_flat ).as_array.tolist()
print("Curvature is: ")
pprint(simplify(R))


### Kink energy
print("----> III). Calculating Kink energy in 1+1 dimensional theory\n")

t, x, x0  = symbols("t x x0")
lam, v, u = symbols("lambda v u")#, positive = True)
phi       = Function("phi")(t, x)

dec = CoordinateChart("dec", [t, x])
dec.assign_indices([i, j, k])

g   = Metric([-1, -1], dec, np.diag([1, -1]))
CD  = CoordinateDerivative(i)

L = Rational(1,2)*(CD[-i](phi) * CD[i](phi)) \
        - Rational(1,2) * lam*(phi**2 - v**2)**2
print("Lagrangian density:")
pprint(Eq(Symbol("L"), L.as_array.tolist()))

T = t_diff(L, CD[-i](phi))*CD[-j](phi) - kronecker_delta([i, -j]) * L

#because of integration failure we can only compute static kink energy
u = 0
x0 = 0

kink_solution = v * tanh(v * sqrt(lam) * (x - u*t - x0)/sqrt(1 - u**2))
print("\nStatic Kink solution:")
pprint(Eq(phi, kink_solution))

T00 = simplify(T[i,j].as_array[0,0].subs(phi, kink_solution).doit())
print("\nEnergy density:")
pprint(Eq(Symbol('T^00'), T00))
print("\nEnergy:")
pprint(Eq(Symbol('E'), simplify(sp.integrate(T00, (x, -oo, oo)))))




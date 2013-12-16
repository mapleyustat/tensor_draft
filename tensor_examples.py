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
    
M4        = Manifold("M4", 4)
spherical = CoordinateChart("sph", M4, [t, r, theta, phi])
spherical.assign_indices([i, j, k])

g         = Metric([-1, -1], spherical, a_gdd)

e = get_Einstein(spherical, [i, j]).components.tolist()
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
  
C2        = Manifold("C2", 4)
cf_flat   = CoordinateChart("sph", C2, [x, y])
cf_flat.assign_indices([i, j, k])

g         = Metric([-1, -1], cf_flat, np.diag([exp(-omega(x, y)), \
                                                exp(-omega(x, y))]))
R = get_Curvature( cf_flat ).components.tolist()
print("Curvature is: ")
pprint(simplify(R))


### Kink energy
print("----> III). Calculating Kink energy in 1+1 dimensional theory\n")

t, x, x0  = symbols("t x x0")
lam, v, u = symbols("lambda v u")#, positive = True)
phi       = Function("phi")(t, x)

M2  = Manifold("M2", 2)
dec = CoordinateChart("dec", M2, [t, x])
dec.assign_indices([i, j, k])

g   = Metric([-1, -1], dec, np.diag([1, -1]))
CD  = CoordinateDerivative(dec)

L = Rational(1,2)*(CD[-i](phi)*CD[i](phi)) - lam/2*(phi**2 - v**2)**2
print("Lagrangian density:")
pprint(Eq(Symbol("L"), L.components.tolist()))

T = t_diff(L, CD[-i](phi))*CD[-j](phi) - kronecker_delta([i, -j]) * L

#because of integration failure we can only compute static kink energy
u = 0
x0 = 0

kink_solution = v * tanh(v * sqrt(lam) * (x - u*t - x0)/sqrt(1 - u**2))
print("\nStatic Kink solution:")
pprint(Eq(phi, kink_solution))

T00 = simplify(T[i,j].components[0,0].subs(phi, kink_solution).doit())
print("\nEnergy density:")
pprint(Eq(Symbol('T^00'), T00))
print("\nEnergy:")
pprint(Eq(Symbol('E'), simplify(sp.integrate(T00, (x, -oo, oo)))))

""" In short - too slow
### 't Hooft monopole
print("----> III). monopole\n")

H, K             = symbols('H K', cls=Function)
t, x, y, z       = symbols("t x y z", positive=True)
lam, v, q        = symbols("lambda v q")
i, j, k          = map(SumIdx, ["i", "j", "k"])
a, b, c, d, e, f = map(SumIdx, ["a", "b", "c", "d", "e", "f"])

Y                = symbols('Y')
r = sqrt(x**2 + y**2 + z**2)

M3  = Manifold("M3", 3)
dec = CoordinateChart("dec", M3, [x, y, z])
dec.assign_indices([i, j, k])

SU2 = Manifold("SU2", 3)
paul = CoordinateChart("pauli", SU2, [])
paul.assign_indices([a,b,c,d,e,f])

g   = Metric([-1, -1], dec, np.diag([1, 1, 1]))
sug = Metric([-1, -1], paul, np.diag([1, 1, 1]))
CD  = CoordinateDerivative()

phi = 1/(q*r**2)*H(v*q*r)*dec.coords(a)
A = (1/(q*r**2)*levi_civita([i, a, b])*dec.coords(-b)*(1-K(v*q*r))).transpose([a, i])

Dphi = (CD[-i](phi[a]) - q * levi_civita([a,b,c]) * A[-b, -i] * phi[-c]).transpose([a, -i])
DDphi = CD[i](Dphi) - q * levi_civita([a,b,c]) * A[-b, i]* Dphi[-c, -i]
dV = 2 * lam * phi[a] * (phi[b]*phi[-b] - v ** 2)
eq = DDphi + dV
#pprint((simplify(eq[0].components.tolist().subs(r, Y/(q*v)))) ) 

F = (CD[i](A[a, j]) - CD[j](A[a, i]) - q * levi_civita([a, b, c]) * A[-b, i] * A[-c, j]).transpose([a, i, j])

DF = CD[-i](F[a, i, j]) - q * levi_civita([a, b, c]) * A[-b, -i] * F[-c, i, j]
#pprint(simplify(DF[1, 2].components.tolist().subs([(y, 0), (z, 0)])).subs(r, Y/(q*v)))
"""


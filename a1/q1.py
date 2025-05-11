import numpy as np
from profile1D import Planet, G

R = 3389.5E3    # [m]
dr = 10         # [m]
rho = 3934.0

layers = [[0.0, R]]
rho_vals = [rho]

M = (4/3) * np.pi * R**3 * rho
g = (G * M) / R**2
p = (3 * G * M**2)/(8 * np.pi * R**4)
MoI = 0.4

Mars = Planet("Mars", layers, rho_vals, [0.0], [0.0], [0.0], [0.0], dr, 0, "Euler")

Mars.report_results()
Mars.compute_rel_errs([M, g, p, MoI], True)

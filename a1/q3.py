import numpy as np
from profile1D import Planet, G

layers = [[0, 1834.64E3], [1834.64E3, 1899.12E3], [1899.12E3, 2841.56E3], [2841.56E3, 2980.45E3], [2980.45E3, 3317.74E3], [3317.74E3, 3389.5E3]]
rho_vals0 = [6263.583, 4018.795, 3725.2, 3495.027, 3466.179, 3064.647]
rho_vals = [6100.0, 3975.0, 3750.0, 3475.0, 3300.0, 2925.0]

Mars0 = Planet("Mars Initial", layers, rho_vals0, [0.0], [0.0], [0.0], [0.0], 10.0, 5454395, "Euler")
Mars = Planet("Mars Modified", layers, rho_vals, [0.0], [0.0], [0.0], [0.0], 10.0, 5454395, "Euler")

Mars0.report_results()
index0 = np.argmin(np.abs(Mars0.r - layers[0][-1]))
Pcmb0 = Mars0.p[index0]
print(f"Pcmb: {Pcmb0/1E9} [GPa]")

Mars.report_results()
index = np.argmin(np.abs(Mars.r - layers[0][-1]))
Pcmb = Mars.p[index]
print(f"Pcmb: {Pcmb/1E9} [GPa]")

# M = 6.417E23 [kg]
# g = 3.73 [m/s^2]
# MoI = 0.3634 - 0.3658 [-]

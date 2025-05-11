import numpy as np
from profile1D import Planet
import matplotlib.pyplot as plt

dr = 10.0 # [m]
data = np.loadtxt("Tprofile.dat")
data_burnman = np.loadtxt("burnman_results.txt")

m2_mean = "Model 2 - Mean"
m2_min = "Model 2 - Min."
m2_max = "Model 2 - Max."
b_str = "Model 3"

layers = [[0, 1834.64E3], [1834.64E3, 1899.12E3], [1899.12E3, 2841.56E3], [2841.56E3, 2980.45E3], [2980.45E3, 3317.74E3], [3317.74E3, 3389.5E3]]
rho_vals = [6100.0, 3975.0, 3750.0, 3475.0, 3300.0, 2925.0]

alpha_arr = len(layers)*[3E-5]
Cp_arr = [831, 1080, 1080, 1080, 741, 741]
eta_arr = len(layers)*[10**(20.5)]
k_arr = [40, 40, 4, 20, 9, 2.5]
qs = 22.0E-3
gamma = 4.152E-12

Mars = Planet("Mars", layers, rho_vals, alpha_arr, Cp_arr, eta_arr, k_arr, dr, 5454395, "Euler")

print(f"Rayleigh number of the core: {round(Mars.Ra(0, rho_vals[0], 2268.791-2081.381),1)} [-]")
print(f"Rayleigh number of the lowermost TBL: {round(Mars.Ra(1, rho_vals[1], 2081.381-2028.604),1)} [-]")
print(f"Rayleigh number of the convecting mantle: {round(Mars.Ra(2, rho_vals[2], 2028.604-1857.904), 1)} [-]")
print(f"Rayleigh number of the uppermost TBL: {round(Mars.Ra(3, rho_vals[3], 1857.904-1616.140),1)} [-]")
print(f"Rayleigh number of the lithosphere: {round(Mars.Ra(4, rho_vals[4], 1616.140-706.386),1)} [-]")
print(f"Rayleigh number of the crust: {round(Mars.Ra(5, rho_vals[5], 706.386-220),1)} [-]\n")

Tcore_model = float(round(data[0,0]))
print(f"The core temperature from literature (Samuel et. al): {Tcore_model} [K]")
T_min = Mars.compute_T(2100.0, qs, gamma)
T_max = Mars.compute_T(2400.0, qs, gamma)
T_profile = Mars.compute_T(Tcore_model, qs, gamma)

print(f"Surface temperature (model/mean): {round(T_profile[-1],3)} [K]")
print(f"Surface temperature (min.): {round(T_min[-1],3)} [K]")
print(f"Surface temperature (max.): {round(T_max[-1],3)} [K]\n")

plt.figure()
plt.gca().invert_yaxis()
plt.ylabel("Depth [km]")
plt.xlabel("T [K]")
plt.grid()
plt.plot(data[:,0], Mars.radius/1E3 - data[:,1], label="Literature")
plt.plot(T_profile, (Mars.radius-Mars.r)/1E3, label=m2_mean)
plt.plot(T_min, (Mars.radius-Mars.r)/1E3, label=m2_min, linestyle="--")
plt.plot(T_max, (Mars.radius-Mars.r)/1E3, label=m2_max, linestyle="--")
plt.plot(data_burnman[-1,:], (Mars.radius-data_burnman[0,:])/1E3, label=b_str, linestyle="--")
plt.legend(loc="lower left")
plt.savefig("Mars_T_profile.png", dpi=300)
plt.show()

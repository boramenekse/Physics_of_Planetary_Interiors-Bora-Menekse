import numpy as np
from profile1D import Planet
import matplotlib.pyplot as plt

dr = 10.0 # [m]
data_rho = np.loadtxt("rho_profile.dat")

layers = [[0, 1834.64E3], [1834.64E3, 1899.12E3], [1899.12E3, 2841.56E3], [2841.56E3, 2980.45E3], [2980.45E3, 3317.74E3], [3317.74E3, 3389.5E3]]
rho_vals = [6100.0, 3975.0, 3750.0, 3475.0, 3300.0, 2925.0]

alpha_arr = len(layers)*[3E-5]
Cp_arr = [831, 1080, 1080, 1080, 741, 741]
eta_arr = len(layers)*[10**(20.5)]
k_arr = [40, 40, 4, 20, 9, 2.5]
qs = 22.0E-3
gamma = 4.152E-12
tol = 1.0

fig, (ax1, ax0) = plt.subplots(1, 2)
for ax in fig.axes:
    ax.grid(linestyle="dotted", alpha=0.5)
    ax.set_xlabel("Radius [km]")

ax1.set_ylabel(r"Gravity [$m/s^2$]")
ax0.set_ylabel("Pressure [GPa]")

plt.figure()
plt.gca().invert_yaxis()
plt.ylabel("Depth [km]")
plt.xlabel(r"Density [$kg/m^3$]")
plt.grid()

Mars = Planet("Mars", layers, rho_vals, alpha_arr, Cp_arr, eta_arr, k_arr, dr, 5454395, "Euler")
Mars1 = Planet("Mars min.", layers, rho_vals, alpha_arr, Cp_arr, eta_arr, k_arr, dr, 5454395, "Euler")
Mars2 = Planet("Mars max.", layers, rho_vals, alpha_arr, Cp_arr, eta_arr, k_arr, dr, 5454395, "Euler")
r_km = (Mars.radius-Mars.r)/1E3
rho_initial = np.array([Mars.rho(r_val) for r_val in Mars.r])
plt.plot(rho_initial, r_km, label="Initial")
ax1.plot(Mars.r/1E3, Mars.g, label="Initial", linestyle="--")
ax0.plot(Mars.r/1E3, Mars.p/1E9, label="Initial", linestyle="--")

Mars.report_results()

Mars.compute_T(2269.0, qs, gamma)
Mars.update_density(tol)
Mars.compute_MoI()
Mars.report_results()
plt.plot(np.array([Mars.rho(r_val) for r_val in Mars.r]), r_km, label="Converged model")
ax1.plot(Mars.r/1E3, Mars.g, label="Converged model")
ax0.plot(Mars.r/1E3, Mars.p/1E9, label="Converged model")

Mars1.compute_T(2173.0, qs, gamma)
Mars1.update_density(tol)
Mars1.compute_MoI()
Mars1.report_results()
plt.plot(np.array([Mars1.rho(r_val) for r_val in Mars1.r]), r_km, label="Converged min.", linestyle="--")
ax1.plot(Mars.r/1E3, Mars1.g, label="Converged min.")
ax0.plot(Mars.r/1E3, Mars1.p/1E9, label="Converged min.")

Mars2.compute_T(2373.0, qs, gamma)
Mars2.update_density(tol)
Mars2.compute_MoI()
Mars2.report_results()
plt.plot(np.array([Mars2.rho(r_val) for r_val in Mars2.r]), r_km, label="Converged max.", linestyle="--")
ax1.plot(Mars.r/1E3, Mars2.g, label="Converged max.")
ax0.plot(Mars.r/1E3, Mars2.p/1E9, label="Converged max.")

plt.plot(data_rho[:,0], (Mars.radius/1E3)-data_rho[:,1], linestyle="--", label="Literature")

fig.tight_layout()
ax0.legend()
ax1.legend()
plt.legend()
plt.savefig("Mars_rho_profile_iterated.png", dpi=300)
fig.savefig("Mars_gp_profiles_iterated.png", dpi=300)
# plt.show()

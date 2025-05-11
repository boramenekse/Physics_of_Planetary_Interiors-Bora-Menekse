from burnman import Layer, Planet
from burnman import minerals
import burnman
import numpy as np
import matplotlib.pyplot as plt

R = 3389.5E3
Rcore = 1834.64E3
Tsurface = 220.0
z_crust = R - 2980.45E3
T_top = 1616.140

Pcmb = 19.4E9

Ncore = 41
Nmantle = 41
Ncrust = 21


Core = Layer('Core', radii=np.linspace(0., Rcore, Ncore))

FeS = burnman.Composite(
    [minerals.HP_2011_ds62.iron(), minerals.HP_2011_ds62.S()], 
    fractions = [0.2417, 0.7583], 
    fraction_type = "molar",
    name= "Troilite"
)

core_material = burnman.Composite(
    [minerals.SE_2015.fcc_iron(), minerals.HP_2011_ds62.Ni(), FeS],
    fractions = [0.5335, 0.076, 0.3905],
    fraction_type = "molar",
    name = "core material"
)

Core.set_material(core_material)
Core.set_temperature_mode('adiabatic')
Core.set_pressure_mode(pressure_mode='self-consistent', pressure_top=Pcmb, gravity_bottom=0)


Mantle = Layer('Mantle', radii=np.linspace(Rcore, R-z_crust, Nmantle))

mantle_material = burnman.Composite(
    [minerals.SLB_2011.periclase(), minerals.SLB_2022.olivine([1.0, 0.0]), minerals.SLB_2022.alpv()],
    fractions = [0.195, 0.58, 0.225],
    fraction_type = "molar",
    name = "mantle material"
)

Mantle.set_material(mantle_material)
Mantle.set_temperature_mode('adiabatic', temperature_top=T_top)


Crust = Layer('crust', radii=np.linspace(R-z_crust, R, Ncrust))

crust_material = burnman.Composite(
    [minerals.SLB_2011.clinopyroxene([0.2]*5), minerals.SLB_2011.orthopyroxene([0.25]*4), minerals.SLB_2011.plagioclase([0.5]*2)],
    fractions = [0.285, 0.655, 0.06],
    fraction_type = "molar",
    name = "crust material"
)

Crust.set_material(crust_material)
Crust.set_temperature_mode('user-defined', np.linspace(Tsurface, T_top, Ncrust)[::-1])

Mars = Planet('Mars', [Core, Mantle, Crust], verbose=True)
Mars.make()

print(f'Mass: {Mars.mass:.5e} [kg]')
print(f'Gravity: {Mars.gravity[-1]:.3f} [m/s^2]')
print(f'MoI : {Mars.moment_of_inertia_factor:.5f} [-]')

test = np.vstack((Mars.radii, Mars.density, Mars.gravity, Mars.pressure, Mars.temperature))
np.savetxt("burnman_results.txt", test)

fig = plt.figure(figsize=(14, 7))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
ax[0].plot(Mars.pressure/1.e9, 3389.5-Mars.radii/1.e3)
ax[1].plot(Mars.temperature, 3389.5-Mars.radii/1.e3)
ax[2].plot(Mars.gravity, 3389.5-Mars.radii/1.e3)
ax[3].plot(Mars.density, Mars.radii/1.e3)
for i in range(3):
    ax[i].set_ylim(3389.5-Mars.radii[0]/1.e3,
                   3389.5-Mars.radii[-1]/1.e3)
    ax[i].set_ylabel('Depth (km)')

ax[0].set_xlabel('Pressure (GPa)')
ax[1].set_xlabel('Temperature (K)')
ax[2].set_xlabel('Gravity (m/s$^2$)')
ax[3].set_xlabel(r'Density [$kg/m^3$]')

fig.set_layout_engine('tight')
plt.show()

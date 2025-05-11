import numpy as np
from profile1D import Planet, G

R = 3389.5E3    # [m]
dr = 10         # [m]

def mg_theoretical(layers:list[list], rho_vals:list[float]) -> tuple[float]:
    n_layers = len(layers)
    m = 0

    for i in range(n_layers):
        m += ((4*np.pi)/3)*rho_vals[i]*(layers[i][-1]**3 - layers[i][0]**3)

    g = (G*m)/layers[-1][-1]**2

    return m, g

def run_model(name:str, layers:list[list], rho_vals:list[float], integrator:str, downscale:float, P_theo:float, MoI_theo:float, selected_index:int):
    planet = Planet(name, layers, rho_vals, [0.0], [0.0], [0.0], [0.0], dr, 5454395, integrator, downscale)

    planet.report_results()
    mg_planet = mg_theoretical(layers, rho_vals)

    _ = planet.compute_rel_errs([mg_planet[0], mg_planet[1], P_theo, MoI_theo], True)
    # _ = planet.compute_rel_errs([6.417E23, 3.73, P_theo, 0.3635], True)

    if name == "Mars_3" or name == "Mars_4" or name == "Mars_5":
        # Only the pressure at CMB is known for these models, correct the rel. error
        index = np.argmin(np.abs(planet.r - layers[0][-1]))
        planet_numerical = planet.p[index]
        print(f"Using the pressure at r = {planet.r[index]/1E3} [km]: {planet_numerical/1E9} [GPa]")
        print(f"{name} eps_p: {100*abs(planet_numerical - P_theo)/P_theo} [%]")

    if name == f"Mars_{selected_index}":
        print(f"Saving the profiles for the model {name}...")
        fig = planet.plot_profiles()
        fig.savefig(f"Mars_gp_profiles.png", dpi=300)

    return None

# Model 1
layers1 = [[0, R-1750E3], [R-1750E3, R-1112.5E3], [R-1112.5E3, R-50E3], [R-50E3, R]]
rho_vals1 = [6900, 3925, 3500, 2900]

# Model 2
layers2 = [[0, R-2000E3], [R-2000E3, R-1112.5E3], [R-1112.5E3, R-50E3], [R-50E3, R]]
rho_vals2 = [8350, 3975, 3500, 2900]

# Model 3
layers3 = [[0, 1788E3], [1788E3, R-50E3], [R-50E3, R]]
rho_vals3 = [6162.5, 3250, 3000]

# Model 4
layers4 = [[0, 1834.64E3], [1834.64E3, 1899.12E3], [1899.12E3, 2841.56E3], [2841.56E3, 2980.45E3], [2980.45E3, 3317.74E3], [3317.74E3, 3389.5E3]]
rho_vals4 = [6263.583, 4018.795, 3725.2, 3495.027, 3466.179, 3064.647]

# Model 4 - Modified (Chosen profile)
rho_vals4_m = [6100.0, 3975.0, 3750.0, 3475.0, 3300.0, 2925.0]

layers_all_models = [layers1, layers2, layers3, layers4, layers4]
rho_vals_all_models = [rho_vals1, rho_vals2, rho_vals3, rho_vals4, rho_vals4_m]
P_theo_all_models = [39E9, 43E9, 19.4E9, 19.0E9, 19.0E9]
MoI_theo_all_models = [0.3635, 0.3635, 0.3646, 0.36379, 0.36379]

for i in range(len(layers_all_models)):
    layers_model = layers_all_models[i]
    rho_vals_model = rho_vals_all_models[i]
    P_theo_model = P_theo_all_models[i]
    MoI_theo_model = MoI_theo_all_models[i]

    run_model(f"Mars_{i+1}", layers_model, rho_vals_model, "Euler", 1.0, P_theo_model, MoI_theo_model, 5)

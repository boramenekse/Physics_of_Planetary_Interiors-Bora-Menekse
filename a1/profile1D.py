import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from numerical_methods import Integrator
from scipy.interpolate import interp1d

G = 6.6743E-11  # [m^3/kg*s^2]

class DensityProfile:

    def __init__(self, layers:list[list], rho_vals:list):
        self.layers = layers
        self.check_layers()

        self.rho_vals = rho_vals
        self.R = layers[-1][-1]

    def check_layers(self):
        layers_flattened = [bound for layer in self.layers for bound in layer]
        
        if np.any(np.array(layers_flattened) < 0):
            raise ValueError("Radius cannot be negative")

        if float(layers_flattened[0]) != 0.0:
            raise ValueError("Layers should start from a radius value value of 0.0")

        if sorted(layers_flattened) != layers_flattened:
            raise ValueError("Bounds of the layers are unrealistic")

        layers_arr = np.array(self.layers)
        if not np.all(layers_arr[:-1, 1] == layers_arr[1:, 0]):
            raise ValueError("Bounds of the layers are not continuous")

        return None

    def __call__(self, r:float) -> float:
        if r < 0 or r > self.R:
            raise ValueError(f"Input radius value ({r}) is outside the bounds of the radial profile [0, {self.R}] (m)")
        
        for (lower_bound, upper_bound), rho in zip(self.layers, self.rho_vals):
            if lower_bound <= r <= upper_bound:
                if not callable(rho):
                    return rho
                
                return rho(r)
        

@njit
def g_interp(r:float, r_arr:np.ndarray, g_arr:np.ndarray) -> float:    
    if r <= r_arr[0]:
        return g_arr[0]
    elif r >= r_arr[-1]:
        return g_arr[-1]
    
    lower = 0
    upper = r_arr.shape[0] - 1
    while lower <= upper:
        mid = (lower + upper) // 2
        if r_arr[mid] < r:
            lower = mid + 1
        else:
            upper = mid - 1

    # Variable lower ends being the upper bound
    r0 = r_arr[upper]
    g0 = g_arr[upper]
    
    return g0 + (g_arr[lower] - g0) * (r - r0) / (r_arr[lower] - r0)


class Planet:

    def __init__(self, name:str, layers:list[list[float]], rho_vals:list[float], alpha_arr:list[float], Cp_arr:list[float], eta_arr:list[float], k_arr:list[float], dr:float, seedval:int, integrator:str="RK4", downscale_factor:float=1.0):
        self.name = name
        self.layers = layers
        self.rho = DensityProfile(layers, rho_vals)
        self.radius = layers[-1][-1]
        self.dr = dr

        self.layer_indices = np.concatenate(([0], (np.array(layers)[:, 1]/dr).astype(int)))
   
        depth_flat = (self.radius-np.concatenate(layers))[::-1]
        self.depth = [[depth_flat[2*i], depth_flat[2*i+1]] for i in range(len(layers))]
        self.T = None

        self.alpha_arr = alpha_arr
        self.Cp_arr = Cp_arr
        self.eta_arr = eta_arr
        self.k_arr = k_arr

        self.alpha = DensityProfile(layers, alpha_arr)
        self.Cp = DensityProfile(layers, Cp_arr)
        self.eta = DensityProfile(layers, eta_arr)
        self.k = DensityProfile(layers, k_arr)
        self.q = None

        self.downscale_factor = downscale_factor
        self.integrator = integrator

        self.rng = np.random.default_rng(seedval)
        
        self.__integM = Integrator(self.__fM, self.integrator)
        self.__integMoI = Integrator(self.__fMoI, self.integrator)
        self.__integP = Integrator(self.__fP, self.integrator)
        self.__integq = Integrator(self.__fq, self.integrator)
        self.__integT_conductive = Integrator(self.__fT_conductive, self.integrator)
        self.__integT_convective = Integrator(self.__fT_convective, self.integrator)

        self.r, self.M, self.g, self.p, self.mass, self.gravity, self.pressure, self.MoI = np.zeros(8)

        self.compute_profiles()
        self.compute_MoI()

    def compute_profiles(self) -> None:
        self.M, self.r = self.__integrate_mass()
        self.M = self.M[:, 0]
        self.mass = self.M[-1]

        self.g = np.zeros_like(self.M)
        self.g[1:] = G * self.M[1:] / np.square(self.r[1:])
        self.gravity = self.g[-1]

        self.p = self.__integrate_pressure()
        self.pressure = self.p[0]
        
        return None
    
    def compute_MoI(self) -> None:
        MoI_arr, _ = self.__integMoI.integrate(np.array([0.0]), 0.0, self.radius, self.downscale_factor*self.dr, {"backwards": False})
        self.MoI = ((8*np.pi)/3) * MoI_arr[-1, 0] / (self.mass * self.radius**2)

        return None
    
    def recalculate_properties(self, verbose:bool=False) -> None:
        self.compute_profiles()
        self.compute_MoI()

        if verbose:
            self.report_results()

        return None
    
    def return_results(self, verbose:bool=False) -> tuple:
        if verbose:
            self.report_results()

        return self.r, self.M, self.g, self.p, self.mass, self.gravity, self.pressure, self.MoI
    
    def change_integrator(self, integrator_new:str) -> None:
        if self.integrator == integrator_new:
            return None
        
        self.integrator = integrator_new

        self.__integM = Integrator(self.__fM, self.integrator)
        self.__integMoI = Integrator(self.__fMoI, self.integrator)
        self.__integP = Integrator(self.__fP, self.integrator)
        self.__integq = Integrator(self.__fq, self.integrator)
        self.__integT_conductive = Integrator(self.__fT_conductive, self.integrator)
        self.__integT_convective = Integrator(self.__fT_convective, self.integrator)

        return None

    def __fM(self, r:float, u:np.ndarray, p:dict) -> float:
        return 4 * np.pi * self.rho(r) * r**2
    
    def __fP(self, r:float, u:np.ndarray, p:dict) -> float:
        return -self.rho(r) * g_interp(r, self.r, self.g)
        
    def __fMoI(self, r:float, u:np.ndarray, p:dict) -> float:
        return self.rho(r) * r**4
    
    def __fq(self, z:float, u:np.ndarray, p:dict) -> float:
        q = u[0]
        r = self.radius - z
        eps_val = p["eps"]
        if callable(eps_val):
            eps_val = p["eps"](r)

        return 2*(q/r) - self.rho(r)*eps_val

    def __fT_convective(self, r:float, u:np.ndarray, p:dict) -> float:
        T = u[0]
        p["index"] += 1

        return - ((self.alpha(r)*self.g[p["index"]-1])/self.Cp(r)) * T

    def __fT_conductive(self, r:float, u:np.ndarray, p:dict) -> float:
        p["index"] -= 1

        return - self.q[p["index"]+1] / self.k(r)

    def __integrate_T_convective(self, T0:float, r0:float, rN:float, index:int):
        return self.__integT_convective.integrate(np.array([T0]), r0, rN, self.dr, {"index": index, "backwards": False})

    def __integrate_T_conductive(self, T0:float, r0:float, rN:float, index:int):
        return self.__integT_conductive.integrate(np.array([T0]), r0, rN, self.dr, {"index": index, "backwards": False})

    def __integrate_q(self, q0:float, z0:float, zN:float, eps:float):
        return self.__integq.integrate(np.array([q0]), z0, zN, self.dr, {"eps": eps, "backwards": False})

    def __integrate_mass(self):
        return self.__integM.integrate(np.array([0.0]), 0.0, self.radius, self.dr, {"backwards": False})

    def __integrate_pressure(self):
        p_arr, _ = self.__integP.integrate(np.array([0.0]), self.radius, 0.0, -self.dr, {"backwards": True})

        return p_arr[::-1, 0]    
    
    def __rel_err(self, num:float, theo:float) -> float:
        return 100 * abs((num - theo) / theo)

    def __return_max_len(self, strings:list[str]) -> int:
        lens = [len(str_val) for str_val in strings]

        return max(lens)

    def compute_rel_errs(self, theoretical_vals:list[float], verbose:bool=False) -> list[float]|None:
        if len(theoretical_vals) != 4:
            raise ValueError(f"Please provide a value for all of M, g, p, and MoI!")
        
        tM, tg, tp, tMoI = theoretical_vals

        relM = self.__rel_err(self.mass, tM)
        relg = self.__rel_err(self.gravity, tg)
        relp = self.__rel_err(self.pressure, tp)
        relMoI = self.__rel_err(self.MoI, tMoI)
        
        rel_list = [relM, relg, relp, relMoI]

        if verbose:
            self.report_results(results=rel_list, symbols_units=[["eps_M", "eps_g", "eps_p", "eps_MoI"], 4*["%"]], title_str="RELATIVE ERRORS")
        
            return None

        return rel_list
    
    def report_results(self, results:list[float]=None, symbols_units:list=[["M","g","p","MoI"],["kg","m/s^2","Pa","-"]], title_str:str="PROPERTIES", space_symbol:int=2, space_unit:int=1, top_bot_chr:chr="-", sides_chr:chr="|") -> None:
        if len(symbols_units[0]) != len(symbols_units[1]):
            raise ValueError(f"Each symbol (string) should have a unit (string)!")
        
        blank = space_symbol * " "

        sM, sg, sp, sMoI = symbols_units[0]
        uM, ug, up, uMoI = symbols_units[1]

        rM, rg, rp, rMoI = self.mass, self.gravity, self.pressure, self.MoI
        if results != None and len(results) == 4:
            rM, rg, rp, rMoI = results

        max_len_symbol = self.__return_max_len([sM, sg, sp, sMoI])

        str_M = f"{sM}{(max_len_symbol - len(sM))*' '}{blank}: {rM}"
        str_g = f"{sg}{(max_len_symbol - len(sg))*' '}{blank}: {rg}"
        str_p = f"{sp}{(max_len_symbol - len(sp))*' '}{blank}: {rp}"
        str_MoI = f"{sMoI}{(max_len_symbol - len(sMoI))*' '}{blank}: {rMoI}"

        max_len0 = self.__return_max_len([str_M, str_g, str_p, str_MoI])

        str_M += (max_len0 - len(str_M) + space_unit) * " " + f"[{uM}]\n"
        str_g += (max_len0 - len(str_g) + space_unit) * " " + f"[{ug}]\n"
        str_p += (max_len0 - len(str_p) + space_unit) * " " + f"[{up}]\n"
        str_MoI += (max_len0 - len(str_MoI) + space_unit) * " " + f"[{uMoI}]\n"

        max_len1 = self.__return_max_len([str_M, str_g, str_p, str_MoI])
        if max_len1 % 2 == 1:
            max_len1 += 1

        top_bot_str = f"{max_len1 * top_bot_chr}\n"
        header_str = f"{title_str} - {self.name}"
        blank_sides = (max_len1//2 - 1 - len(header_str)//2) * " "

        title = f"{sides_chr}{blank_sides}{header_str}{blank_sides}{sides_chr}\n"
        
        if len(header_str) % 2 == 1:
            title = title.replace(" ", '', 1)
            
        header = top_bot_str + title + top_bot_str

        print(header + str_M + str_g + str_p + str_MoI)
    
    def plot_profiles(self):
        fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True)
        
        for ax in fig.axes:
            ax.grid(linestyle="dotted", alpha=0.5)
            ax.set_xlabel("Radius [km]")

        r_km = self.r/1E3

        ax0.plot(r_km, self.g)
        ax0.set_ylabel(r"Gravity [$m/s^2$]")

        ax1.plot(r_km, self.p/1E9)
        ax1.set_ylabel("Pressure [GPa]")

        fig.tight_layout()
        return fig
    
    def dT_dz_Fourier(self, q:float, k:float) -> float:
        return -q / k

    def sample_params(self, N:int, mean_params:list[float], std_params:list[float], range_Ts:list[float]) -> tuple[np.ndarray]:
        mean_Tcore, mean_Tcmb, mean_t_lit, mean_qs = mean_params
        std_Tcore, std_Tcmb, std_t_lit, std_qs = std_params

        t_lit = self.rng.normal(mean_t_lit, std_t_lit, N)
        beta = t_lit / self.radius

        Tcore = self.rng.normal(mean_Tcore, std_Tcore, N)
        Tcmb = self.rng.normal(mean_Tcmb, std_Tcmb, N)
        qs = self.rng.normal(mean_qs, std_qs, N)
        Ts = loguniform(range_Ts[0], range_Ts[1]).rvs(size=N, random_state=self.rng)

        dT_dz_crust = np.array([self.dT_dz_Fourier(qs_val, self.k_arr[-1]) for qs_val in qs])
        T3 = Ts - dT_dz_crust * (self.radius-self.rho.layers[-1][0])

        T2 = np.array([self.rng.normal(0.5*(Tcmb_val+T3_val), abs(Tcmb_val-T3_val)/8) for Tcmb_val, T3_val in zip(Tcmb, T3)])

        return Ts, T3, T2, Tcmb, Tcore, beta

    def Ra(self, layer_index:int, rho: float, dT:float) -> float:
        D = self.layers[layer_index][-1] - self.layers[layer_index][0]
        Kappa = self.k_arr[layer_index] / (rho*self.Cp_arr[layer_index])

        return (rho * self.alpha_arr[layer_index] * self.gravity * dT * D**3) / (Kappa * self.eta_arr[layer_index])

    def linear_line(self, z0:float, T0:float, zN:float, TN:float) -> tuple[np.ndarray]:
        range_layer = zN - z0
        if range_layer % self.dr != 0:
            raise ValueError(f"dz should be a multiple of the layer range (zN-z0)\ndz: {self.dr}\nLayer range: {range_layer}")

        dT_dz = (TN - T0) / range_layer
        z_arr = np.linspace(0.0, range_layer, 1+int(range_layer/self.dr))

        return z0 + z_arr, T0 + dT_dz * z_arr

    def T_conductive(self, T0:float, TN:float, z0:float, zN:float) -> tuple[np.ndarray]:
        if TN < T0:
            raise ValueError("Temperature at the end of the layer has to be higher than the temperature at the start of the layer")
        
        return self.linear_line(z0, T0, zN, TN)

    def T_convective(self, alpha:float, Cp:float, beta:float, p0:list[float], pN:list[float]) -> tuple[np.ndarray]:
        z0, T0 = p0
        zN, TN = pN

        Tavg = 0.5 * (T0 + TN)
        z_conductive = beta * (zN - z0)

        z1 = round((z0 + z_conductive) / self.dr) * self.dr
        z2 = round((zN - z_conductive) / self.dr) * self.dr
        z_adiabat = z2 - z1

        dT_dz = (alpha * self.gravity * Tavg) / Cp

        T1 = Tavg - 0.5 * dT_dz * z_adiabat
        T2 = T1 + dT_dz * z_adiabat

        z_c1, c1 = self.linear_line(z0, T0, z1, T1)
        z_ad, ad = self.linear_line(z1, T1, z2, T2)
        z_c2, c2 = self.linear_line(z2, T2, zN, TN)

        z_arr = np.concatenate((z_c1, z_ad[1:], z_c2[1:]))
        T_arr = np.concatenate((c1, ad[1:], c2[1:]))

        return z_arr, T_arr
    
    def Tprofile_old(self, sample_size:int, mean_params:list[float], std_params:list[float], range_Ts:list[float]):
        Ts, T3, T2, Tcmb, Tcore, beta = self.sample_params(sample_size, mean_params, std_params, range_Ts)

        dT_arr = [T3 - Ts, T2 - T3, Tcmb - T2, Tcore - Tcmb]
        Tnodes = [Tcore, Tcmb, T2, T3, Ts]

        Nr = int(self.radius/self.dr)
        Tresults = np.zeros((1+Nr, sample_size))
        z_arr = np.linspace(0.0, self.radius, 1+Nr)

        for sample_index in range(sample_size):
            Tprofile_sample = []
            zprofile_sample = []

            beta_sample = beta[sample_index]
            for i in range(len(self.rho.layers)):
                alpha_sample = self.alpha_arr[i]
                Cp_sample = self.Cp_arr[i]

                if callable(self.rho.rho_vals[i]):
                    avg_rho = 0.5*(self.rho.rho_vals[i](self.rho.layers[i][0])+self.rho.rho_vals[i](self.rho.layers[i][-1]))
                    Ra_val = self.Ra(avg_rho, alpha_sample, dT_arr[-1-i][sample_index], self.rho.layers[i][-1]-self.rho.layers[i][0], self.k_arr[i]/(avg_rho*Cp_sample), self.eta_arr[i])

                else:
                    Ra_val = self.Ra(self.rho.rho_vals[i], alpha_sample, dT_arr[-1-i][sample_index], self.rho.layers[i][-1]-self.rho.layers[i][0], self.k_arr[i]/(self.rho.rho_vals[i]*Cp_sample), self.eta_arr[i])

                if Ra_val > 100:
                    z_sample_layer, T_sample_layer = self.T_convective(alpha_sample, Cp_sample, beta_sample, [self.depth[-1-i][0], Tnodes[i+1][sample_index]], [self.depth[-1-i][-1], Tnodes[i][sample_index]])
                else:
                    z_sample_layer, T_sample_layer = self.T_conductive(Tnodes[i+1][sample_index], Tnodes[i][sample_index], self.depth[-1-i][0], self.depth[-1-i][-1])


                if i == 0:
                    zprofile_sample.append(z_sample_layer[::-1])
                    Tprofile_sample.append(T_sample_layer[::-1])
                    continue

                zprofile_sample.append(z_sample_layer[:-1][::-1])
                Tprofile_sample.append(T_sample_layer[:-1][::-1])

            zprofile_sample = np.concatenate(zprofile_sample)
            Tprofile_sample = np.concatenate(Tprofile_sample)

            if np.any(zprofile_sample[::-1] != z_arr):
                raise ValueError("Depth array of the profile is incorrect!")
            
            Tresults[:,sample_index] = Tprofile_sample[::-1]

        self.T = np.mean(Tresults, axis=1)[::-1]
        return self.T
    
    def Tprofile(self, Tcore:float, Tsurface:float):
        Ccmb = ((self.alpha_arr[0]*self.gravity)/(2*self.Cp_arr[0]))*(self.depth[-1][-1]-self.depth[-1][0])
        Tcmb = Tcore*(1-Ccmb)/(1+Ccmb)

        C2 = ((self.alpha_arr[1]*self.gravity)/(2*self.Cp_arr[1]))*(self.depth[-2][-1]-self.depth[-2][0])
        T2 = Tcmb*(1-C2)/(1+C2)

        C1 = ((self.alpha_arr[2]*self.gravity)/(2*self.Cp_arr[2]))*(self.depth[-3][-1]-self.depth[-3][0])
        T1 = T2*(1-C1)/(1+C1)

        # slope01 = self.dT_dz_Fourier(qs, self.k_arr[-1])
        # T0 = T1 - slope01 * (self.depth[-4][-1]-self.depth[-4][0])

        # print(self.depth)
        # print(self.alpha_arr)
        # print(self.Cp_arr)
        # print(self.k_arr)
        z01, Tp01 = self.linear_line(0.0, Tsurface, self.depth[0][-1], T1)
        z12, Tp12 = self.linear_line(self.depth[1][0], T1, self.depth[1][-1], T2)
        z23, Tp23 = self.linear_line(self.depth[2][0], T2, self.depth[2][-1], Tcmb)
        z34, Tp34 = self.linear_line(self.depth[3][0], Tcmb, self.depth[3][-1], Tcore)
        
        T = np.concatenate((Tp01, Tp12[1:], Tp23[1:], Tp34[1:]))[::-1]
        self.T = T
        return T
    
    def compute_T(self, Tcore:float, q0:float, eps):
        self.q, zq = self.__integrate_q(q0, 0.0, self.depth[-2][-1], eps)
        self.q = self.q[:,0]
        
        index1 = np.argmin(np.abs(zq-(self.radius-self.layers[1][0])))
        index3 = np.argmin(np.abs(zq-(self.radius-self.layers[3][0])))
        index4 = np.argmin(np.abs(zq-(self.radius-self.layers[4][0])))
        index5 = np.argmin(np.abs(zq-(self.radius-self.layers[5][0])))
        
        T01, r01 = self.__integrate_T_convective(Tcore, 0.0, self.layers[0][-1], 0)
        
        T12, r12 = self.__integrate_T_conductive(T01[-1,0], self.layers[1][0], self.layers[1][-1], index1)
        
        T23, r23 = self.__integrate_T_convective(T12[-1,0], self.layers[2][0], self.layers[2][-1], self.layer_indices[2])
        
        T34, r34 = self.__integrate_T_conductive(T23[-1,0], self.layers[3][0], self.layers[3][-1], index3)

        T45, r45 = self.__integrate_T_conductive(T34[-1,0], self.layers[4][0], self.layers[4][-1], index4)
        
        T56, r56 = self.__integrate_T_conductive(T45[-1,0], self.layers[5][0], self.layers[5][-1], index5)

        Tp = np.concatenate((T01[:,0], T12[1:,0], T23[1:,0], T34[1:,0], T45[1:,0], T56[1:,0]))

        self.T = Tp
        return self.T
    
    def update_density(self, tol:float) -> None:
        K = DensityProfile(self.layers, [149.02E9, 137.81E9, 137.81E9, 137.81E9, 137.81E9, 104.32E9])
        alpha = DensityProfile(self.layers, [5.7597E-5, 2.0363E-5, 2.0363E-5, 2.0363E-5, 2.1424E-5, 2.1424E-5])

        rho = DensityProfile(self.layers, [5555.0, 3496.35, 3496.35, 3496.35, 3496.35, 3050.0])

        rho_current = np.array([rho(r_val) for r_val in self.r])
        alpha_current = np.array([alpha(r_val) for r_val in self.r])
        K_current = np.array([K(r_val) for r_val in self.r])
        p_current = np.copy(self.p)
        T_current = np.copy(self.T)
        
        p_prev = 1E5*np.ones_like(p_current)
        T_prev = 298*np.ones_like(T_current)

        for i in range(150):
            rho_new = rho_current*(1 - alpha_current * (T_current-T_prev) + (p_current-p_prev)/K_current)
            # rho_new = rho_current*(1 - alpha_current * (T_prev-T_current) + (p_prev-p_current)/K_current)
            
            coeffs0 = np.polyfit(self.r[self.layer_indices[0]: 1+self.layer_indices[1]], rho_new[self.layer_indices[0]: 1+self.layer_indices[1]], 3)
            rho0 = np.poly1d(coeffs0)

            coeffs1 = np.polyfit(self.r[self.layer_indices[1]: 1+self.layer_indices[2]], rho_new[self.layer_indices[1]: 1+self.layer_indices[2]], 3)
            rho1 = np.poly1d(coeffs1)

            coeffs2 = np.polyfit(self.r[self.layer_indices[2]: 1+self.layer_indices[3]], rho_new[self.layer_indices[2]: 1+self.layer_indices[3]], 3)
            rho2 = np.poly1d(coeffs2)

            coeffs3 = np.polyfit(self.r[self.layer_indices[3]: 1+self.layer_indices[4]], rho_new[self.layer_indices[3]: 1+self.layer_indices[4]], 3)
            rho3 = np.poly1d(coeffs3)

            coeffs4 = np.polyfit(self.r[self.layer_indices[4]: 1+self.layer_indices[5]], rho_new[self.layer_indices[4]: 1+self.layer_indices[5]], 3)
            rho4 = np.poly1d(coeffs4)

            coeffs5 = np.polyfit(self.r[self.layer_indices[5]: 1+self.layer_indices[6]], rho_new[self.layer_indices[5]: 1+self.layer_indices[6]], 3)
            rho5 = np.poly1d(coeffs5)

            self.rho.rho_vals = [rho0, rho1, rho2, rho3, rho4, rho5]

            self.compute_profiles()
            
            change = np.max(np.abs(rho_new - rho_current))
            print(f"Max. diff. in rho at iteration {i+1}: {change} [kg/m^3]")
            if change < tol:
                break

            rho_current = np.copy(rho_new)
            p_prev = np.copy(p_current)
            T_prev = np.copy(T_current)
            p_current = np.copy(self.p)
            T_current = np.copy(self.T)

        return None


if __name__ == "__main__":

    start_main = time.time()
    
    R = 3389.5E3    # [m]
    dr = 10         # [m]

    data_rho = np.loadtxt("rho_profile.dat")

    layers = [[0, 1834.64E3], [1834.64E3, 1899.12E3], [1899.12E3, 2841.56E3], [2841.56E3, 2980.45E3], [2980.45E3, 3317.74E3], [3317.74E3, 3389.5E3]]
    rho_vals = [6100.0, 4000.0, 3750.0, 3500.0, 3300.0, 2900.0]
    alpha_arr = len(layers)*[3E-5]
    Cp_arr = [831, 1080, 1080, 1080, 741, 741]
    eta_arr = len(layers)*[10**(20.5)]
    k_arr = [40, 40, 4, 20, 9, 2.5]
    qs = 22.0E-3
    gamma = 4.152E-12
    
    Mars = Planet("Mars", layers, rho_vals, alpha_arr, Cp_arr, eta_arr, k_arr, dr, 5454395, "Euler")
    rho_initial = np.array([Mars.rho(r_val) for r_val in Mars.r])

    Mars.compute_T(2269.0, qs, gamma)

    Mars.update_density(1.0)

    # plt.figure()
    # plt.gca().invert_yaxis()
    # plt.ylabel("Depth [km]")
    # plt.xlabel(r"Density [$kg/m^3$]")
    # plt.grid()
    # plt.plot(rho_initial, (Mars.radius-Mars.r)/1E3, label="Initial")
    # plt.plot(np.array([Mars.rho(r_val) for r_val in Mars.r]), (Mars.radius-Mars.r)/1E3, label="Converged")
    # plt.plot(data_rho[:,0], (Mars.radius/1E3)-data_rho[:,1], label="Literature")
    # plt.legend()

    Mars.compute_MoI()
    Mars.report_results()

    print(f"Execution time: {round(time.time()-start_main,3)} [s]")
    # plt.show()

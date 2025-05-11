import numpy as np

class Integrator():
    def __init__(self, f:callable, method:str):
        method2propagator = {
            "RK4": self.prop_fixed,
            "RK45": self.prop_adaptive,
            "Euler": self.prop_fixed
        }
        method2integrator = {
            "RK4": self.RK4,
            "RK45": self.RK45,
            "Euler": self.Euler
        }
        self.method = method
        self.prop = method2propagator[method]
        self.integ = method2integrator[method]
        self.f = f

    def integrate(self, u0:np.ndarray, t0:float|int, T:float|int, dt:float|int, p:dict) -> tuple[np.ndarray, np.ndarray]:
        return self.prop(u0, t0, T, dt, p)
    
    def Euler(self, u:np.ndarray, t:float|int, dt:float|int, p:dict) -> np.ndarray:
        # INFO: Only computes a single step
        # NOTE: See the self.prop_fixed function for the time integration, a generic function
        return u + dt * self.f(t, u, p)

    def RK4(self, u:np.ndarray, t:float|int, dt:float|int, p:dict) -> np.ndarray:
        # INFO: Only computes a single step
        # NOTE: See the self.prop_fixed function for the time integration, a generic function
        k1 = self.f(t, u, p)
        k2 = self.f(t + 0.5*dt, u + 0.5*dt*k1, p)
        k3 = self.f(t + 0.5*dt, u + 0.5*dt*k2, p)
        k4 = self.f(t + dt, u + dt*k3, p)

        return u + (k1 + 2*k2 + 2*k3 + k4) * (dt/6)

    def RK45(self, u:np.ndarray, t:float|int, dt:float|int, tf:float|int, p:dict, tol:float=1E-8) -> tuple[np.ndarray, float|int, float|int]:
        a = np.array([0, 0.25, 3/8, 12/13, 1, 0.5])
        b = np.array([
            [0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40]
        ])
        ca = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
        c = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        diff = c - ca
        
        while True:
            k = []
            k.append(self.f(t, u, p))
            for i in range(1, 6):
                k.append(self.f(t + dt * a[i], u + dt * sum([b[i][j] * k[j] for j in range(i)]), p))
            
            e = dt * sum([diff[i] * k[i] for i in range(6)])
            err = np.max(np.abs(e))
            
            if err < tol:
                u_next = u + dt * sum([c[i] * k[i] for i in range(6)])
                break
            
            dt *= 0.9 * (tol / err) ** (1/5)
        
        t_next = t + dt
        dt_next = dt * 0.9 * (tol / err) ** (1/5)
        if t_next + dt_next > tf:
            dt_next = tf - t_next

        return u_next, dt, dt_next
    
    def prop_fixed(self, u0:np.ndarray, t0:float|int, T:float|int, dt:float|int, p:dict) -> tuple[np.ndarray, np.ndarray]:
        u_arr = [u0]
        t_arr = [t0]
        t = t0
        multiplier = 1

        # if float(t0) != 0.0:
        if p["backwards"]:
            multiplier = -1

        while (T - t) * multiplier > 1E-15:
            u_next = self.integ(u_arr[-1], t, dt, p)
            t += dt
            u_arr.append(u_next)
            t_arr.append(t)

        return np.array(u_arr), np.array(t_arr)

    def prop_adaptive(self, u0:np.ndarray, t0:float|int, T:float|int, dt0:float|int, p:dict) -> tuple[np.ndarray, np.ndarray]:
        u_arr = [u0]
        t_arr = [t0]
        t = t0
        dt = dt0
        multiplier = 1

        # if float(t0) != 0.0:
        if p["backwards"]:
            multiplier = -1

        while (T - t) * multiplier > 1E-15:
            u_next, h, h_next = self.integ(u_arr[-1], t, dt, T, p)
            t += h
            dt = h_next
            u_arr.append(u_next)
            t_arr.append(t)

        return np.array(u_arr), np.array(t_arr)

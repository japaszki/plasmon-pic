import numpy as np
import math

class constants:
    eps_0 = 8.854e-12
    mu_0 = 4*math.pi*1e-7
    q_e = 1.602e-19
    m_e = 9.109e-31
    c =  eps_0**-0.5 * mu_0**-0.5

class params:
    def __init__(self, Nx, Nz, N_rho, dx, dy, dz, dt):
        self.Nx = Nx
        self.Nz = Nz
        self.N_rho = N_rho
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        
class geometry:
    def __init__(self, bc_xmin, bc_xmax, bc_zmin, bc_zmax, eps_r, sigma, w_p, f_c):
        
        bc_list = ['pec', 'pmc', 'per']
        bc_inputs = [bc_xmin, bc_xmax, bc_zmin, bc_zmax]
        
        if any([(not (bc in bc_list)) for bc in bc_inputs]):
            raise Exception('Unrecognised boundary condition.')
            
        #Make sure either both boundaries are periodic or neither is
        if((bc_xmin == 'per') != (bc_xmax == 'per')):
            raise Exception('Both x-boundaries must be periodic.')       
        if((bc_zmin == 'per') != (bc_zmax == 'per')):
            raise Exception('Both z-boundaries must be periodic.') 
          
        self.bc_xmin = bc_xmin
        self.bc_xmax = bc_xmax
        self.bc_zmin = bc_zmin
        self.bc_zmax = bc_zmax
        self.inv_eps_r = 1./eps_r #inverse of permittivity
        self.sigma = sigma #conductivity
        self.wp_sq = w_p**2 #square of Drude plasma frequency
        self.f_c = f_c #Drude collision frequency
        
class fields:    
    def __init__(self, params, frameno):
        self.Ex = np.zeros((params.Nx, params.Nz), dtype=float)
        self.Ez = np.zeros((params.Nx, params.Nz), dtype=float)
        self.Jx = np.zeros((params.Nx, params.Nz), dtype=float)
        self.Jz = np.zeros((params.Nx, params.Nz), dtype=float)
        self.Px = np.zeros((params.Nx, params.Nz), dtype=float) # Drude response
        self.Pz = np.zeros((params.Nx, params.Nz), dtype=float) # Drude response
        self.By = np.zeros((params.Nx-1, params.Nz-1), dtype=float)
        self.frameno = frameno #frame index

class particle:
    def __init__(self, pos_x, pos_z, vx, vz, size, q, m):
        self.pos_x = [pos_x]
        self.pos_z = [pos_z]
        self.vx = [vx]
        self.vz = [vz]
        self.size = size
        self.q = q
        self.m = m
        self.collided = False

    def pos_step(self, params):
        last_pos_index = len(self.pos_x)-1
        pos_x = self.pos_x[last_pos_index]
        pos_z = self.pos_z[last_pos_index]
        
        last_vel_index = len(self.vx)-1
        vx = self.vx[last_vel_index]
        vz = self.vz[last_vel_index]
        dt = params.dt
        
        pos_x_next = pos_x + vx*dt
        pos_z_next = pos_z + vz*dt
        
        self.pos_x.append(pos_x_next)
        self.pos_z.append(pos_z_next)
        
        #perform collision check:
        dx = params.dx
        dz = params.dz
        Nx = params.Nx
        Nz = params.Nz
        
        x_min = dx * 0.5
        z_min = dz * 0.5
        x_max = dx * (Nx - 1.5)
        z_max = dz * (Nz - 1.5)
        
        if(pos_x_next >= x_max or pos_x_next <= x_min or pos_z_next >= z_max or pos_z_next <= z_min):
            self.collided = True
 
    def vel_step(self, Ex, Ez, By_curr, By_next, params, constants):
        q = self.q
        dt = params.dt
        dx = params.dx
        dz = params.dz
        
        last_pos_index = len(self.pos_x)-1
        pos_x = self.pos_x[last_pos_index]
        pos_z = self.pos_z[last_pos_index]
        
        last_vel_index = len(self.vx)-1
        vx = self.vx[last_vel_index]
        vz = self.vz[last_vel_index]
        
        By_mid = 0.5 * (By_curr + By_next)
        
        Ex_loc = self.get_local_field(Ex, pos_x, pos_z, params)
        Ez_loc = self.get_local_field(Ez, pos_x, pos_z, params)
        #note that B-grid is offset by half a step
        By_loc = self.get_local_field(By_mid, pos_x - 0.5*dx, pos_z - 0.5*dz, params)
        
        #half step of electric force
        px, pz = self.v_to_p(vx, vz, constants)
        px_half = px + 0.5*q*dt*Ex_loc
        pz_half = pz + 0.5*q*dt*Ez_loc
        
        vx_half, vz_half = self.p_to_v(px_half, pz_half, constants)
        
#        vx_half = vx + q*dt/(2*m)*Ex_loc
#        vz_half = vz + q*dt/(2*m)*Ez_loc
        
        #full step of magnetic force using half step velocity, 
        #half step of electric force
        #note directions wrt cross product vxB!
        px_next = px_half - q*dt*vz_half*By_loc + 0.5*q*dt*Ex_loc
        pz_next = pz_half + q*dt*vx_half*By_loc + 0.5*q*dt*Ez_loc
        
        vx_next, vz_next = self.p_to_v(px_next, pz_next, constants)
        
#        vx_next = vx_half - q*dt/m*vz_half*By_loc + q*dt/(2*m)*Ex_loc
#        vz_next = vz_half + q*dt/m*vx_half*By_loc + q*dt/(2*m)*Ez_loc
#        
        self.vx.append(vx_next)
        self.vz.append(vz_next)
        
    def v_to_p(self, vx, vz, constants):
        v_mag = (vx**2 + vz**2)**0.5
        
        if (v_mag != 0):
            #hard limit on particle speed
            if(v_mag > constants.c):
                v_mag = constants.c
            p_mag = self.m * v_mag * (1 - v_mag**2 * constants.c**-2)**-0.5
            px = p_mag * vx / v_mag
            pz = p_mag * vz / v_mag
        else:
            px = 0
            pz = 0
        return px, pz
        
    def p_to_v(self, px, pz, constants):
        p_mag = (px**2 + pz**2)**0.5
        
        if(p_mag != 0):
            v_mag = (p_mag / self.m) * (1 + (p_mag / self.m)**2 * constants.c**-2)**-0.5
            vx = v_mag * px / p_mag
            vz = v_mag * pz / p_mag
        else:
            vx = 0
            vz = 0
        
        return vx, vz
        
    def get_local_field(self, fld, pos_x, pos_z, params):
        dx = params.dx
        dz = params.dz
        
        x_index = pos_x / dx
        z_index = pos_z / dz
        
        x_index_int = math.floor(x_index)
        z_index_int = math.floor(z_index)
        
        x_index_frac = x_index - x_index_int
        z_index_frac = z_index - z_index_int
        
        fld_lowx_lowz = fld[x_index_int, z_index_int]
        fld_lowx_hiz = fld[x_index_int, z_index_int+1]
        fld_hix_lowz = fld[x_index_int+1, z_index_int]
        fld_hix_hiz = fld[x_index_int+1, z_index_int+1]
        
        fld_lowz_interp = fld_lowx_lowz * (1-x_index_frac) + \
        fld_hix_lowz * x_index_frac
        
        fld_hiz_interp = fld_lowx_hiz * (1-x_index_frac) + \
        fld_hix_hiz * x_index_frac
        
        fld_interp = fld_lowz_interp * (1-z_index_frac) + \
        fld_hiz_interp * z_index_frac
        
        return fld_interp
    
    def interpolate_charge(self, frame_index, params):
        q = self.q
        size = self.size
        dx = params.dx
        dy = params.dy
        dz = params.dz
        Nx = params.Nx
        Nz = params.Nz
        N_rho = params.N_rho
        
        pos_x = self.pos_x[frame_index]
        pos_z = self.pos_z[frame_index]
        
        x_index = pos_x / dx
        z_index = pos_z / dz
        
        x_index_int = math.floor(x_index)
        z_index_int = math.floor(z_index)
        
        x_index_frac = x_index - x_index_int
        z_index_frac = z_index - z_index_int
        
        rho_tot = q / (dx * dy * dz)
#        rho_hix_hiz = rho_tot * x_index_frac * z_index_frac
#        rho_lowx_hiz = rho_tot * (1-x_index_frac) * z_index_frac
#        rho_hix_lowz = rho_tot * x_index_frac * (1-z_index_frac)
#        rho_lowx_lowz =  rho_tot * (1-x_index_frac) * (1-z_index_frac)
        
        rho = np.zeros((params.Nx, params.Nz), dtype=float)
        
#        rho[x_index_int, z_index_int] = rho_lowx_lowz
#        rho[x_index_int, z_index_int+1] = rho_lowx_hiz
#        rho[x_index_int+1, z_index_int] = rho_hix_lowz
#        rho[x_index_int+1, z_index_int+1] = rho_hix_hiz
        
        #Assign relative charge density using 2D Gaussian:
        
        for x in range(x_index_int - N_rho, x_index_int + N_rho + 2):
            for z in range(z_index_int - N_rho, z_index_int + N_rho + 2):
                #Ensure indices within bounds of array:
                if(x >= 0 and x < Nx and z >= 0 and z < Nz):
                    x_rho = (x - x_index_int - x_index_frac) * dx
                    z_rho = (z - z_index_int - z_index_frac) * dz
                    rho[x,z] = math.exp(-0.5*size**-2 * (x_rho**2 + z_rho**2))
                
        #Normalise charge density to ensure total charge is correct
        rho = rho * rho_tot / np.sum(rho)
        
        return rho
    
    def interpolate_current(self, params):        
        #interpolate rho at half time step
        last_pos_index = len(self.pos_x)-1
        rho_curr = self.interpolate_charge(last_pos_index, params)
        rho_prev = self.interpolate_charge(last_pos_index-1, params)
        rho_mid = 0.5 * (rho_prev + rho_curr)
        
        vx = self.vx[last_pos_index]
        vz = self.vz[last_pos_index]
        
        Jx = rho_mid * vx
        Jz = rho_mid * vz
        
        return Jx, Jz
    
    
class static_solver:
    def __init__(self, bc_xmin, bc_xmax, bc_zmin, bc_zmax, init_V, source_V, mask, rho_over_eps, dx, dz):
        
        bc_list = ['pec', 'pmc', 'per']
        bc_inputs = [bc_xmin, bc_xmax, bc_zmin, bc_zmax]
        
        if any([(not (bc in bc_list)) for bc in bc_inputs]):
            raise Exception('Unrecognised boundary condition.')
            
        #Make sure either both boundaries are periodic or neither is:
        if((bc_xmin == 'per') != (bc_xmax == 'per')):
            raise Exception('Both x-boundaries must be periodic.')       
        if((bc_zmin == 'per') != (bc_zmax == 'per')):
            raise Exception('Both z-boundaries must be periodic.')         
        
        self.V = init_V
        self.source_V = source_V
        self.mask = mask
        self.rho_over_eps = rho_over_eps
        self.dx = dx
        self.dz = dz
        
        self.bc_xmin = bc_xmin
        self.bc_xmax = bc_xmax
        self.bc_zmin = bc_zmin
        self.bc_zmax = bc_zmax
        
        array_shape = np.shape(source_V)
        self.Nx = array_shape[0]
        self.Nz = array_shape[1]
        self.error = list()
        
        if(bc_xmin == 'pec'):
            self.mask[0,:] = True
        if(bc_zmin == 'pec'):
            self.mask[self.Nx-1,:] = True
        if(bc_zmin == 'pec'):
            self.mask[:,0] = True
        if(bc_zmax == 'pec'):
            self.mask[:,self.Nz-1] = True
        
    def iterate(self):
        V = self.V
        source_V = self.source_V
        rho_over_eps = self.rho_over_eps
        mask = self.mask
        Nx = self.Nx
        Nz = self.Nz
        dx = self.dx
        dz = self.dz
        
        #iterate succesive relaxation
#        L = 0.25*(V[0:Nx-2,1:Nz-1] + V[2:Nx,1:Nz-1] + \
#             V[1:Nx-1,0:Nz-2] + V[1:Nx-1,2:Nz])
        #experimental version for non-square cells:
        L = 0.5/(dx+dz)*(dz*(V[0:Nx-2,1:Nz-1] + V[2:Nx,1:Nz-1]) + \
             dx*(V[1:Nx-1,0:Nz-2] + V[1:Nx-1,2:Nz]))
        
        V[1:Nx-1,1:Nz-1] = L - 0.25 * rho_over_eps[1:Nx-1,1:Nz-1] * dx * dz
        
        #apply PMC boundary conditions
        if(self.bc_xmin == 'pmc'):
            V[0,:] = V[1,:]
        if(self.bc_xmax == 'pmc'):
            V[Nx-1,:] = V[Nx-2,:]
        if(self.bc_zmin == 'pmc'):
            V[:,0] = V[:,1]
        if(self.bc_zmax == 'pmc'):
            V[:,Nz-1] = V[:,Nz-2]
        
        #apply periodic boundaries
        if(self.bc_xmin == 'per' and self.bc_xmax == 'per'):
#            L_xmin = 0.25*(V[0,0:Nz-2] + V[0,2:Nz] + \
#                           V[Nx-1,1:Nz-1] + V[1,1:Nz-1])
            L_xmin = 0.5/(dx+dz)*(dx*(V[0,0:Nz-2] + V[0,2:Nz]) + \
                           dz*(V[Nx-1,1:Nz-1] + V[1,1:Nz-1]))
            V[0,1:Nz-1] = L_xmin - 0.25 * rho_over_eps[0,1:Nz-1] * dx * dz
        
#            L_xmax = 0.25*(V[Nx-1,0:Nz-2] + V[Nx-1,2:Nz] + \
#                           V[Nx-2,1:Nz-1] + V[0,1:Nz-1])
            L_xmax = 0.5/(dx+dz)*(dx*(V[Nx-1,0:Nz-2] + V[Nx-1,2:Nz]) + \
                           dz*(V[Nx-2,1:Nz-1] + V[0,1:Nz-1]))
            V[Nx-1,1:Nz-1] = L_xmax - 0.25 * rho_over_eps[Nx-1,1:Nz-1] * dx * dz
        
        if(self.bc_zmin == 'per' and self.bc_zmax == 'per'):
#            L_zmin = 0.25*(V[0:Nx-2,0] + V[2:Nx,0] + \
#                           V[1:Nx-1,Nz-1] + V[1:Nx-1,1])
            L_zmin = 0.5/(dx+dz)*(dz*(V[0:Nx-2,0] + V[2:Nx,0]) + \
                           dx*(V[1:Nx-1,Nz-1] + V[1:Nx-1,1]))
            V[1:Nx-1,0] = L_zmin - 0.25 * rho_over_eps[1:Nx-1,0] * dx * dz
        
#            L_zmax = 0.25*(V[0:Nx-2,Nz-1] + V[2:Nx,Nz-1] + \
#                           V[1:Nx-1,Nz-2] + V[1:Nx-1,0])
            L_zmax = 0.5/(dx+dz)*(dz*(V[0:Nx-2,Nz-1] + V[2:Nx,Nz-1]) + \
                           dx*(V[1:Nx-1,Nz-2] + V[1:Nx-1,0]))
            V[1:Nx-1,Nz-1] = L_zmax - 0.25 * rho_over_eps[1:Nx-1,Nz-1] * dx * dz
        
        #apply source potentials
        V = np.transpose(np.array([[source_V[x,z] if mask[x,z] else V[x,z]\
                                    for x in range(0,Nx)] \
                                    for z in range(0,Nz)]))
        delta = self.V[1:Nx-1,1:Nz-1] - V[1:Nx-1,1:Nz-1]
        #update potential matrix
        self.V = V
    
        #record max error 
        self.error.append(np.abs(delta).max())
        
    def get_E(self):
        V = self.V
        Nx = self.Nx
        Nz = self.Nz
        dx = self.dx
        dz = self.dz
        mask = self.mask
        
        Ex = np.zeros((Nx, Nz), dtype=float)
        Ez = np.zeros((Nx, Nz), dtype=float)
        
        Ex[1:Nx-1,1:Nz-1] = 0.5 * (V[2:Nx,1:Nz-1] - V[0:Nx-2,1:Nz-1]) / dx
        Ez[1:Nx-1,1:Nz-1] = 0.5 * (V[1:Nx-1,2:Nz] - V[1:Nx-1,0:Nz-2]) / dz
        
        #apply PMC boundary conditions
        if(self.bc_xmin == 'pmc'):
            Ez[0,:] = Ez[1,:]
            Ex[0,:] = 0
        if(self.bc_xmax == 'pmc'):
            Ez[Nx-1,:] = Ez[Nx-2,:]
            Ex[Nx-1,:] = 0
        if(self.bc_zmin == 'pmc'):
            Ex[:,0] = Ex[:,1]
            Ez[:,0] = 0
        if(self.bc_zmax == 'pmc'):
            Ex[:,Nz-1] = Ex[:,Nz-2]
            Ez[:,Nz-1] = 0
                     
        #periodic boundary conditions
        if(self.bc_xmin == 'per' and self.bc_xmax == 'per'):
            Ez[0,1:Nz-1] = 0.5 * (V[0,2:Nz] - V[0,0:Nz-2]) / dz
            Ex[0,1:Nz-1] = 0.5 * (V[1,1:Nz-1] - V[Nx-1,1:Nz-1]) / dx
            
            Ez[Nx-1,1:Nz-1] = 0.5 * (V[Nx-1,2:Nz] - V[Nx-1,0:Nz-2]) / dz
            Ex[Nx-1,1:Nz-1] = 0.5 * (V[0,1:Nz-1] - V[Nx-2,1:Nz-1]) / dx
            
        if(self.bc_zmin == 'per' and self.bc_zmax == 'per'):
            Ex[1:Nx-1,0] = 0.5 * (V[2:Nx,0] - V[0:Nx-2,0]) / dx
            Ez[1:Nx-1,0] = 0.5 * (V[1:Nx-1,1] - V[1:Nx-1,Nz-1]) / dz
            
            Ex[1:Nx-1,Nz-1] = 0.5 * (V[2:Nx,Nz-1] - V[0:Nx-2,Nz-1]) / dx
            Ez[1:Nx-1,Nz-1] = 0.5 * (V[1:Nx-1,0] - V[1:Nx-1,Nz-2]) / dz
        
        #ensure zero field in source regions
        #also reverse direction of field
        Ex = np.transpose(np.array([[0 if mask[x,z] else -Ex[x,z]\
                                    for x in range(0,Nx)] \
                                    for z in range(0,Nz)]))
        Ez = np.transpose(np.array([[0 if mask[x,z] else -Ez[x,z]\
                                    for x in range(0,Nx)] \
                                    for z in range(0,Nz)]))
    
        return Ex, Ez
        
class solver:    
    def __init__(self, params, constants, geometry):
        self.params = params
        self.constants = constants
        self.geometry = geometry
        self.frames = [fields(self.params, 0)]
        self.particles = []
        
        dt_max = min(params.dx, params.dz) * math.sqrt(0.5*constants.eps_0*constants.mu_0)
        if(params.dt > dt_max):
            raise Exception('Courant criterion not satisfied! Max dt = {:.2e}'.format(dt_max))
    
    def get_last_frame(self):
        last_index = len(self.frames)-1
        return self.frames[last_index]
    
    def curl_Ey(self, Ex, Ez):
        Nx = self.params.Nx
        Nz = self.params.Nz
        dx = self.params.dx
        dz = self.params.dz
        
        dEz_dx_full = (Ez[1:Nx, :] - Ez[0:Nx-1, :])/dx
        dEz_dx_avg = 0.5 * (dEz_dx_full[:, 0:Nz-1] + dEz_dx_full[:, 1:Nz])
        
        dEx_dz_full = (Ex[:, 1:Nz] - Ex[:, 0:Nz-1])/dz
        dEx_dz_avg = 0.5 * (dEx_dz_full[0:Nx-1, :] + dEx_dz_full[1:Nx, :])
        
        return dEz_dx_avg - dEx_dz_avg

    def curl_Bx(self, By):
        Nx = self.params.Nx
        Nz = self.params.Nz
        dz = self.params.dz
        
        #note that B-grid has 1 smaller dimensions than E-grid
        dBy_dz_full = (By[:, 1:Nz-1] - By[:, 0:Nz-2])/dz
        dBy_dz_avg = 0.5 * (dBy_dz_full[0:Nx-2, :] + dBy_dz_full[1:Nx-1, :])
        return -dBy_dz_avg

    def curl_Bz(self, By):
        Nx = self.params.Nx
        Nz = self.params.Nz
        dx = self.params.dx
        
        #note that B-grid has 1 smaller dimensions than E-grid
        dBy_dx_full = (By[1:Nx-1, :] - By[0:Nx-2, :])/dx
        dBy_dx_avg = 0.5 * (dBy_dx_full[:, 0:Nz-2] + dBy_dx_full[:, 1:Nz-1]) 
        return dBy_dx_avg
        
    def propagate_B(self, Ex, Ez, By):
        dt = self.params.dt
        return By + self.curl_Ey(Ex, Ez) * dt

    def propagate_E(self, Ex, Ez, Px, Pz, By, Jx, Jz):
        Nx = self.params.Nx
        Nz = self.params.Nz
        dt = self.params.dt
        mu_0 = self.constants.mu_0
        eps_0 = self.constants.eps_0
        inv_eps_r = self.geometry.inv_eps_r[1:Nx-1, 1:Nz-1]
        wp_sq =  self.geometry.wp_sq[1:Nx-1, 1:Nz-1]
        
        Ex_next = Ex   
        Ex_next[1:Nx-1, 1:Nz-1] = Ex_next[1:Nx-1, 1:Nz-1] + \
        np.multiply((self.curl_Bx(By) / mu_0 - Jx[1:Nx-1, 1:Nz-1]), inv_eps_r) / eps_0 * dt -\
        np.multiply(wp_sq, Px[1:Nx-1, 1:Nz-1]) * dt**2
        
        Ez_next = Ez
        Ez_next[1:Nx-1, 1:Nz-1] = Ez_next[1:Nx-1, 1:Nz-1] + \
        np.multiply((self.curl_Bz(By) / mu_0 - Jz[1:Nx-1, 1:Nz-1]), inv_eps_r) / eps_0 * dt -\
        np.multiply(wp_sq, Pz[1:Nx-1, 1:Nz-1]) * dt**2
        
        return Ex_next, Ez_next
    
    def Drude_step(self, Px, Pz, Ex, Ez):
        dt = self.params.dt
        f_c = self.geometry.f_c
        
        P_scaling = np.exp(-f_c * dt)
        Px_next = np.multiply(P_scaling, Px) + Ex 
        Pz_next = np.multiply(P_scaling, Pz) + Ez
        
        return Px_next, Pz_next
    
    def apply_pec_boundaries(self, Ex, Ez, By):
        Nx = self.params.Nx
        Nz = self.params.Nz
        dx = self.params.dx
        dz = self.params.dz
        dt = self.params.dt
        mu_0 = self.constants.mu_0
        eps_0 = self.constants.eps_0
        inv_eps_r = self.geometry.inv_eps_r
        
        Ex_bc = Ex
        Ez_bc = Ez
        
        if(self.geometry.bc_xmin == 'pec'):
            #zero tangential field
            Ez_bc[0,:] = 0
            
            #obtain normal field using curl of B, treating PEC as symmetry plane
            dBy_dz = (By[0, 1:Nz-1] - By[0, 0:Nz-2])/dz
            Ex_bc[0,1:Nz-1] = Ex_bc[0,1:Nz-1] + \
            np.multiply(-dBy_dz, inv_eps_r[0,1:Nz-1]) / (mu_0 * eps_0) * dt
            
            
        if(self.geometry.bc_xmax == 'pec'):
            #zero tangential field
            Ez_bc[Nx-1,:] = 0
            
            #obtain normal field using curl of B, treating PEC as symmetry plane
            dBy_dz = (By[Nx-2, 1:Nz-1] - By[Nx-2, 0:Nz-2])/dz
            Ex_bc[Nx-1,1:Nz-1] = Ex_bc[Nx-1,1:Nz-1] + \
            np.multiply(-dBy_dz, inv_eps_r[Nx-1,1:Nz-1]) / (mu_0 * eps_0) * dt

            
        if(self.geometry.bc_zmin == 'pec'):
            #zero tangential field
            Ex_bc[:,0] = 0
            
            #obtain normal field using curl of B, treating PEC as symmetry plane
            dBy_dx = (By[1:Nx-1, 0] - By[0:Nx-2,0])/dx
            Ez_bc[1:Nx-1,0] = Ez_bc[1:Nx-1,0] + \
            np.multiply(dBy_dx, inv_eps_r[1:Nx-1,0]) / (mu_0 * eps_0) * dt
            
            
        if(self.geometry.bc_zmax == 'pec'):
            #zero tangential field
            Ex_bc[:,Nz-1] = 0
            
            #obtain normal field using curl of B, treating PEC as symmetry plane
            dBy_dx = (By[1:Nx-1, Nz-2] - By[0:Nx-2, Nz-2])/dx
            Ez_bc[1:Nx-1, Nz-1] = Ez_bc[1:Nx-1, Nz-1] + \
            np.multiply(dBy_dx, inv_eps_r[1:Nx-1, Nz-1]) / (mu_0 * eps_0) * dt
            
        return Ex_bc, Ez_bc
        

    def apply_pmc_boundaries(self, By):
        Nx = self.params.Nx
        Nz = self.params.Nz
        
        By_bc = By
              
        if(self.geometry.bc_xmin == 'pmc'):
            By_bc[0,:] = 0
            
        if(self.geometry.bc_xmax == 'pmc'):
            By_bc[Nx-2,:] = 0
            
        if(self.geometry.bc_zmin == 'pmc'):
            By_bc[:,0] = 0
            
        if(self.geometry.bc_zmax == 'pmc'):
            By_bc[:,Nz-2] = 0
        
        return By_bc
    
    def apply_per_boundaries(self, Ex, Ez, Px, Pz, By, Jx, Jz):
        Nx = self.params.Nx
        Nz = self.params.Nz
        dx = self.params.dx
        dz = self.params.dz
        dt = self.params.dt
        mu_0 = self.constants.mu_0
        eps_0 = self.constants.eps_0
        inv_eps_r = self.geometry.inv_eps_r
        wp_sq =  self.geometry.wp_sq
        
        Ex_bc = Ex
        Ez_bc = Ez
        
        if(self.geometry.bc_xmin == 'per' and self.geometry.bc_xmax == 'per'):           
            #calculate curl of B at boundary, looping over grid
            dBy_dz_low_edge = (By[0, 1:Nz-1] - By[0, 0:Nx-2])/dz
            dBy_dz_hi_edge = (By[Nx-2, 1:Nz-1] - By[Nx-2, 0:Nz-2])/dz
            dBy_dz_avg = 0.5 * (dBy_dz_low_edge + dBy_dz_hi_edge) 
            
            #same operation as propagate_E but applied to special case of boundary
            Ex_bc[0,1:Nz-1] = Ex_bc[0,1:Nz-1] + \
            np.multiply((-dBy_dz_avg / mu_0 - Jx[0,1:Nz-1]), inv_eps_r[0,1:Nz-1]) / eps_0 * dt -\
            np.multiply(wp_sq[0,1:Nz-1], Px[0,1:Nz-1]) * dt**2
            
            #calculate curl of B at boundary, looping over grid
            dBy_dx_full = (By[0, :] - By[Nx-2, :])/dx
            dBy_dx_avg = 0.5 * (dBy_dx_full[0:Nx-2] + dBy_dx_full[1:Nx-1])
            
            #same operation as propagate_E but applied to special case of boundary
            Ez_bc[0,1:Nz-1] = Ez_bc[0,1:Nz-1] + \
            np.multiply((dBy_dx_avg / mu_0 - Jz[0,1:Nz-1]), inv_eps_r[0,1:Nz-1]) / eps_0 * dt -\
            np.multiply(wp_sq[0,1:Nz-1], Pz[0,1:Nz-1]) * dt**2
            
            #apply result to both edges
            Ex_bc[Nx-1,:] = Ex_bc[0,:]
            Ez_bc[Nx-1,:] = Ez_bc[0,:]
        
        if(self.geometry.bc_zmin == 'per' and self.geometry.bc_zmax == 'per'):           
            #calculate curl of B at boundary, looping over grid
            dBy_dz_full = (By[:, 0] - By[:, Nz-2])/dz
            dBy_dz_avg = 0.5 * (dBy_dz_full[0:Nx-2] + dBy_dz_full[1:Nx-1])
            
            #same operation as propagate_E but applied to special case of boundary
            Ex_bc[1:Nx-1,0] = Ex_bc[1:Nx-1,0] + \
            np.multiply((-dBy_dz_avg / mu_0 - Jx[1:Nx-1,0]), inv_eps_r[1:Nx-1,0]) / eps_0 * dt -\
            np.multiply(wp_sq[1:Nx-1,0], Px[1:Nx-1,0]) * dt**2
            
            #calculate curl of B at boundary, looping over grid
            dBy_dx_low_edge = (By[1:Nx-1, 0] - By[0:Nx-2, 0])/dx
            dBy_dx_hi_edge = (By[1:Nx-1, Nz-2] - By[0:Nx-2, Nz-2])/dx
            dBy_dx_avg = 0.5 * (dBy_dx_low_edge + dBy_dx_hi_edge) 
            
            #same operation as propagate_E but applied to special case of boundary
            Ez_bc[1:Nx-1,0] = Ez_bc[1:Nx-1,0] + \
            np.multiply((dBy_dx_avg / mu_0 - Jz[1:Nx-1,0]), inv_eps_r[1:Nx-1,0]) / eps_0 * dt -\
            np.multiply(wp_sq[1:Nx-1,0], Pz[1:Nx-1,0]) * dt**2
            
            #apply result to both edges
            Ex_bc[:,Nz-1] = Ex_bc[:,0]
            Ez_bc[:,Nz-1] = Ez_bc[:,0]
            
        return Ex_bc, Ez_bc
    
    def apply_conductive_media(self, Ex_curr, Ex_next, Ez_curr, Ez_next):
        eps_0 = self.constants.eps_0
        dt = self.params.dt
        sigma = self.geometry.sigma
        inv_eps_r = self.geometry.inv_eps_r
        
        Ex_mid = 0.5 * (Ex_curr + Ex_next)
        Ez_mid = 0.5 * (Ez_curr + Ez_next)
        
        Ex_cond = Ex_next - np.multiply(np.multiply(Ex_mid, sigma), inv_eps_r) / eps_0 * dt
        Ez_cond = Ez_next - np.multiply(np.multiply(Ez_mid, sigma), inv_eps_r) / eps_0 * dt
            
        return Ex_cond, Ez_cond
    
    def propagate_particles(self, Ex, Ez, By_curr, By_next):
        for curr_particle in self.particles:
            if(not curr_particle.collided):
                curr_particle.vel_step(Ex, Ez, By_curr, By_next, self.params, self.constants)
                curr_particle.pos_step(self.params)
                
    def obtain_particle_J(self):
        Jx = np.zeros((self.params.Nx, self.params.Nz), dtype=float)
        Jz = np.zeros((self.params.Nx, self.params.Nz), dtype=float)
        
        for curr_particle in self.particles:
            if(not curr_particle.collided):
                dJx, dJz = curr_particle.interpolate_current(self.params)
                Jx = Jx + dJx
                Jz = Jz + dJz
        
        return Jx, Jz
                
    def propagate_sim(self):
        last_index = len(self.frames)-1
        fields_curr = self.frames[last_index]
        
        Ex_curr = fields_curr.Ex
        Ez_curr = fields_curr.Ez
        Px_curr = fields_curr.Px
        Pz_curr = fields_curr.Pz
        By_curr = fields_curr.By
            
        By_next = self.propagate_B(Ex_curr, Ez_curr, By_curr)
        By_next = self.apply_pmc_boundaries(By_next)
        
        self.propagate_particles(Ex_curr, Ez_curr, By_curr, By_next)
        Jx, Jz = self.obtain_particle_J()
        
        Ex_next, Ez_next = self.propagate_E(Ex_curr, Ez_curr, Px_curr, Pz_curr,\
                                            By_next, Jx, Jz)
        Ex_next, Ez_next = self.apply_per_boundaries(Ex_next, Ez_next, Px_curr, Pz_curr,\
                                            By_next, Jx, Jz)
        Ex_next, Ez_next = self.apply_conductive_media(Ex_curr, Ex_next, Ez_curr, Ez_next)
        Ex_next, Ez_next = self.apply_pec_boundaries(Ex_next, Ez_next, By_next)
        
        Px_next, Pz_next = self.Drude_step(Px_curr, Pz_curr, Ex_next, Ez_next)
        
        self.frames.append(fields(self.params, fields_curr.frameno+1))
        self.frames[last_index+1].By = By_next
        self.frames[last_index+1].Ex = Ex_next
        self.frames[last_index+1].Ez = Ez_next
        self.frames[last_index+1].Px = Px_next
        self.frames[last_index+1].Pz = Pz_next
        self.frames[last_index+1].Jx = Jx
        self.frames[last_index+1].Jz = Jz
    
    
    def calc_energy(self, frame):
        mu_0 = self.constants.mu_0
        eps_0 = self.constants.eps_0
        dx = self.params.dx
        dy = self.params.dy
        dz = self.params.dz
        
        Ex = self.frames[frame].Ex
        Ez = self.frames[frame].Ez
        By = self.frames[frame].By
        inv_eps_r = self.geometry.inv_eps_r
        
        energy_e = 0.5*eps_0*dx*dy*dz*sum(sum((np.power(Ex, 2) + np.power(Ez, 2))/inv_eps_r))
        energy_b = 0.5*dx*dy*dz*sum(sum(np.power(By, 2))/mu_0)
        
        return energy_e + energy_b

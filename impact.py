import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pic_solver


def plot_Emag(fields, params):
    frameno = fields.frameno
    E_mag = np.power((np.power(fields.Ex, 2) + np.power(fields.Ez, 2)), 0.5)
    
    plot_y, plot_x = np.mgrid[0:params.Nx+1, 0:params.Nz+1]
    plt.pcolormesh(plot_x, plot_y, E_mag, cmap='hot', shading='flat', vmin=0, vmax=5e7)#E_mag.max())
    plt.title('|E|, t = {:.2f} fs'.format(frameno*params.dt*1e15))
    plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
    plt.xlabel('z position (cells)')
    plt.ylabel('x position (cells)')
    plt.colorbar()
    plt.rc('font', size=16)
    plt.savefig('./results/frame' + str(frameno) + '.png')
    plt.show()

def plot_V(fields, params):
    Nx = params.Nx
    Nz = params.Nz
    frameno = fields.frameno
    Ex_line = fields.Ex[:,midpoint]
    Vx_line = [sum(Ex_line[0:z])*params.dx for z in range(Nz-1)]
    
    plt.plot([1e9*dx*x for x in range(0, Nx-1)], Vx_line)
    plt.axis([0, 1e9*dx*(Nx-1), -4, 8])
    plt.ylabel('Potential (V)')
    plt.xlabel('x position (nm)')
    plt.title('t = {:.2f} fs'.format(frameno*params.dt*1e15))
    plt.rc('font', size=16)
    plt.savefig('./results/frame' + str(frameno) + '.png')
    plt.show()
    
#Create directory to write results if does not already exist:
if not os.path.exists('./results/'):
    os.makedirs('./results/')

#initialise solver object
Nx = 200
Nz = 200
N_rho = 5 #Number of cells to spread charge over
dx = 2e-9
dy = 2e-9
dz = 2e-9
dt = 2.5e-18
params_obj = pic_solver.params(Nx, Nz, N_rho, dx, dy, dz, dt)
constants_obj = pic_solver.constants()

eps_r = np.ones((Nx, Nz), dtype=float)
sigma = np.zeros((Nx, Nz), dtype=float)
w_p = np.zeros((Nx, Nz), dtype=float)
f_c = np.zeros((Nx, Nz), dtype=float)

#Set up absorbing layer
pml_thickness = 6
taper_thickness = 6
sigma_0 = math.sqrt(constants_obj.eps_0 / constants_obj.mu_0) / (params_obj.dx)

for x in range(1,taper_thickness+1):
    sigma[Nx-pml_thickness-x, pml_thickness+x-1:Nz-pml_thickness-x+1] = sigma_0 * math.pow(2, -x)
    sigma[0:Nx-pml_thickness-x, pml_thickness+x-1] = sigma_0 * math.pow(2, -x)
    sigma[0:Nx-pml_thickness-x, Nz-pml_thickness-x] = sigma_0 * math.pow(2, -x)
sigma[Nx-pml_thickness:Nx, :] = sigma_0
sigma[Nx-pml_thickness:Nx, :] = sigma_0
sigma[0:Nx, 0:pml_thickness] = sigma_0
sigma[0:Nx, Nz-pml_thickness:Nz] = sigma_0


#define copper region
midpoint = math.floor(Nz/2)
surface_height = math.floor(Nx/4)

copper_shape = np.zeros((Nx, Nz), dtype=bool)
copper_shape[0:surface_height,:] = True

w_p[copper_shape] = 1.32e16
f_c[copper_shape] = 10.5e13

geometry_obj = pic_solver.geometry('pec', 'pec', 'pec', 'pec', eps_r, sigma, w_p, f_c)
solver_obj = pic_solver.solver(params_obj, constants_obj, geometry_obj)

#Set up particles
q_e = pic_solver.constants.q_e
m_e = pic_solver.constants.m_e
particle_size = 2e-9

#Give particle a small amount of initial kinetic energy to stop it getting stuck
# due to charge rasterisation issues
v_init = -5e7
barrier_width = 30e-9

#Don't add this particle to the solver until later
particle_1 = pic_solver.particle(0.98*Nx*dx, 0.5*Nz*dz, v_init, 0, particle_size, -q_e, m_e)

##Missing charge at emission site:
#particle_2 = pic_solver.particle((surface_height-1)*dx, 0.5*Nz*dz, 0, 0, particle_size, -q_e, m_e)
#solver_obj.particles.append(particle_2)
#
##Positive charge at end of potential barrier, for charge conservation:
#particle_3 = pic_solver.particle((surface_height-1)*dx + barrier_width, 0.5*Nz*dz, 0, 0, particle_size, q_e, m_e)
#solver_obj.particles.append(particle_3)

#DC solver needed to generate initial conditions

#Set up geometry for DC solver
dc_sources = np.zeros((Nx, Nz), dtype=float)
dc_mask = np.zeros((Nx, Nz), dtype=float)
dc_mask[0,:] = True
dc_mask[Nx-1,:] = True

#The electrostatic solver expects an initial potential distribution.
#It converges much faster if it is initialised with a guess of the result that is roughly correct
init_V = np.zeros((Nx, Nz), dtype=float)
    
#obtain charge density from particle distribution
rho_init = np.zeros((Nx, Nz), dtype=float)
    
for curr_particle in solver_obj.particles:
    if(not curr_particle.collided):
        d_rho = curr_particle.interpolate_charge(0, solver_obj.params)
        rho_init = rho_init + d_rho
        
rho_over_eps = np.divide(rho_init, eps_r) / constants_obj.eps_0      

#Initialise DC solver
dc_obj = pic_solver.static_solver('pec', 'pec', 'pmc', 'pmc', init_V, dc_sources, dc_mask, rho_over_eps, dx, dz)

#Solve DC potential
#Run for enough iterations to allow solution to propagate through entire domain
for x in range(0, max([Nx, Nz])+300):
    dc_obj.iterate()

dc_V = dc_obj.V

#Plot DC potential
plot_y, plot_x = np.mgrid[0:params_obj.Nx+1, 0:params_obj.Nz+1]
plt.pcolormesh(plot_x, plot_y, dc_V, cmap='afmhot', shading='flat', vmin=0, vmax=dc_V.max())
plt.title('Static V')
plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
plt.xlabel('x position (cells)')
plt.ylabel('z position (cells)')
plt.colorbar()
plt.rc('font', size=16)
plt.show()

dc_Ex, dc_Ez = dc_obj.get_E()
dc_Emag = np.power(np.power(dc_Ex, 2) + np.power(dc_Ez, 2), 0.5)

#Plot full DC field
plt.pcolormesh(plot_x, plot_y, dc_Emag, cmap='hot', shading='flat', vmin=0, vmax=dc_Emag.max())
plt.title('Static E')
plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
plt.xlabel('x position (cells)')
plt.ylabel('z position (cells)')
plt.rc('font', size=16)
plt.show()

#Plot convergence of static solver
plt.semilogy(dc_obj.error)
plt.title('Static solver convergence')
plt.xlabel('Iteration')
plt.show()

#initialise FDTD solver with electrostatic field:
solver_obj.frames[0].Ex = dc_Ex
solver_obj.frames[0].Ez = dc_Ez

#remove positive charges from fdtd solver:
#solver_obj.particles.remove(particle_2)
#solver_obj.particles.remove(particle_3)

#add actual particle:
solver_obj.particles.append(particle_1)

#run FDTD simulation
blocks = 20
disp_per_block = 100
disp_interval = 16

#DC field to accelerate particle
E_dc = 0

Ex_vec = []

for x in range(0,blocks):

    #Clear data from previous block to avoid filling up memory
    if(x >= 1):
        last_frame = solver_obj.get_last_frame()
        solver_obj.frames = [pic_solver.fields(params_obj, last_frame.frameno+1)]
        solver_obj.frames[0] = last_frame
        
    for y in range(0,disp_per_block):
        for z in range(0,disp_interval):
            solver_obj.propagate_sim()
            
            #Apply DC field to particle separately
            #The assumed DC field distribution is not compatible with the absorbers 
            #on the sides of the domain and would collapse immediately
            solver_obj.particles[0].vx[-1] += q_e*E_dc*dt/m_e
                    
        last_frame = solver_obj.get_last_frame()
        plot_Emag(last_frame, params_obj)
#        plot_V(last_frame, params_obj)
        
        #Record surface field at emission location
        Ex_vec.append(last_frame.Ex[surface_height, midpoint])

last_index = len(solver_obj.frames)-1
time_vec = [x*dt for x in range(0, solver_obj.frames[last_index].frameno, disp_interval)]

#plot surface field at inmpact point vs. time
plt.plot(time_vec, [x*1e-6 for x in Ex_vec])
plt.ylabel('Surface E field (MV/m)')
plt.xlabel('Time (s)')
plt.show()
    
#plot particle trajectory
particle_posx = solver_obj.particles[0].pos_x
particle_vx = solver_obj.particles[0].vx
particle_time = [x*dt for x in range(0, len(particle_posx))]

plt.plot(particle_time, [1e9 * x for x in particle_posx])
plt.ylabel('Particle x position (nm)')
plt.xlabel('Time (s)')
plt.show()

plt.plot(particle_time, [vx for vx in particle_vx])
plt.ylabel('Particle x velocity (m/s)')
plt.xlabel('Time (s)')
plt.show()

plt.plot([1e9 * x for x in particle_posx], [0.5 * m_e * vx**2 / q_e for vx in particle_vx])
plt.ylabel('Particle x kinetic energy (eV)')
plt.xlabel('Particle x position (nm)')
plt.show()

#notes:
# - Teleportation model of tunneling: particle starts its trajectory 30 nm (ie
#   end of potential barrier), while the field of a positive charge is placed 
#   at the emission site to represent the removal of a charge.

# - Particle is given some initial kinetic energy to help overcome forces that
#   keep it stuck in place due to the rasterisation of the charge

# - Note that the 30 nm value for potential barrier width may change with DC field
#   and mesh size! Double-check this if either value is changed!
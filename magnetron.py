import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pic_solver


def plot_Emag(fields, params):
    frameno = fields.frameno
    E_mag = np.power((np.power(fields.Ex - dc_Ex, 2) + np.power(fields.Ez - dc_Ez, 2)), 0.5)
    
    plot_y, plot_x = np.mgrid[0:params.Nx+1, 0:params.Nz+1]
    plt.pcolormesh(plot_x, plot_y, E_mag, cmap='hot', shading='flat', vmin=0, vmax=3e5)#E_mag.max())
    plt.title('|E|, t = {:.2f} ns'.format(frameno*params.dt*1e9))
    plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
    plt.xlabel('z position (cells)')
    plt.ylabel('x position (cells)')
    plt.colorbar()
    plt.rc('font', size=16)
    plt.savefig('./results/frame' + str(frameno) + '.png')
    plt.show()
    
#Create directory to write results if does not already exist:
if not os.path.exists('./results/'):
    os.makedirs('./results/')

#initialise solver object
Nx = 250
Nz = 250
N_rho = 5 #Number of cells to spread charge over
dx = 0.75e-3
dy = 1e-2
dz = 0.75e-3
dt = 1e-12
params_obj = pic_solver.params(Nx, Nz, N_rho, dx, dy, dz, dt)
constants_obj = pic_solver.constants()

eps_r = np.ones((Nx, Nz), dtype=float)
sigma = np.zeros((Nx, Nz), dtype=float)
w_p = np.zeros((Nx, Nz), dtype=float)
f_c = np.zeros((Nx, Nz), dtype=float)

r_central = 40e-3
r_cav = 20e-3
r_pos_cav = 61e-3
N_cav = 8
slot_ang_width = 8 * math.pi / 180
r_cathode  = 20e-3


#define anode
anode_shape = np.ones((Nx, Nz), dtype=bool)

#Cut out central cavity
for x in range(0,Nx):
    for z in range(0,Nz):
        if ((x-0.5*Nx)*dx)**2 + ((z-0.5*Nz)*dz)**2 <= r_central**2:
            anode_shape[x,z] = False

#Cut out resonating cavities
for cav_index in range(0, N_cav):
    for x in range(0,Nx):
        for z in range(0,Nz):
            #Define centre of cavity:
            cav_theta = 2*math.pi * cav_index / N_cav
            cav_centre_x = r_pos_cav * math.cos(cav_theta)
            cav_centre_z = r_pos_cav * math.sin(cav_theta)
            
            if ((x-0.5*Nx)*dx - cav_centre_x)**2 + ((z-0.5*Nz)*dz - cav_centre_z)**2 <= r_cav**2:
                anode_shape[x,z] = False
                
#Cut out coupling holes
for cav_index in range(0, N_cav):
    for x in range(0,Nx):
        for z in range(0,Nz):
            #Define angle of cavity:
            cav_theta = 2*math.pi * cav_index / N_cav
            curr_theta = math.atan2((x-0.5*Nx), (z-0.5*Nz))
            curr_r_sq = ((x-0.5*Nx)*dx)**2 + ((z-0.5*Nz)*dz)**2
            
            if abs(curr_theta % (2*math.pi) - cav_theta % (2*math.pi)) <= slot_ang_width and curr_r_sq <= r_pos_cav**2:
                anode_shape[x,z] = False
            
cathode_shape = np.zeros((Nx, Nz), dtype=bool)

#Create cathode
for x in range(0,Nx):
    for z in range(0,Nz):
        if ((x-0.5*Nx)*dx)**2 + ((z-0.5*Nz)*dz)**2 <= r_cathode**2:
            cathode_shape[x,z] = True
            
copper_shape = np.logical_or(cathode_shape, anode_shape)
            
plt.matshow(copper_shape)

w_p[copper_shape] = 1e11
f_c[copper_shape] = 1e8

geometry_obj = pic_solver.geometry('pec', 'pec', 'pec', 'pec', eps_r, sigma, w_p, f_c)
solver_obj = pic_solver.solver(params_obj, constants_obj, geometry_obj)

#DC solver needed to generate initial conditions

#Set up geometry for DC solver
dc_sources = cathode_shape * -2e4
dc_mask = copper_shape

#The electrostatic solver expects an initial potential distribution.
#It converges much faster if it is initialised with a guess of the result that is roughly correct
init_V = np.zeros((Nx, Nz), dtype=float)    
rho_over_eps = np.zeros((Nx, Nz), dtype=float)

#Initialise DC solver
dc_obj = pic_solver.static_solver('pec', 'pec', 'pec', 'pec', init_V, dc_sources, dc_mask, rho_over_eps, dx, dz)

#Solve DC potential
#Run for enough iterations to allow solution to propagate through entire domain
for x in range(0, max([Nx, Nz]) + 1000):
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
plt.colorbar()
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

#add magnetic field:
solver_obj.frames[0].By = 0.033

#add particles:
q_e = pic_solver.constants.q_e
m_e = pic_solver.constants.m_e
particle_size = 1e-3
particle_pop = 1e9
particle_num = 16
particle_offset = -0.5e-3
particle_v_init = 1e6

for i in range(0,particle_num):
    particle_theta = 2*math.pi*i/particle_num
    particle_deltax = (r_cathode + particle_offset) * math.cos(particle_theta)
    particle_deltaz = (r_cathode + particle_offset) * math.sin(particle_theta)
    particle_vx_init = particle_v_init * math.cos(particle_theta)
    particle_vz_init = particle_v_init * math.sin(particle_theta)
    
    solver_obj.particles.append(pic_solver.particle(0.5*Nx*dx + particle_deltax, \
        0.5*Nz*dz + particle_deltaz, particle_vx_init, particle_vz_init, \
        particle_size, -particle_pop*q_e, particle_pop*m_e))

#run FDTD simulation
blocks = 10
disp_per_block = 100
disp_interval = 32

for x in range(0,blocks):

    #Clear data from previous block to avoid filling up memory
    if(x >= 1):
        last_frame = solver_obj.get_last_frame()
        solver_obj.frames = [pic_solver.fields(params_obj, last_frame.frameno+1)]
        solver_obj.frames[0] = last_frame
        
    for y in range(0,disp_per_block):
        for z in range(0,disp_interval):
            solver_obj.propagate_sim()
                    
        last_frame = solver_obj.get_last_frame()
        plot_Emag(last_frame, params_obj)
    

#plot particle trajectory
particle_posx = solver_obj.particles[0].pos_x
particle_posz = solver_obj.particles[0].pos_z
particle_vx = solver_obj.particles[0].vx
particle_vz = solver_obj.particles[0].vz
particle_v = [(particle_vx[i]**2 + particle_vz[i]**2)**0.5 for i in range(0, len(particle_posx))]
particle_r = [((particle_posx[i]-0.5*dx*Nx)**2 + (particle_posz[i]-0.5*dz*Nz)**2)**0.5 for i in range(0, len(particle_posx))]
particle_time = [x*dt for x in range(0, len(particle_posx))]

plt.plot(particle_time, [1e3 * r for r in particle_r])
plt.ylabel('Particle radial position (mm)')
plt.xlabel('Time (s)')
plt.show()

plt.plot(particle_time, [v for v in particle_v])
plt.ylabel('Particle velocity (m/s)')
plt.xlabel('Time (s)')
plt.show()

plt.plot([1e3 * x for x in particle_posx], [1e3 * z for z in particle_posz])
plt.xlabel('Particle x position (mm)')
plt.ylabel('Particle z position (mm)')
plt.xlim(0, 1e3 * Nx*dx)
plt.ylim(0, 1e3 * Nz*dz)
plt.show()

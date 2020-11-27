import numpy as np
import matplotlib.pyplot as plt
import pic_solver
import os

#Define a function for plotting the eletric field magnitude
def plot_Emag(fields, ofs_field, params):
    frameno = fields.frameno
    E_mag = np.power((np.power(fields.Ex - ofs_field.Ex, 2) + np.power(fields.Ez - ofs_field.Ez, 2)), 0.5)
    
    plot_y, plot_x = np.mgrid[0:params.Nx+1, 0:params.Nz+1]
    plt.pcolormesh(plot_x, plot_y, E_mag, cmap='hot', shading='flat', vmin=0, vmax=1.5e-3)
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
    
#Define parameters of the solution domain
Nx = 200  #Number of cells in x direction
Nz = 100  #Number of cells in z direction
N_rho = 5 #Number of cells to spread charge over
dx = 1e-3 #Cell size in x direction
dy = 1e-3 #Cell size in z direction
dz = 1e-3 #Domain thickness in y direction
dt = 1e-12 #Time step

#Create a parameters object to store parameter information
params_obj = pic_solver.params(Nx, Nz, N_rho, dx, dy, dz, dt)

#Define the material properties of the domain
eps_r = np.ones((Nx, Nz), dtype=float)  #Relative permittivity = 1
sigma = np.zeros((Nx, Nz), dtype=float) #Conductivity = 0
w_p = np.zeros((Nx, Nz), dtype=float)   #Drude plasma frequency = 0
f_c = np.zeros((Nx, Nz), dtype=float)   #Drude collision frequency = 0

#Create a block of dielectric in the domain:
eps_r[50:70, :] = 5
eps_r[130:150, :] = 5

#Create a geometry object which stores materials properties and boundary conditions
#In this case, the x-boundaries are PEC, and the z-boundaries are PMC
geometry_obj = pic_solver.geometry('pec', 'pec', 'pmc', 'pmc', eps_r, sigma, w_p, f_c)

#Create a constants object to contain  relevant physical constants
constants_obj = pic_solver.constants()

#Create the FDTD solver object and pass it all the parameters we've defined
solver_obj = pic_solver.solver(params_obj, constants_obj, geometry_obj)

#Set up a uniform electric field in the domain in the first frame:
solver_obj.frames[0].Ex[:,:] = -1e6

#Create a particle object, defining its initial position velocity, charge and mass:
particle_1 = pic_solver.particle(1e-3, 50e-3, 0, 0, 1e-3, -constants_obj.q_e, constants_obj.m_e)

#Add this particle to the list of particles in the simulation
solver_obj.particles.append(particle_1)

#Define how many time steps we want to run the simulation for:
disp_interval = 8#16   #Display one frame in this many
disp_per_block = 300#100 #Run simulation for this many frames
        
for x in range(0,disp_per_block):
    for y in range(0,disp_interval):
        #This function propagates the simulation by one frame
        solver_obj.propagate_sim()
        
    #Get the last frame and plot it
    last_frame = solver_obj.get_last_frame()
    plot_Emag(last_frame, solver_obj.frames[0], params_obj)
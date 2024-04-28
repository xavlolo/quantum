#-----------------------------------------------------------------#
#------------------------Imports-------------------------------#
#-----------------------------------------------------------------#

import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

#-----------------------------------------------------------------#
#------------------------Parameters----------------------------------#
#-----------------------------------------------------------------#

# The computational basis: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩.
r0 = 0 #initial r value 
tau = 10 # synaptic depression
dt = 0.1 # "dt" is the time step used for numerical integration
t_max = 700 #max time simulation
tpoints = np.arange(0, t_max + dt, dt)
U_values = np.array([0.5])  # Different U values to test.
# Define the diagonal values for the Hamiltonian 
diagonal_sets = [
  [-3, -1,-1, 1, -1, 1, 1, 3],# case homogenous energy (computed manually)
]

results = {} # Initialize an empty dictionary to store results from simulation

#-----------------------------------------------------------------#
#------------------------Functions----------------------------------#
#-----------------------------------------------------------------#

########## initialization of rho: excitaion first qubit on rho_44 |100⟩#######
for U in U_values:
    rho0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
        
################# evolution of rho ###################### 

    def rho_derivative(H, rho):
        return -1j * np.dot(H, rho) + 1j * np.dot(rho, H)
    
################# Runge-Kutta ######################    

    def runge_kutta_4(r_prev, rho_prev, H_prev, H_func, dt, S):
        # 1st evaluation
        k1_rho = dt * rho_derivative(H_prev, rho_prev)
        k1_r = dt * ((1 - r_prev) / tau - U * r_prev * S)
    
        # 2nd evaluation
        r_mid1 = r_prev + k1_r / 2
        H_mid1 = H_func(r_mid1)
        rho_mid1 = rho_prev + k1_rho / 2
        
        k2_rho = dt * rho_derivative(H_mid1, rho_mid1)
        k2_r = dt * ((1 - r_mid1) / tau - U * r_mid1 * S)
    
        # 3rd evaluation
        r_mid2 = r_prev + k2_r / 2
        H_mid2 = H_func(r_mid2)
        rho_mid2 = rho_prev + k2_rho / 2
        
        k3_rho = dt * rho_derivative(H_mid2, rho_mid2)
        k3_r = dt * ((1 - r_mid2) / tau - U * r_mid2 * S)
    
        # 4th evaluation
        r_next = r_prev + k3_r
        H_next = H_func(r_next)  # Calculate H_next
        rho_next = rho_prev + k3_rho  # Calculate rho_next
        
        k4_rho = dt * rho_derivative(H_next, rho_next)
        k4_r = dt * ((1 - r_next) / tau - U * r_next * S)
    
        # Combine evaluations to estimate the next value of r and rho
        r_new = r_prev + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
        rho_new = rho_prev + (k1_rho + 2 * k2_rho + 2 * k3_rho + k4_rho) / 6


        return r_new, rho_new, H_next

#######################  define Hamiltonian #########
    def H_func(r):
        return np.array([
            [-3, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0.05*r, 0, 0, 0, 0, 0],
            [0, 0.05*r, -1, 0, 0.05*r, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0.05*r, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 3]], dtype=float)

    
################## Iterate over the time range, updating the density matrix and calculating its elements at each step
    def integrate_equation(r0, tau, U, dt, t_max): # parameters are used to perform a numerical integration using the Runge-Kutta method
        t = np.arange(0, t_max+dt, dt)
        r = np.zeros(len(t))
        rho = np.zeros((len(t), 8, 8), dtype=complex)
        H = np.zeros((len(t), 8, 8), dtype=complex)
        r[0] = r0 # Set initial value for r
        rho[0] = rho0
        H[0] = H_func(r0)
  
       
        # Create an array to store the x values
        x_values = np.zeros_like(tpoints)
        t = np.arange(0, t_max+dt, dt)      
        rho00 = np.zeros_like(tpoints, dtype=complex)
        rho11 = np.zeros_like(tpoints, dtype=float) # Qubit 3
        rho22 = np.zeros_like(tpoints, dtype=float) # Qubit 2
        rho33 = np.zeros_like(tpoints, dtype=float)
        rho44 = np.zeros_like(tpoints, dtype=float) # Qubit 1 |100⟩
        rho55 = np.zeros_like(tpoints, dtype=float)
        rho55 = np.zeros_like(tpoints, dtype=float)
        rho66 = np.zeros_like(tpoints, dtype=float)
        rho77 = np.zeros_like(tpoints, dtype=float)


        for i in range(1, len(t)):
            # Generate random x between 0 and 1
            x = random.random()
            x_values[i] = x 
            # Set S according to the value of x
            ##### Measurement #####
            if x < rho22[i-1]:# change this depending where we measure,  S is set to 1 if x is less than the previous value of rho44[i-1], indicating a condition or threshold is met
                S = 1
            elif x > rho22[i-1]:
                S = 0
                
            ##########updates the state of r, rhos, and H to their new values#################
            r[i], rho[i], H[i]= runge_kutta_4(r[i - 1], rho[i - 1], H[i - 1], H_func, dt, S) ###previous timestep [i−1]
       
            
            ####Normalize the density matrix#####
            normalization_factor = np.sum(np.real(np.diag(rho[i]))) # Calculate the normalization factor as the sum of the real parts of the diagonal elements of rho[i]
            rho[i] /= normalization_factor # Normalize the density matrix rho[i] by dividing it by the normalization factor
            rho00[i] = np.real(rho[i, 0, 0]) # Extract and store the real part of the (0, 0) element of rho[i] in rho00[i]
            rho11[i] = np.real(rho[i, 1, 1])
            rho22[i] = np.real(rho[i, 2, 2])
            rho33[i] = np.real(rho[i, 3, 3])
            rho44[i] = np.real(rho[i, 4, 4])
            rho55[i] = np.real(rho[i, 5, 5])
            rho66[i] = np.real(rho[i, 6, 6])
            rho77[i] = np.real(rho[i, 7, 7])


        return t, r, rho11, rho22, rho33, rho00, rho44, rho55, rho66, rho77
    t, r, rho11, rho22, rho33, rho00, rho44, rho55, rho66, rho77= integrate_equation(r0, tau, U, dt, t_max) # The function returns the calculated arrays, which are then assigned to respective variables to hold the entire time series data for r and the probabilities.
    
    results[U] = (t, r, rho11, rho22, rho33, rho00, rho44, rho55, rho66, rho77) #store the results of the simulation in a dictionary results where the key is U, useful for multiple U values




#--------------------------------------------------------------#
#------------------------Plots----------------------------------#
#--------------------------------------------------------------#

######simple plot for all qubits for the first U in array######

# Create a single figure with specified dimensions
fig, ax1 = plt.subplots(figsize=(26, 6))

# Plotting data
plt.plot(tpoints, rho11, color='violet', label='Q3')
plt.plot(tpoints, rho22, color='orange',label='Q2')
plt.plot(tpoints, rho44, color='green',label='Q1')
plt.plot(t, r, label='r', color='blue')


ax1.set_xlim([0, max(tpoints)])
ax1.set_ylim([-0.1, 1.1])
ax1.xaxis.label.set_size(40)
ax1.yaxis.label.set_size(40)
ax1.tick_params(axis='both', which='major', labelsize=40, length=10, width=2)
ax1.set_yticks([0, 0.5, 1])
ax1.set_yticklabels(['0', '0.5', '1'], fontsize=40)


plt.legend(loc='upper right', fontsize=30)
plt.xlabel(' ', fontsize=46)
plt.ylabel('$P_E$', fontsize=46)

plt.show()

######example of multiple plot for different U for Q1######

# fig, ax1 = plt.subplots(figsize=(26, 6))
# for U in results.keys():
#     time_points, variable_r, rho11, rho22, rho33, rho00, rho44, rho55, rho66, rho77 = results[U]
#     plt.plot(time_points, rho44, label=f'U-Q1= {U}')
    
# ax1.set_xlim([0, max(tpoints)])
# ax1.set_ylim([-0.1, 1.1])
# ax1.xaxis.label.set_size(40)
# ax1.yaxis.label.set_size(40)
# ax1.tick_params(axis='both', which='major', labelsize=40, length=10, width=2)
# ax1.set_yticks([0, 0.5, 1])
# ax1.set_yticklabels(['0', '0.5', '1'], fontsize=40)

# plt.xlabel('Time')
# plt.ylabel('$P_E$', fontsize=46)
# plt.legend(loc='upper right', fontsize=30)
# plt.xlabel(' ', fontsize=46)
# plt.grid(False)
# plt.show()



#--------------------------------------------------------------#
#------------------------Export Data---------------------------#
#--------------------------------------------------------------#
# data = {
#     'Time': tpoints,
#     'Q1': rho44,
#     'Q2': rho22,
#     'Q3': rho11
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Export to a TXT file
# txt_file_path = 'output_data.txt'
# df.to_csv(txt_file_path, sep='\t', index=False)

# print(f'Data exported successfully to {txt_file_path}')

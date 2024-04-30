
# Code for 3 qubits
# Homogeneous case
# Excitation on Q1
# Occupation probability
# Depression only
# Entanglement (single site entanglement and entanglement between qubits 1-3) 
# Data export single file for each qubit#
# Time for 2.5 period for different tau (0.01=222.43, 10=512.14, 100=3028.99, 500=14163.89)

#-----------------------------------------------------------------#
#------------------------Imports----------------------------------#
#-----------------------------------------------------------------#

import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

#-----------------------------------------------------------------#
#------------------------Parameters-------------------------------#
#-----------------------------------------------------------------#

# The computational basis: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩.
r0 = 0 #initial r value 
tau = 0.01# synaptic depression
dt = 0.01 # "dt" is the time step used for numerical integration
t_max = 222 #max time simulation
U_values = [0.5]  # Different U values to test.
# Define the diagonal values for the Hamiltonian 
diagonal_H = [-3, -1,-1, 1, -1, 1, 1, 3]# case homogenous (computed manually)

#-----------------------------------------------------------------#
#------------------------Dictionary-------------------------------#
#-----------------------------------------------------------------#

# Initialize an empty dictionary to store results from simulation
results = {}
all_negativity_1 = {} 
all_negativity_2 = {} 
all_negativity_3 = {}
all_negativity_4 = {} 

#-----------------------------------------------------------------#
#------------------------Functions--------------------------------#
#-----------------------------------------------------------------#
     
################# evolution of rho ###################### 

def rho_derivative(H, rho):
    return -1j * np.dot(H, rho) + 1j * np.dot(rho, H)
    
################# Runge-Kutta 4th order ######################    

def runge_kutta_4(r_prev, rho_prev, H_prev, H_func, dt, tau, U, S):
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

####################### Hamiltonian ##################
def H_func(r):
        return np.array([
            [diagonal_H[0], 0, 0, 0, 0, 0, 0, 0],
            [0, diagonal_H[1], 0.05*r, 0, 0, 0, 0, 0],
            [0, 0.05*r, diagonal_H[2], 0, 0.05*r, 0, 0, 0],
            [0, 0, 0, diagonal_H[3], 0, 0, 0, 0],
            [0, 0, 0.05*r, 0, diagonal_H[4], 0, 0, 0],
            [0, 0, 0, 0, 0, diagonal_H[5], 0, 0],
            [0, 0, 0, 0, 0, 0,diagonal_H[6], 0],
            [0, 0, 0, 0, 0, 0, 0, diagonal_H[7]]], dtype=float)
    
    
################## Iterate over the time range, updating the density matrix and calculating its elements at each step
def integrate_equation(r0, tau, U, dt, t_max): # parameters are used to perform a numerical integration using the Runge-Kutta method
        tpoints = np.arange(0, t_max, dt)
        r = np.zeros(len(tpoints))
        H = np.zeros((len(tpoints), 8, 8), dtype=complex)
        rho = np.zeros((len(tpoints), 8, 8), dtype=complex)
        r[0] = r0
        rho[0] = np.zeros((8, 8), dtype=float)
        rho[0][4, 4] = 1  #initialization of rho: excitaion first qubit on rho_44 |100⟩##
        H[0] = H_func(r0)

# define the rhos name
        rho00 = np.zeros_like(tpoints, dtype=complex)
        rho11 = np.zeros_like(tpoints, dtype=float) # Qubit 3 |001⟩
        rho22 = np.zeros_like(tpoints, dtype=float) # Qubit 2 |010⟩
        rho33 = np.zeros_like(tpoints, dtype=float)
        rho44 = np.zeros_like(tpoints, dtype=float) # Qubit 1 |100⟩
        rho55 = np.zeros_like(tpoints, dtype=float)
        rho66 = np.zeros_like(tpoints, dtype=float)
        rho77 = np.zeros_like(tpoints, dtype=float)

#initial negativity
        negativity_1_list = [0]  # Starting with zero
        negativity_3_list = [0]  # Starting with zero
        negativity_2_list = [0]  # Starting with zero
        negativity_4_list = [0]  # Starting with zero

        for i in range(1, len(tpoints)):
        # Generate random x and set S according to the value of x
            x = random.random() 
            S = 1 if x < np.real(rho[i-1, 2, 2]) else 0   # Measure on Q2 change this for Q3 (rho[i-1, 1, 1]) or Q1 (rho[i-1, 4, 4]),  S is set to 1 if x is less than the previous value of rho22[i-1], indicating a condition or threshold is met
                
            ##########updates the state of r, rhos, and H to their new values#################
            r[i], rho[i], H[i] = runge_kutta_4(r[i - 1], rho[i - 1], H[i - 1], H_func, dt, tau, U, S)       
            
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
                 
#-----------------------------------------------------------------#
#------------------------Entanglement-----------------------------#
#-----------------------------------------------------------------#

##### Compute the partial transpose of the density matrix with respect to system 1|23 #####
# Create a copy of the i-th 8x8 density matrix from the list rho
            rho_1 = rho[i].copy()
            # Modify rho_A as needed (following Irene documentation swapping things same for other (manual))
            rho_1 = np.block([
                [rho_1[:4, :4], rho_1[4:, :4]],
                [rho_1[:4, 4:], rho_1[4:, 4:]]
            ])
            # Compute the eigenvalues of the partial transpose and sum the absolute values of the negative ones
            eigenvalues_1 = np.linalg.eigvals(rho_1)
            negativity_1 = 0.5 * (sum(abs(eig) for eig in eigenvalues_1) -1)  # equation (12) of the paper for negativity, results checked with Elliot
            negativity_1_list.append(negativity_1)
            
##### Compute the partial transpose of the density matrix with respect to system 12|3 #####
   
            rho_3 = rho[i].copy() #creates a copy of the i-th 8x8 density matrix from the list rho. This copy is made to avoid modifying the original matrix.
            swaps = [(2, 9), (4, 11), (6, 13), (8, 15),
                          (18, 25), (20, 27), (22, 29), (24, 31),
                          (34, 41), (36, 43), (38, 45), (40, 47),
                          (50, 57), (52, 59), (54, 61), (56, 63)]
                # Perform the swaps according to the predefined list
            for swap in swaps:  #for loop iterates over each pair of indices in the swaps
                    i_s, j_s = swap  #The two indices of the pair are unpacked into i_s and j_s.
                    i_index = np.unravel_index(i_s-1, rho_3.shape) #The np.unravel_index function is used to convert a flat index into a coordinate in the 8x8 matrix. This is done because the swap indices are provided as if the 8x8 matrix was flattened into a 64-element array. The -1 is needed because Python uses 0-based indexing.
                    j_index = np.unravel_index(j_s-1, rho_3.shape)
                    rho_3[i_index], rho_3[j_index] = rho_3[j_index], rho_3[i_index] #This line performs the swap operation in the density matrix.
                
                # Compute the eigenvalues of the partial transpose and sum the absolute values of the negative ones
            eigenvalues_3 = np.linalg.eigvals(rho_3)
                # If an eigenvalue is positive, it doesn't contribute to the negativity.
            negativity_3 = 0.5 * (sum(abs(eig) for eig in eigenvalues_3) -1)
            negativity_3_list.append(negativity_3)


##### Compute the partial transpose of the density matrix with respect to system 13|2 #####
            rho_2 = rho[i].copy()
            rho_2_top_left1 = rho[i][2:4, 0:2] # rho[i][2:4, 0:2]: This selects the 3rd and 4th row, and the 1st and 2nd column of the 8x8 matrix rho[i]. 
            rho_2_top_right1 = rho[i][0:2, 2:4]
            rho_2_bottom_left1 = rho[i][6:8, 0:2]
            rho_2_bottom_right1 = rho[i][4:6, 2:4]
            rho_2_top_left2 = rho[i][2:4, 4:6]
            rho_2_top_right2 = rho[i][0:2, 6:8]
            rho_2_bottom_left2 = rho[i][6:8, 4:6]
            rho_2_bottom_right2 = rho[i][4:6, 6:8]
                # construct new 8x8 matrix
            rho_2 = np.block([
                    [rho[i][0:2, 0:2], rho_2_top_left1, rho[i][0:2, 4:6], rho_2_top_left2],
                    [rho_2_top_right1, rho[i][2:4, 2:4], rho_2_top_right2, rho[i][2:4, 6:8]],
                    [rho[i][4:6, 0:2], rho_2_bottom_left1, rho[i][4:6, 4:6], rho_2_bottom_left2],
                    [rho_2_bottom_right1, rho[i][6:8, 2:4], rho_2_bottom_right2, rho[i][6:8, 6:8]]
                ])
                # print("Shape of rho_2 = ", rho_2.shape)
         
                # Compute the eigenvalues of the partial transpose and sum the absolute values of the negative ones
            eigenvalues_2 = np.linalg.eigvals(rho_2)
                # If an eigenvalue is positive, it doesn't contribute to the negativity.
            negativity_2 = 0.5 * (sum(abs(eig) for eig in eigenvalues_2) -1)
            negativity_2_list.append(negativity_2)
            
                      
##### Compute the partial transpose of the density matrix trace q2 #####
# Create a copy of the i-th 8x8 density matrix from the list rho
            rho_4 = rho[i].copy()
            
            # Initialize the reduced density matrix
            rho_reduced = np.zeros((4, 4), dtype=complex)
            
            # Fill in the reduced density matrix according to the rules
            rho_reduced[0, 0] = rho_4[0, 0] + rho_4[2, 2]
            rho_reduced[0, 1] = rho_4[0, 1] + rho_4[2, 3]
            rho_reduced[1, 0] = rho_4[1, 0] + rho_4[3, 2]
            rho_reduced[1, 1] = rho_4[1, 1] + rho_4[3, 3]
            
            rho_reduced[0, 2] = rho_4[0, 4] + rho_4[2, 6]
            rho_reduced[0, 3] = rho_4[0, 5] + rho_4[2, 7]
            rho_reduced[1, 2] = rho_4[1, 4] + rho_4[3, 6]
            rho_reduced[1, 3] = rho_4[1, 5] + rho_4[3, 7]
            
            rho_reduced[2, 0] = rho_4[4, 0] + rho_4[6, 2]
            rho_reduced[2, 1] = rho_4[4, 1] + rho_4[6, 3]
            rho_reduced[3, 0] = rho_4[5, 0] + rho_4[7, 2]
            rho_reduced[3, 1] = rho_1[5, 1] + rho_4[7, 3]
            
            rho_reduced[2, 2] = rho_4[4, 4] + rho_4[6, 6]
            rho_reduced[2, 3] = rho_4[4, 5] + rho_4[6, 7]
            rho_reduced[3, 2] = rho_4[5, 4] + rho_4[7, 6]
            rho_reduced[3, 3] = rho_4[5, 5] + rho_4[7, 7]
                        
            # Compute the partial transpose of the reduced density matrix with respect to subsystem q1
            rho_PT = rho_reduced.reshape(2, 2, 2, 2).transpose(2, 1, 0, 3).reshape(4, 4)           
            
            # Compute the eigenvalues of the partial transpose and sum the absolute values of the negative ones
            eigenvalues_4 = np.linalg.eigvals(rho_PT)
            negativity_4 = 0.5 * (sum(abs(eig) for eig in eigenvalues_4) -1)
            negativity_4_list.append(negativity_4)

        return tpoints, negativity_1_list, negativity_3_list, negativity_2_list, negativity_4_list, r, rho11, rho22, rho33, rho00, rho44, rho55, rho66, rho77
           
#-----------------------------------------------------------------#
#------------------------Loop-------------------------------------#
#-----------------------------------------------------------------#

for U in U_values:
    tpoints, negativity_1,negativity_3,negativity_2, negativity_4, r, rho11, rho22, rho33, rho00, rho44, rho55, rho66, rho77 = integrate_equation(r0, tau, U, dt, t_max)
    all_negativity_1[U] = negativity_1
    all_negativity_3[U] = negativity_3
    all_negativity_2[U] = negativity_2
    all_negativity_4[U] = negativity_4
    print(f"Simulation complete for U={U}")
    
#-----------------------------------------------------------------#
#------------------------Plots------------------------------------#
#-----------------------------------------------------------------#

###############  Plotting the results Neg 1###########################
fig, ax1 = plt.subplots(figsize=(26, 6))
for U, negativity in all_negativity_1.items():
    ax1.plot(tpoints, negativity, label=f'Negativity 1|23 for U={U}',color='green')
    
    
# Set the x-axis limits to start at 0
ax1.set_xlim([0, max(tpoints)])
ax1.xaxis.label.set_size(46)  # Set font size for x-axis label
ax1.yaxis.label.set_size(46)  # Set font size for y-axis label of ax1

# Manually setting specific x-axis ticks and labels
specific_ticks = [0, 50, 100, 150, 200]  # Example tick positions
specific_labels = ['0', '50', '100','150', '200']  # Leave the last label empty to "remove" it
ax1.set_xticks(specific_ticks)
ax1.set_xticklabels(specific_labels, fontsize=40)


ax1.tick_params(axis='both', which='major', labelsize=40, length=10, width=2)  # Adjusted tick mark size
# Set y-axis tick values for ax1
ax1.set_yticks([0, 0.5])
# Set y-axis tick labels for ax1 and ax2 with potentially increased font size
ax1.set_yticklabels(['0', '0.5'], fontsize=40)


ax1.set_xlabel("Time")
ax1.set_ylabel("$N_{ss}^{Q_1}$")
ax1.legend(loc='upper right', fontsize=30)
ax1.set_xlim([0, t_max])
plt.show()


###############  Plotting the results Neg 3 ###########################
fig, ax2 = plt.subplots(figsize=(26, 6))
for U, negativity in all_negativity_3.items():
    ax2.plot(tpoints, negativity, label=f'Negativity 12|3 for U={U}', color='violet')
    
    
# Set the x-axis limits to start at 0
ax2.set_xlim([0, max(tpoints)])
ax2.xaxis.label.set_size(46)  # Set font size for x-axis label
ax2.yaxis.label.set_size(46)  # Set font size for y-axis label of ax1

# Manually setting specific x-axis ticks and labels
specific_ticks = [0, 50, 100, 150, 200]  # Example tick positions
specific_labels = ['0', '50', '100','150', '200']  # Leave the last label empty to "remove" it
ax2.set_xticks(specific_ticks)
ax2.set_xticklabels(specific_labels, fontsize=40)


ax2.tick_params(axis='both', which='major', labelsize=40, length=10, width=2)  # Adjusted tick mark size
# Set y-axis tick values for ax1
ax2.set_yticks([0, 0.5])
# Set y-axis tick labels for ax1 and ax2 with potentially increased font size
ax2.set_yticklabels(['0', '0.5'], fontsize=40)


ax2.set_xlabel("Time")
ax2.set_ylabel("$N_{ss}^{Q_3}$")
ax2.legend(loc='upper right', fontsize=30)
ax2.set_xlim([0, t_max])
plt.show()

###############  Plotting the results Neg 2 ###########################
fig, ax3 = plt.subplots(figsize=(26, 6))
for U, negativity in all_negativity_2.items():
    ax3.plot(tpoints, negativity, label=f'Negativity 13|2 for U={U}', color='orange')
    
    
# Set the x-axis limits to start at 0
ax3.set_xlim([0, max(tpoints)])
ax3.xaxis.label.set_size(46)  # Set font size for x-axis label
ax3.yaxis.label.set_size(46)  # Set font size for y-axis label of ax1

# Manually setting specific x-axis ticks and labels
specific_ticks = [0, 50, 100, 150, 200]  # Example tick positions
specific_labels = ['0', '50', '100','150', '200']  # Leave the last label empty to "remove" it
ax3.set_xticks(specific_ticks)
ax3.set_xticklabels(specific_labels, fontsize=40)


ax3.tick_params(axis='both', which='major', labelsize=40, length=10, width=2)  # Adjusted tick mark size
# Set y-axis tick values for ax1
ax3.set_yticks([0, 0.5])
# Set y-axis tick labels for ax1 and ax2 with potentially increased font size
ax3.set_yticklabels(['0', '0.5'], fontsize=40)


ax3.set_xlabel("Time")
ax3.set_ylabel("$N_{ss}^{Q_2}$")
ax3.legend(loc='upper right', fontsize=30)
ax3.set_xlim([0, t_max])
plt.show()

###############  Plotting the results Neg 4 Trace Q2 ###########################
fig, ax4 = plt.subplots(figsize=(26, 6))
for U, negativity in all_negativity_4.items():
    ax4.plot(tpoints, negativity, label=f'Negativity Trace Q2 for U={U}', color='black')
    
    
# Set the x-axis limits to start at 0
ax4.set_xlim([0, max(tpoints)])
ax4.xaxis.label.set_size(46)  # Set font size for x-axis label
ax4.yaxis.label.set_size(46)  # Set font size for y-axis label of ax1

# Manually setting specific x-axis ticks and labels
specific_ticks = [0, 50, 100, 150, 200]  # Example tick positions
specific_labels = ['0', '50', '100','150', '200']  # Leave the last label empty to "remove" it
ax4.set_xticks(specific_ticks)
ax4.set_xticklabels(specific_labels, fontsize=40)


ax4.tick_params(axis='both', which='major', labelsize=40, length=10, width=2)  # Adjusted tick mark size
# Set y-axis tick values for ax1
ax4.set_yticks([0, 0.5])
# Set y-axis tick labels for ax1 and ax2 with potentially increased font size
ax4.set_yticklabels(['0', '0.5'], fontsize=40)


ax4.set_xlabel("Time")
ax4.set_ylabel("$N_{ss}^{Q_1Q3}$")
ax4.legend(loc='upper right', fontsize=30)
ax4.set_xlim([0, t_max])
plt.show()

#######simple plot for all qubits for the first U in array ######

# Create a single figure with specified dimensions
fig, ax1 = plt.subplots(figsize=(26, 6))

# Plotting data
plt.plot(tpoints, rho11, color='violet', label= f'Q3-U={U}')
plt.plot(tpoints, rho22, color='orange',label='Q2')
plt.plot(tpoints, rho44, color='green',label='Q1')
plt.plot(tpoints, r, label='r', color='blue')


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

# ######example of multiple plot for different U for Q1######

# fig, ax2 = plt.subplots(figsize=(26, 6))
# for U in results.keys():
#     time_points, variable_r, rho11, rho22, rho33, rho00, rho44, rho55, rho66, rho77 = results[U]
#     plt.plot(time_points, rho44, label=f'U-Q1= {U}')
    
# ax2.set_xlim([0, max(tpoints)])
# ax2.set_ylim([-0.1, 1.1])
# ax2.xaxis.label.set_size(40)
# ax2.yaxis.label.set_size(40)
# ax2.tick_params(axis='both', which='major', labelsize=40, length=10, width=2)
# ax2.set_yticks([0, 0.5, 1])
# ax2.set_yticklabels(['0', '0.5', '1'], fontsize=40)

# plt.xlabel('Time')
# plt.ylabel('$P_E$', fontsize=46)
# plt.legend(loc='upper right', fontsize=30)
# plt.xlabel(' ', fontsize=46)
# plt.grid(False)
# plt.show()

#--------------------------------------------------------------#
#------------------------Export Data---------------------------#
#--------------------------------------------------------------#

# Data for probabilities separate file
data = {
    'Time': tpoints,
    'Q1': rho44,
    'Q2': rho22,
    'Q3': rho11
}

# Create a DataFrame
df = pd.DataFrame(data)


# Export each series to a separate TXT file with the 'Time' column
for column in df.columns[1:]:  # Skip 'Time' column on this iteration
    # Filename for each quantity
    txt_file_path = f'Occupation_Probabilities_{column}.txt'
    # Select the 'Time' column and the current quantity column
    df_subset = df[['Time', column]]
    # Export to TXT file
    df_subset.to_csv(txt_file_path, sep='\t', index=False)

print("Data for each rho saved successfully in separate files.")


# Data for negativity for each plot to save separately
datasets = {
    'negativity_Q1.txt': all_negativity_1,
    'negativity_Q2.txt': all_negativity_2,
    'negativity_Q3.txt': all_negativity_3,
    'negativity_Q4.txt': all_negativity_4,
}

# Directly write each dataset to a separate file
for filename, negativity_data in datasets.items():
    with open(filename, 'w') as f:
        # Write the header
        f.write('Time Qbit\n')
        # Write the data
        for i in range(len(tpoints[1:])):  # We start from 1 because plots start from tpoints[1:]
            line = f"{tpoints[i+1]} "
            for value in negativity_data.values():
                line += f"{value[i]} "
            f.write(line.strip() + '\n')

print("Data saved.")

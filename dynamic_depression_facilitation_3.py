import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd


# Depression and faciliation with also temperature protocol and data export

# Define the initial conditions with different paramaters
r0 = 0
tau = 0.01#synaptic depression 
tau_facil=10# synaptic facilatation longer time according to paper (mongillo)
dt = 0.01 # "dt" is the time step used for numerical integration
t_max = 222
tpoints = np.arange(0, t_max + dt, dt)
beta_values = [5]  # list of beta values (beta=5 no temperature)
U_values = np.array([0.5])  # Different U values to test

# Define the energy levels eigenvalues homogenous case, computed manually  
eigen= np.array([-1, 1 ,1, 3, -3, -1 ,-1, 1])


# Define the diagonal elements for the hamiltonian with 3 qubits
diagonal_sets = [
  [-3, -1,-1, 1, -1, 1, 1, 3],# case homogenous
]

results = {} # Initialize an empty dictionary to store results from simulations

################# evolution of rho ###################### 

def rho_derivative(H, rho):
    return -1j * np.dot(H, rho) + 1j * np.dot(rho, H)
    
################# Runge-Kutta for r and u ######################    
def runge_kutta_4(r_prev, u_prev, rho_prev, H_prev, H_func, dt, S,  tau, tau_facil):        
        k1_rho = dt * rho_derivative(H_prev, rho_prev) # Derivative of rho at initial point
        k1_r = dt * ((1 - r_prev) / tau - u_prev * r_prev * S)  # Derivative of r at initial point
        k1_u = dt * ((U-u_prev)/ tau_facil + U * (1 - u_prev) * S)  # Derivative of u at initial point
        
        r_next = r_prev + 0.5 * k1_r  # Update r using its derivative
        u_next = u_prev + 0.5 * k1_u  # Update u using its derivative
        H_next = H_func(r_next,u_next)  # Calculate H at the updated r
       
        k2_rho = dt * rho_derivative(H_next, rho_prev + 0.5 * k1_rho)
        k2_r = dt * ((1 - r_next) / tau - (u_next + 0.5 * k1_u) * r_next * S)
        k2_u = dt * ((U - u_next) / tau_facil + U * (1 - u_next) * S)
        
        r_next = r_prev + 0.5 * k2_r
        u_next = u_prev + 0.5 * k2_u  # Update u using its derivative
        H_next = H_func(r_next,u_next)  # Calculate H at the updated r
       
        k3_rho = dt * rho_derivative(H_next, rho_prev + 0.5 * k2_rho)
        k3_r = dt * ((1 - r_next) / tau - (u_next+ 0.5 * k2_u) * r_next * S)
        k3_u = dt * ((U - u_next) / tau_facil + U * (1 - u_next) * S)
       
        r_next = r_prev + k3_r
        u_next = u_prev + k3_u  # Update u
        H_next = H_func(r_next, u_next)  # Calculate H at the updated r
       
        k4_rho = dt * rho_derivative(H_next, rho_prev + k3_rho)
        k4_r = dt * ((1 - r_next) / tau - (u_next + k3_u) * r_next * S)
        k4_u = dt * ((U - u_next) / tau_facil + U * (1 - u_next) * S)      
       
        rho_new = rho_prev + (k1_rho + 2*k2_rho + 2*k3_rho + k4_rho) / 6
        r_new = r_prev + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
        u_new = u_prev + (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6
       
        return r_new, u_new, rho_new, H_next

################# Hamiltonian ###################### We only use r in the Hamiltonian but if add u we have a shift)
def H_func(r,u):
    return np.array([
            [diagonal_values[0], 0, 0, 0, 0, 0, 0, 0],
            [0, diagonal_values[1], 0.05*(r), 0, 0, 0, 0, 0],
            [0, 0.05*(r), diagonal_values[2], 0, 0.05*(r), 0, 0, 0],
            [0, 0, 0, diagonal_values[3], 0, 0, 0, 0],
            [0, 0, 0.05*(r), 0, diagonal_values[4], 0, 0, 0],
            [0, 0, 0, 0, 0, diagonal_values[5], 0, 0],
            [0, 0, 0, 0, 0, 0, diagonal_values[6], 0],
            [0, 0, 0, 0, 0, 0, 0, diagonal_values[7]]], dtype=complex)

################# initial rho0 with no temp#################
        # Removing the beta and temperature dependency from the initialization of rho0
rho0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]], dtype=complex)
            
            
################# initial rho0 with temp#################
for U in U_values:
    for beta in beta_values:
        Z = np.sum(np.exp(-eigen * beta))
        for diagonal_values in diagonal_sets:
            rho0 = np.array(np.multiply((1/Z), [[np.exp(-beta * eigen[0]), 0, 0, 0, 0, 0, 0, 0], 
                                        [0, np.exp(-beta * eigen[1]), 0, 0, 0, 0, 0, 0],
                                        [0, 0, np.exp(-beta * eigen[2]), 0, 0, 0, 0, 0], 
                                        [0, 0, 0, np.exp(-beta * eigen[3]), 0, 0, 0, 0],
                                        [0, 0, 0, 0, np.exp(-beta * eigen[4]), 0, 0, 0],
                                        [0, 0, 0, 0, 0, np.exp(-beta * eigen[5]), 0, 0],
                                        [0, 0, 0, 0, 0, 0, np.exp(-beta * eigen[6]), 0],
                                        [0, 0, 0, 0, 0, 0, 0, np.exp(-beta * eigen[7])]], dtype=complex))            

    
# Iterate over the time range, updating the density matrix and calculating its elements at each step
    def integrate_equation(r0,  U, dt, t_max, beta, tau, tau_facil):
        t = np.arange(0, t_max+dt, dt)
        r = np.zeros(len(t))
        rho = np.zeros((len(t), 8, 8), dtype=complex)
        H = np.zeros((len(t), 8, 8), dtype=complex)
        u = np.zeros(len(t))  # Define u
        u[0] = U  # Set initial value for u
        r[0] = r0 # Set initial value for r
        rho[0] = rho0
        H[0] = H_func(r0, U) # is initializing the Hamiltonian matrix H at time index 0 (t=0) using the initial value of r, which is r0. The H_func function is called with the initial value of r (r0) as its argument, and the resulting Hamiltonian matrix is assigned to the first element of the H array (i.e., H[0]). The array H will store the Hamiltonian matrices at each time step as the system evolves.
        # Create an array to store the x values
        x_values = np.zeros_like(tpoints)
        t = np.arange(0, t_max+dt, dt)      
        rho00 = np.zeros_like(tpoints, dtype=float) 
        rho11 = np.zeros_like(tpoints, dtype=float) # Qubit 3
        rho22 = np.zeros_like(tpoints, dtype=float) # Qubit 2
        rho33 = np.zeros_like(tpoints, dtype=float) 
        rho44 = np.zeros_like(tpoints, dtype=float) # Qubit 1
        rho55 = np.zeros_like(tpoints, dtype=float)
        rho66 = np.zeros_like(tpoints, dtype=float)
        rho77 = np.zeros_like(tpoints, dtype=float)
        
##### Measurement #####
        for i in range(1, len(t)):
            # Generate random x between 0 and 1
            x = random.random()
            x_values[i] = x 
            # Set S according to the value of x
            if x < rho22[i-1]: # change this depending where we measure,  S is set to 1 if x is less than the previous value of rho44[i-1], indicating a condition or threshold is met
                S = 1
            elif x > rho22[i-1]:
                S = 0
                
            # Define r(t) and u(t) 
            r[i], u[i], rho[i], H[i] = runge_kutta_4(r[i - 1], u[i - 1], rho[i - 1], H[i - 1], H_func, dt, S, tau, tau_facil)
                        
            # Normalize the density matrix
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


        return t, r, rho11, rho22, rho33, rho00, rho44, rho55, rho66, rho77, u
    t, r, rho11, rho22, rho33, rho00, rho44, rho55, rho66, rho77, u = integrate_equation(r0, U, dt, t_max, beta,tau, tau_facil)
 
    
    results[beta] = (t, r, rho11, rho22, rho33, rho00, rho44, rho55, rho66, rho77, u)


#####Plot#######
fig, ax1 = plt.subplots(figsize=(26, 6))
# Plot rho44 in violet
plt.plot(tpoints, rho44, color='green', label='Q1')
# Plot rho22 in red
plt.plot(tpoints, rho22, color='orange', label='Q2')
# Plot rho11 in orange
plt.plot(tpoints, rho11, color='violet', label='Q3')


# plt.title(f'Occupation Probability for U = {U} (tau = {tau}, tau_facil = {tau_facil}, dt = {dt})')

# Add legend
plt.legend()

# Plot r(t) and u(t) on the same graph
plt.plot(t, r, label='r', color='blue')  # Plot r(t)
plt.plot(t, u, label='u', color='red')  # Plot u(t)

# Set the xlabel and ylabel with fontsize
plt.xlabel('Time', fontsize=46)
plt.ylabel('$P_E$', fontsize=46)

plt.legend(loc='upper right', fontsize=30)
plt.xlabel(' ', fontsize=46)

ax1.set_xlim([0, max(tpoints)])
ax1.set_ylim([-0.1, 1.1])
ax1.xaxis.label.set_size(40)
ax1.yaxis.label.set_size(40)
ax1.tick_params(axis='both', which='major', labelsize=40, length=10, width=2)
ax1.set_yticks([0, 0.5, 1])
ax1.set_yticklabels(['0', '0.5', '1'], fontsize=40)

plt.show()

#####export data txt#######


data = {
    'Time': tpoints,
    'Q1': rho44,
    'Q2': rho22,
    'Q3': rho11
}

# Create a DataFrame
df = pd.DataFrame(data)

# Export to a TXT file
txt_file_path = 'output_data.txt'
df.to_csv(txt_file_path, sep='\t', index=False)

print(f'Data exported successfully to {txt_file_path}')

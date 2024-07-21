# code for Torres replication

# Load library
import matplotlib.pyplot as plt
import numpy as np
import random


# Define the initial conditions with different paramaters
r0 = 0
tau = 500# is the parameter to see rabbit oscillations or not  if tau big like 500 only
U = 0.5
dt = 0.1 # "dt" is the time step used for numerical integration
t_max = 600000
tpoints = np.arange(0, t_max + dt, dt)

# Define the initial density matrix
rho0 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)

print(rho0)

# Compute the negativity of the system in function of time
negativity = np.zeros_like(tpoints)

# Define the function to compute the derivative of rho
def rho_derivative(H, rho):
    return -1j * np.dot(H, rho) + 1j * np.dot(rho, H)

# Define the function for fourth-order Runge-Kutta
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
    H_next = H_func(r_next)
    rho_next = rho_prev + k3_rho
    k4_rho = dt * rho_derivative(H_next, rho_next)
    k4_r = dt * ((1 - r_next) / tau - U * r_next * S)

    # Combine evaluations to estimate the next value of r and rho
    r_new = r_prev + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
    rho_new = rho_prev + (k1_rho + 2 * k2_rho + 2 * k3_rho + k4_rho) / 6

    return r_new, rho_new, H_next

# Define the Hamiltonian function
def H_func(r):
    # return np.array([[0, 0, 0, 0], [0, 0, 0.05* r, 0], [0, 0.05 * r, 0, 0], [0, 0, 0, 0]], dtype=complex)
      return np.array([[0, 0, 0, 0], [0, 0.1, 0.05 * r, 0], [0, 0.05* r, -0.1, 0], [0, 0, 0, 0]], dtype=complex)


# Iterate over the time range, updating the density matrix and calculating its elements at each step
def integrate_equation(r0,tau, U, dt, t_max):
    t = np.arange(0, t_max+dt, dt)
    r = np.zeros(len(t))
    rho = np.zeros((len(t), 4, 4), dtype=complex)
    H = np.zeros((len(t), 4, 4), dtype=complex)
    r[0] = r0
    rho[0] = rho0
    H[0] = H_func(r0) 
    # is initializing the Hamiltonian matrix H at time index 0 (t=0) using the initial value of r, 
    # which is r0. The H_func function is called with the initial value of r (r0) as its argument, and the resulting Hamiltonian matrix is assigned to the first element of the H array (i.e., H[0]).
    # The array H will store the Hamiltonian matrices at each time step as the system evolves.
    # Create an array to store the x values
    x_values = np.zeros_like(tpoints)

    rho00 = np.zeros_like(tpoints, dtype=complex)
    rho11 = np.zeros_like(tpoints, dtype=complex)
    rho22 = np.zeros_like(tpoints, dtype=complex)
    rho33 = np.zeros_like(tpoints, dtype=complex)

    for i in range(1, len(t)):
        
        # Generate random x between 0 and 1
        x = random.random()
        x_values[i] = x 

        # Set S according to the value of x
        if x < rho22[i-1]:
            S = 1
        elif x > rho22[i-1]:
            S = 0
        # Define r(t)    
        r[i], rho[i], H[i] = runge_kutta_4(r[i - 1], rho[i - 1], H[i - 1], H_func, dt, S)

        # Normalize the density matrix
        normalization_factor = np.sum(np.real(np.diag(rho[i])))
        rho[i] /= normalization_factor

        rho00[i] = np.real(rho[i, 0, 0]) # assigns the real part of the (0, 0)-th element of the rho matrix at the i-th time step to the i-th element of the rho00 array. In other words, it extracts the real part of the probability of the system being in state 1 at the i-th time step and stores it in rho11
        rho11[i] = np.real(rho[i, 1, 1])
        rho22[i] = np.real(rho[i, 2, 2])
        rho33[i] = np.real(rho[i, 3, 3])
        
        
                # 3. Compute the partial transpose with respect to the second subsystem
        rho_partial_transpose = np.array([
            [rho[i, 0, 0], rho[i, 1, 0], rho[i, 0, 2], rho[i, 1, 2]],
            [rho[i, 0, 1], rho[i, 1, 1], rho[i, 0, 3], rho[i, 1, 3]],
            [rho[i, 2, 0], rho[i, 3, 0], rho[i, 2, 2], rho[i, 3, 2]],
            [rho[i, 2, 1], rho[i, 3, 1], rho[i, 2, 3], rho[i, 3, 3]]
        ])
        
        # 4. Compute the trace norm
        trace_norm = np.trace(np.sqrt(np.dot(rho_partial_transpose, rho_partial_transpose.conj().T)))
        
        # 5. Compute the negativity
        negativity[i] = 0.5 * (trace_norm - 1)
        
        
        # # Compute the partially transposed density matrix
        # rho_partial_transpose = np.transpose(np.conjugate(rho[i, :, :]))
        
        # # # Compute the trace norm
        # trace_norm = np.trace(np.sqrt(np.dot(rho[i, :, :], rho_partial_transpose)))
        # # # Compute the negativity
        # negativity[i] = (trace_norm - 1)/2

    return t, r, rho00, rho11, rho22, rho33

t, r, rho00, rho11, rho22, rho33 = integrate_equation(r0,tau, U, dt, t_max)

# ##### #### #### Plot the negativity of the system in function of time#### #### #### 

fig, ax1 = plt.subplots(figsize=(26, 6))

plt.plot(tpoints, negativity, color='rebeccapurple')
#ax1.set_xlabel('Time', fontsize=46)  # Adjust the font size here
#ax1.set_ylabel("$N_{ss}^{Q_{12}}$", fontsize=46)  # Adjust the font size here

plt.ylim([0, 0.5])

ax1.xaxis.label.set_size(46)  # Set font size for x-axis label
ax1.tick_params(axis='both', which='major', labelsize=49)  # Adjust the font size for tick labels


ax1.set_ylim([-0.1, 0.6])

# Set the x-axis limits to start at 0
ax1.set_xlim([0, max(tpoints)])

# Set y-axis tick values for ax1
ax1.set_yticks([0, 0.5])

# Set y-axis tick labels for ax1
ax1.set_yticklabels(['0', '0.5'])

# Ensure tick marks are displayed at the bottom of the x-axis
ax1.xaxis.set_ticks_position('bottom')

# Adjust the length of the tick marks on the y-axis (e.g., set to 5 units)
ax1.yaxis.set_tick_params(length=10)


# Adjust the length of the tick marks on the x-axis (e.g., set to 5 units)
ax1.xaxis.set_tick_params(length=10)

# # Add labels for r(t) and $P_E$ inside the graph
ax1.text(0.1, 0.8, '$N_{ss}^{Q_{12}}$', transform=ax1.transAxes, fontsize=40, color='rebeccapurple', bbox=dict(facecolor='white', edgecolor='violet', boxstyle='round,pad=0.5'))


# Manually setting specific x-axis ticks and labels
specific_ticks = [0, 100000, 200000, 300000, 400000,500000]  # Example tick positions
specific_labels = ['0', '100000','200000', '300000', '400000', '500000']  # Leave the last label empty to "remove" it
ax1.set_xticks(specific_ticks)
ax1.set_xticklabels(specific_labels, fontsize=40)

plt.show()

####### #### #### #### # Plot the numerical integration of r and rho11 over time#### #### ####

fig, ax1 = plt.subplots(figsize=(26, 6))

ax2 = ax1.twinx()

ax1.plot(tpoints, r, color='violet')
ax2.plot(tpoints, rho22, 'g-')

#ax1.set_xlabel('Time', fontsize=46) # Adjust the font size here

# ax1.set_ylabel('r(t)', color='violet', fontsize=46) # Adjust the font size here
# ax2.set_ylabel('$P_E$', color='g', fontsize=46) # Adjust the font size here

# ax1.xaxis.label.set_size(46) # Set font size for x-axis label
# ax1.yaxis.label.set_size(46) # Set font size for y-axis label of ax1
# ax2.yaxis.label.set_size(46) # Set font size for y-axis label of ax2

# Set y-axis tick values for ax1
ax1.set_yticks([0, 0.5, 1])

# Set y-axis tick values for ax2
#ax2.set_yticks([0, 0.5, 1])

# Set y-axis tick labels for ax1 and ax2
ax1.set_yticklabels(['0', '0.5', '1'])
#ax2.set_yticklabels(['0', '0.5', '1'])

ax1.tick_params(axis='both', which='major', labelsize=46) # Adjust the font size for tick labels
#ax2.tick_params(axis='both', which='major', labelsize=46) # Adjust the font size for tick labels

ax1.set_ylim([-0.1, 1.1])
ax2.set_ylim([-0.1, 1.1])

# Set the x-axis limits to start at 0
ax1.set_xlim([0, max(tpoints)])

# Ensure tick marks are displayed at the bottom of the x-axis
ax1.xaxis.set_ticks_position('bottom')

# Adjust the length of the tick marks on the x-axis (e.g., set to 5 units)
ax1.xaxis.set_tick_params(length=10)

# Adjust the length of the tick marks on the y-axis (e.g., set to 5 units)
ax1.yaxis.set_tick_params(length=10)

# Add labels for r(t) and $P_E$ inside the graph
ax1.text(0.1, 0.8, 'r(t)', transform=ax1.transAxes, fontsize=40, color='violet', bbox=dict(facecolor='white', edgecolor='violet', boxstyle='round,pad=0.5'))
ax2.text(0.1, 0.3, '$P_E$', transform=ax2.transAxes, fontsize=40, color='g', bbox=dict(facecolor='white', edgecolor='g', boxstyle='round,pad=0.5'))

# Turn off ticks and labels for the y-axis of ax2
ax2.set_yticks([])
ax2.set_yticklabels([])

# Manually setting specific x-axis ticks and labels
specific_ticks = [0, 100000, 200000, 300000]  # Example tick positions
specific_labels = ['0', '100000','200000', '300000']  # Leave the last label empty to "remove" it
ax1.set_xticks(specific_ticks)
ax1.set_xticklabels(specific_labels, fontsize=40)

plt.show()



import numpy as np
import matplotlib.pyplot as plt
import random  # Import the random module


def initialize_rho(n_qubits):
    psi_0 = np.zeros((2**n_qubits,), dtype=complex)
    psi_0[2**(n_qubits-1)] = 1
    rho_0 = np.outer(psi_0, psi_0.conj())
    return rho_0

def commutator(H, rho):
    return -1j * (np.dot(H, rho) - np.dot(rho, H))

def dr_dt(r, u, tau_d, S):
    return (1 - r) / tau_d - U * r * S


def rk4_step_for_r_and_u(r, dt, U, tau_d, S):
    # RK4 step for r
    k1_r = dt * dr_dt(r, U, tau_d, S)
    k2_r = dt * dr_dt(r + 0.5 * k1_r, U, tau_d, S)
    k3_r = dt * dr_dt(r + 0.5 * k2_r, U, tau_d, S)
    k4_r = dt * dr_dt(r + k3_r, U, tau_d, S)
    r_next = r + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
    
    
    return r_next

def rk4_step(H, rho, dt):
    k1 = dt * commutator(H, rho)
    k2 = dt * commutator(H, rho + 0.5 * k1)
    k3 = dt * commutator(H, rho + 0.5 * k2)
    k4 = dt * commutator(H, rho + k3)
    return rho + (k1 + 2*k2 + 2*k3 + k4) / 6

def generate_time_dependent_hamiltonian(n_qubits, r, J_max=1.0):
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    
    # Adjust J_0 based on the chain being even or odd
    if n_qubits % 2 == 0:  # Even chain
        J_0 = 2 * J_max / n_qubits
    else:  # Odd chain
        J_0 = J_max / np.sqrt((n_qubits**2 / 4) - 1/4)
    
    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    
    print("Coupling strengths (J_{i,i+1}) for adjacent qubits:")

    for i in range(n_qubits-1):
        J_i_i1 = J_0 * np.sqrt((i+1) * (n_qubits - i - 1))
        
        print(f"J_{i}-{i+1}: {J_i_i1}")
        
       # Initialize operators for the current pair (i, i+1)
        op_xx = np.eye(1, dtype=complex)
        op_yy = np.eye(1, dtype=complex)
        
        
        for j in range(n_qubits):
            if j == i:
                op_xx = np.kron(op_xx, sigma_x)
                op_yy = np.kron(op_yy, sigma_y)
            elif j == i + 1:
                op_xx = np.kron(op_xx, sigma_x)
                op_yy = np.kron(op_yy, sigma_y)
            else:
                op_xx = np.kron(op_xx, identity)
                op_yy = np.kron(op_yy, identity)
        
        H += 0.5 * J_i_i1 * r * (op_xx + op_yy)  # Correctly multiply the terms
    
    return H


def evolve_system(n_qubits, dt, t_max, tau_d, U, r_initial):
    # Initialize rho for the system
    rho = initialize_rho(n_qubits)
    r = r_initial
    rhos = [rho.copy()]
    rs = [r]
    rho22 = [np.real(rho[2, 2])]  # Initialize rho22 with the initial value of qubit 2's occupation probability
    
    # Time points array
    tpoints = np.arange(0, t_max + dt, dt)
    
    for t in tpoints[1:]:  # Start from the second time point since the first is the initial condition
        x = random.random()  # Generate random x between 0 and 1
        
        # Determine S based on the occupation probability of qubit 2
        S = 1 if x < rho22[-1] else 0
        
        # Generate the time-dependent Hamiltonian
        H = generate_time_dependent_hamiltonian(n_qubits, r, J_max=1.0)
        
        # Update rho using the RK4 method
        rho = rk4_step(H, rho, dt)
        rhos.append(rho.copy())
        
        # Update r and u using their respective RK4 method
        r= rk4_step_for_r_and_u(r, dt, U, tau_d, S)
        rs.append(r)
        
        # Update rho22 with the current value of qubit 2's occupation probability
        rho22.append(np.real(rho[2, 2]))
    
    return np.array(rhos), np.array(rs), np.array(rho22)

def qubit_excited_probabilities(probabilities, n_qubits):
    excited_probs = np.zeros((probabilities.shape[0], n_qubits))
    for qubit in range(n_qubits):
        for state in range(2**n_qubits):
            if state & (1 << (n_qubits - qubit - 1)):
                excited_probs[:, qubit] += probabilities[:, state]
    return excited_probs

# Parameters
n_qubits = 6
dt = 0.01
t_max = 1000
tau_d = 100
r_initial = 0
U = 0.5
times = np.arange(0, t_max + dt, dt)  # Generate time points from 0 to t_max

rhos, rs, rho22 = evolve_system(n_qubits, dt, t_max, tau_d, U, r_initial)

# Calculate probabilities and excited state probabilities as before
probabilities = np.array([np.real(np.diag(rho)) for rho in rhos])
excited_probs = qubit_excited_probabilities(probabilities, n_qubits)


plt.figure(figsize=(20, 6))

# Primary axis for qubit occupation probabilities and r(t)
ax1 = plt.gca()  # Get current axis

# Use a colormap to generate colors for each qubit dynamically
cm = plt.get_cmap('tab20')  # Get a colormap from matplotlib (this one has 20 distinct colors)
colors = [cm(1.*i/n_qubits) for i in range(n_qubits)]  # Generate colors for each qubit

# Plotting all qubits, ensuring qubit 6 (index 5) is included
for qubit in range(n_qubits):
    ax1.plot(times, excited_probs[:, qubit], label=f'$Q_ {qubit+1}$', color=colors[qubit])

# Ensure r(t) is plotted on the same scale as the qubit probabilities
ax1.plot(times, rs, label='r', color='blue')

ax1.set_xlabel('Time', fontsize=46)
ax1.set_ylabel('$P_E$', fontsize=46)  # General label since this axis now includes occupation probabilities, r(t), and u(t)
ax1.set_xlim([0, t_max])
ax1.set_ylim([-0.1, 1.1])
ax1.xaxis.label.set_size(40)
ax1.yaxis.label.set_size(40)
ax1.tick_params(axis='both', which='major', labelsize=40, length=10, width=2)
ax1.set_yticks([0, 0.5, 1])
ax1.set_yticklabels(['0', '0.5', '1'], fontsize=40)

# Manually set x-tick labels to exclude the last one
x_ticks = np.linspace(0, t_max, num=5)  # Adjust the number of ticks as needed
x_ticks_labels = [f'{tick:.0f}' for tick in x_ticks[:-1]]  # Exclude the last label
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_ticks_labels + [''], fontsize=40)  # Add an empty label at the end



# Add a legend outside the plot box on the right side
plt.legend(loc='upper right', fontsize=30)

plt.show()




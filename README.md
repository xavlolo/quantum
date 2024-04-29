# Quantum spin chain info
This is a simplified version of the code for 3 qubits

with # The computational basis: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩.

In this file no temperature, no entanglement, no facilitation

Paramaters can be changed

To do do a different measure on qubit update ##### Measurement #####
** Generate random x and set S according to the value of x
                    x = random.random() 
            S = 1 if x < np.real(rho[i-1, 2, 2]) else 0   #change this depending where we measure here Q2 or Q3 (rho[i-1, 1, 1]) etc,  S is set to 1 if x is less than the previous value of rho22[i-1], indicating a condition or threshold is met
               


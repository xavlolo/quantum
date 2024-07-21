# Quantum spin chain info
Code for 3 qubits
Excitation on Q1
Homogeneous case
Depression only
Occupation probability
Entanglement (single site entanglement and entanglement between qubits 1-3) 
Data export single file for each qubit
Time for 2.5 period for different taus (0.01=222.43, 10=512.14, 100=3028.99, 500=14163.89)

with # The computational basis: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩.

Paramaters can be changed

To do do a different measure on qubit update 
Measurement...
** Generate random x and set S according to the value of x
                    x = random.random() 
            S = 1 if x < np.real(rho[i-1, 2, 2]) else 0   #change this depending where we measure here Q2 or Q3 (rho[i-1, 1, 1]) etc,  S is set to 1 if x is less than the previous value of rho22[i-1], indicating a condition or threshold is met

# Other codes, full N qubit chain, depression and facilitation, torres replication
               


# Quantum spin chain info
This is a simplified version of the code for 3 qubits

with # The computational basis: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩.

In this file no temperature, no entanglement, no facilitation

Paramaters can be changed

To do do a different measure on qubit update ##### Measurement #####
if x < rho22[i-1]: #change this depending where we measure rho44 (Qubit 1) or rhoo11 (Qubit 3)
    S = 1
elif x > rho22[i-1]:
    S = 0


import numpy as np

def is_quasi_diagonal(T):
    """Checks if a tensor T is quasi-diagonal."""
    it = np.nditer(T, flags=['multi_index'])
    for val in it:
        idx = it.multi_index
        i1 = idx[0]
        rest = idx[1:]
        
        # If i_1 is not in {i_2, ..., i_m} and the value is not zero, it fails
        if i1 not in rest and val != 0:
            return False
    return True

def generate_quasi_diagonal(m, n):
    """Generates a random quasi-diagonal tensor of order m and dimension n using whole numbers."""
    # Create a tensor filled with random integers from 1 to 9
    T = np.random.randint(1, 10, size=([n] * m))
    
    # Enforce the quasi-diagonal property
    it = np.nditer(T, flags=['multi_index'], op_flags=['readwrite'])
    for val in it:
        idx = it.multi_index
        i1 = idx[0]
        rest = idx[1:]
        
        if i1 not in rest:
            val[...] = 0 # Force to zero
            
    return T

# Set parameters: order m=3, dimension n=2
m = 3
n = 2

# Generate A and B
A = generate_quasi_diagonal(m, n)
B = generate_quasi_diagonal(m, n)

# Verify A and B are quasi-diagonal
print(f"Is A quasi-diagonal? {is_quasi_diagonal(A)}")
print(f"Is B quasi-diagonal? {is_quasi_diagonal(B)}")
print("\nTensor A:\n", A)
print("\nTensor B:\n", B)

# Compute Kronecker product C = B ⊗ A
C = np.kron(B, A)

# Verify if C is quasi-diagonal
print(f"\nIs C = B ⊗ A quasi-diagonal? {is_quasi_diagonal(C)}")

# Find specific violating indices to prove it
it = np.nditer(C, flags=['multi_index'])
for val in it:
    idx = it.multi_index
    if idx[0] not in idx[1:] and val != 0:
        print(f"\nCounterexample found at index {idx} in C.")
        print(f"C{idx} = {val} (Should be 0 because {idx[0]} is not in {idx[1:]})")
        break
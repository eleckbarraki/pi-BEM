import numpy as np


# ============================================================
# 1. Read matrices
# ============================================================

def read_matrix(filename, n):
    A = np.zeros((n, n))

    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue

            i, j, value = line.split()

            i = int(i)
            j = int(j)
            value = float(value)

            A[i, j] = value

    return A


# ============================================================
# 2. Read vectors
# ============================================================

def read_vector(filename, n):
    v = np.zeros(n)

    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue

            i, value = line.split()

            i = int(i)
            value = float(value)

            v[i] = value

    return v


# ============================================================
# 3. Read constraints
# ============================================================

def read_constraints(filename, n):
    constraints = {}

    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue

            data = line.split()

            row = int(data[0])
            n_entries = int(data[1])

            entries = {}

            pos = 2

            for k in range(n_entries):
                column = int(data[pos])
                coefficient = float(data[pos + 1])

                entries[column] = coefficient

                pos += 2

            inhomogeneity = float(data[pos])

            constraints[row] = {
                "entries": entries,
                "inhomogeneity": inhomogeneity
            }

    return constraints


# ============================================================
# Determine number of DoFs
with open("alpha.dat", "r") as f:
    n = sum(1 for line in f if line.strip())

print("Number of DoFs:", n)



# Read data
print("Reading matrices...")

N = read_matrix("neumann_matrix.dat", n)
D = read_matrix("dirichlet_matrix.dat", n)

alpha = read_vector("alpha.dat", n)

dirichlet_nodes = read_vector(
    "dirichlet_nodes.dat", n
)

neumann_nodes = read_vector(
    "neumann_nodes.dat", n
)

constraints = read_constraints(
    "constraints.dat", n
)

print("Data loaded.")


# Construct the matrix from compute_rhs()

# compute_rhs() does:
#
# dst = -(N + diag(alpha)) P_D x
#       + D P_N x
#
# where
#
# P_D = diag(dirichlet_nodes)
# P_N = diag(neumann_nodes)

P_D = np.diag(dirichlet_nodes)
P_N = np.diag(neumann_nodes)

A = -(N + np.diag(alpha)) @ P_D + D @ P_N

# Apply the constraints
# ConstrainedOperator::vmult() replaces a constrained row i by
#
#       x_i - sum_j c_ij x_j
#
# Therefore the constrained row is
#
#       A[i,:] = e_i - sum_j c_ij e_j
#

for row, constraint in constraints.items():
  A[row, :] = 0.0
  A[row, row] = 1.0

  for column, coefficient in constraint["entries"].items():
    A[row, column] -= coefficient


print()
print("Matrix shape:", A.shape)

print("Maximum absolute entry:", np.max(np.abs(A)))

print("Maximum asymmetry:", np.max(np.abs(A - A.T)))


# Compute condition number
print()
cond_2 = np.linalg.cond(N, 2)
cond_inf = np.linalg.cond(N, np.inf)
print("Condition number (2-norm):")
print(cond_2)
print("Condition number (inf-norm):")
print(cond_inf)


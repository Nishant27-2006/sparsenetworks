import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import cg

def generate_sparse_matrix_with_uncertainty(dim, noise_level=0.1):
    A = np.random.rand(dim, dim)
    A = np.triu(A)  # Upper triangular matrix to ensure sparsity
    A = csr_matrix(A)  # Convert to sparse CSR format

    # Add uncertainty (noise) to the matrix
    noise = np.random.randn(A.nnz) * noise_level
    A.data += noise  # Add noise to non-zero elements

    return A

# Function to generate the right-hand side vector
def generate_rhs(dim):
    return np.random.rand(dim)

# Function to plot convergence (Error vs. Iterations) for CG
def plot_convergence(errors):
    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Relative Error')
    plt.title('Convergence of CG Method')
    plt.grid(True)
    plt.show()

# Function to plot the condition number distribution
def plot_condition_number(matrix):
    eigenvalues = np.linalg.eigvals(matrix.toarray())  # Convert sparse matrix to dense
    plt.hist(np.abs(eigenvalues), bins=50)
    plt.xlabel('Eigenvalue Magnitude')
    plt.ylabel('Frequency')
    plt.title('Condition Number Distribution')
    plt.show()

# Function to plot the sparsity pattern of the matrix
def plot_matrix_sparsity(matrix):
    coo = matrix.tocoo()  # Convert sparse matrix to COO format
    plt.scatter(coo.col, coo.row, color='blue', s=1)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.title('Sparsity Pattern of the Matrix')
    plt.show()

def plot_non_zero_elements_distribution(matrix):
    non_zero_values = matrix.data 
    plt.hist(non_zero_values, bins=50)
    plt.xlabel('Non-zero Value Magnitude')
    plt.ylabel('Frequency')
    plt.title('Non-zero Elements Value Distribution')
    plt.show()

def plot_execution_time(dimensions, times):
    plt.plot(dimensions, times, marker='o', color='green')
    plt.xlabel('Matrix Size (dim)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Matrix Size')
    plt.grid(True)
    plt.show()

def solve_system(A, b, max_iter=1000, tol=1e-5):
    x0 = np.zeros_like(b)
    errors = []

    def callback(xk):
        error = np.linalg.norm(A @ xk - b) / np.linalg.norm(b)
        errors.append(error)
        if error < tol:
            return True  
        return False

    x, info = cg(A, b, x0=x0, maxiter=max_iter, callback=callback)

    return x, errors

def test_solver(dim=1000):
    A = generate_sparse_matrix_with_uncertainty(dim)
    b = generate_rhs(dim)

    plot_matrix_sparsity(A)

    plot_condition_number(A)

    _, errors = solve_system(A, b)
    plot_convergence(errors)

    plot_non_zero_elements_distribution(A)

def run_tests():
    dimensions = [500, 1000, 1500, 2000, 2500] 
    times = []
    for dim in dimensions:
        print(f"Testing with matrix dimension: {dim}")
        import time
        start_time = time.time()
        test_solver(dim)
        end_time = time.time()
        times.append(end_time - start_time)

    plot_execution_time(dimensions, times)

run_tests()

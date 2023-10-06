import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns
import random
import networkx as nx
import pandas as pd


# -------------------------------------------------
# helper functions for network feedback model
# -------------------------------------------------

def compute_phase_coherence(data):
    """
    Computes the phase-coherence order parameter of a 2D NumPy array of oscillators.

    Parameters:
    data (numpy.ndarray): The 2D NumPy array of oscillators, where each row is an oscillator and each column is a time domain.

    Returns:
    float: The phase-coherence order parameter of the oscillators.
    """
    # Compute the complex phases of the oscillators
    phases = np.exp(1j * data)

    # Compute the mean of the complex phases
    mean_phase = np.mean(phases, axis=0)

    # Compute the magnitude of the mean phase
    coherence_parameter = np.abs(mean_phase)

    return coherence_parameter


@jit(nopython=True)
def feedback(t, y, L, A, rho, a0, ai, aii, api, eps, delta, K, c, w):
    # set up variables as NumPy arrays indexed by node k
    N = L.shape[0]
    u = np.ascontiguousarray(y[:N])  # contiguous array speeds up computation
    up = np.ascontiguousarray(y[N:2*N])
    theta = np.ascontiguousarray(y[2*N:])

    # scale Laplacian by diffusion constant
    #L = rho * L

    # phase-dynamics
    dtheta = w - c*up + K/N * np.diag(np.dot(np.sin(theta - np.expand_dims(theta, axis=1)), A))
    dtheta *= 1/eps
    thetaM = np.diag(dtheta)

    # create modified Laplacian matrix
    I = np.identity(N)
    LM = L @ (I + delta*thetaM)
    LM = rho*LM

    # heterodimer dynamics
    duk =  -LM @ u + a0 - ai*u   - aii*u*up
    dupk = -LM @ up     - api*up + aii*u*up

    # pack right-hand side
    rhs = np.concatenate((duk, dupk, dtheta))

    return rhs


@jit(nopython=True)
def skewed_heterodimer(t, y, L, A, rho, a0, ai, aii, api, delta):
    # set up variables as NumPy arrays indexed by node k
    N = L.shape[0]
    u = np.ascontiguousarray(y[:N])  # contiguous array speeds up computation
    up = np.ascontiguousarray(y[N:2*N])

    # phase-dynamics
    A = np.diag(A)

    # create modified Laplacian matrix
    I = np.identity(N)
    LM = L @ (I + delta*A)
    LM = rho*LM

    # heterodimer dynamics
    duk =  -LM @ u + a0 - ai*u   - aii*u*up
    dupk = -LM @ up     - api*up + aii*u*up

    # pack right-hand side
    rhs = np.concatenate((duk, dupk))

    return rhs

def plot_feedback(sol, file, A, eps, delta, K, c, w, colours=[]):
    # find N
    N = A.shape[0]

    # extract solution
    u = sol.y[0:N,:]
    up = sol.y[N:2*N,:]
    theta = sol.y[2*N:3*N,:]
    t = sol.t

    # plot u
    plt.style.use('seaborn-muted')
    plt.figure()
    for k in range(N):
        plt.plot(t, u[k], c=colours[k])
    plt.ylabel(r'$u$')
    plt.xlabel('slow time')
    plt.savefig(file+'u.pdf',dpi=300)

    # plot up
    plt.figure()
    for k in range(N):
        plt.plot(t, up[k], c=colours[k])
    plt.ylabel(r'$\tilde{u}$')
    plt.xlabel('slow time')
    plt.savefig(file+'up.pdf',dpi=300)

    # plot activity
    plt.figure()
    dtheta = []
    for k in range(N):
        dthetak = w[k] - c*up[k] + K * sum(A[k, l] * np.sin(theta[l] - theta[k]) for l in range(N))
        dtheta.append(dthetak)
    for k in range(N):
        plt.plot(t, delta/eps*dtheta[k], c=colours[k])
    plt.ylabel(r'$A$')
    plt.xlabel('slow time')
    plt.savefig(file+'A.pdf',dpi=300)

    # plot theta
    fast_time = 10
    subfig_window = (t >= t[-1] - fast_time*eps)
    plt.figure()
    theta_mod = np.sin(theta)  # sinusoid
    for k in range(N):
        plt.plot(t[subfig_window], theta_mod[k,subfig_window], c=colours[k])
    plt.ylabel(r'$\sin{\theta_k}$')
    plt.xlabel('slow time')
    plt.gca().ticklabel_format(useOffset=False, style='plain')  
    plt.savefig(file+'theta.pdf',dpi=300)

    # plot kuramoto order parameter
    phase_coherence = compute_phase_coherence(theta) 

    # create main figure with two subplots
    fast_time = 20
    subfig_window = (t >= t[-1] - fast_time*eps)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.3)

    # plot last one second of order
    for k in range(N):
        axs[0].plot(t[subfig_window], phase_coherence[subfig_window], c=colours[-1])
    axs[0].set_ylabel(r'phase-coherence')
    axs[0].ticklabel_format(useOffset=False, style='plain')
    axs[0].set_ylim([-0.1, 1.1])

    # plot theta over all time
    for k in range(N):
        axs[1].plot(t, phase_coherence, c=colours[-1])
    axs[1].set_ylabel(r'phase-coherence')
    axs[1].set_xlabel('slow time')
    axs[0].ticklabel_format(useOffset=False, style='plain')
    plt.savefig(file+'phase_coherence.pdf',dpi=300)

    return None


def create_clustered_network(N, M, k):
    # Create graph
    G = nx.Graph()

    # Add M clusters
    for i in range(M):
        # Create cluster
        cluster = nx.complete_graph(np.arange(i*N, i*N+N))
        # Add cluster to graph
        G = nx.compose(G,cluster)

    # Connect clusters
    for i in range(M):
        for j in range(i+1, M):
            # Add k random edges between clusters
            nodes_i = range(i*N, (i+1)*N)
            nodes_j = range(j*N, (j+1)*N)
            for _ in range(k):
                u = random.choice(nodes_i)
                v = random.choice(nodes_j)
                G.add_edge(u, v)

    # Return graph
    return G


def get_column_index(arr, C, is_larger=True):
    n, m = arr.shape
    res = np.empty(n, dtype=int)
    res.fill(-1)
    for i in range(n):
        for j in range(m):
            if (is_larger and arr[i,j] > C) or (not is_larger and arr[i,j] < C):
                res[i] = j
                break
    return res


def read_ods_file(file_path):
    # Load the .ods file using pandas
    df = pd.read_excel(file_path, engine='odf', sheet_name=0, header=None)

    # Convert the dataframe to a numpy array
    array = df.values.astype(object)
    return array

def read_csv_file(file_path, delimiter=None, header='infer'):
    dataframe = pd.read_csv(file_path, delimiter=delimiter, header=header)
    numpy_array = dataframe.to_numpy()
    return numpy_array


def average_labels(matrix, braak):
    # Get the number of nodes and labels
    M = matrix.shape[1]
    L = len(np.unique(braak))

    # Initialize the output matrix
    output = np.zeros((L, M))

    # Compute the averages for each label
    for i, label in enumerate(np.unique(braak)):
        mask = (braak == label)
        output[i, :] = np.mean(matrix[mask], axis=0)

    return output

def average_labels2(matrix, braak):
    # Get the number of nodes and labels
    M = matrix.shape[1]
    L = len(np.unique(braak[:, 1]))

    # Initialize the output matrix
    output = np.zeros((L, M))

    # Compute the averages for each label
    for i, label in enumerate(np.unique(braak[:, 1])):
        mask = (braak[:, 1] == label)
        indices = braak[mask][:, 0].astype(int)
        print(label)
        print(indices)
        output[i, :] = np.mean(matrix[indices], axis=0)

    return output


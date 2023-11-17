import numpy as np
import symengine as sym
import time as timer
from scipy.integrate import solve_ivp
from scipy.spatial.distance import hamming
from .brain_analysis import PLI, compute_phase_coherence, butter_bandpass_filter, PLI_from_complex, compute_phase_coherence_from_complex, compute_phase_coherence_old
from scipy.stats import pearsonr
from numba import jit
from math import pi
import warnings
warnings.filterwarnings("ignore", message="Differential equation does not include a delay term.")
warnings.filterwarnings("ignore", message="The target time is smaller than the current time. No integration step will happen. The returned state will be extrapolated from the interpolating Hermite polynomial for the last integration step. You may see this because you try to integrate backwards in time, in which case you did something wrong. You may see this just because your sampling step is small, in which case there is no need to worry.")

# -----------------------------------------------
# in this module, we include functions to compile
# and solve neural mass models using JiTC*DE
# -----------------------------------------------


# --------------------------
# random_initial()
# set up random initial conditions for 
# Hopf network with N nodes
# ---------------------------
def random_initial(N):
    theta0 = np.random.uniform(0, 2*3.14, N)
    R0 = np.random.uniform(0,1,N)
    y0 = np.zeros((2*N)) 
    y0[::2] = R0 * np.cos(theta0)
    y0[1::2] = R0 * np.sin(theta0)
    return y0


# ------------------------------------------
# Compile skewed heterodimer into C++ wrapper
# -------------------------------------------
def compile_skewed_heterodimer(L, A, rho, a0, ai, aii, api, delta, control_pars=[]):
    # import must be within function (or else t will not be caught)
    from jitcode import jitcode, y
    # extract N
    N = L.shape[0]

    # create modified Laplacian matrix
    I = np.identity(N)
    A = np.diag(A)
    LM = L @ (I + delta*A)
    LM = rho*LM
    
    # define ODE
    def skewed_heterodimer():
        for k in range(N):
            yield sum([-LM[k,l]*y(2*l+0) for l in range(N)]) + a0 \
                             - ai*y(2*k+0) - aii*y(2*k+0)*y(2*k+1)
            yield sum([-LM[k,l]*y(2*l+1) for l in range(N)]) \
                             - api*y(2*k+1) + aii*y(2*k+0)*y(2*k+1)

    # compile DDE, set integration parameters, and store number of nodes
    ODE = jitcode(skewed_heterodimer, n=2*N, control_pars=control_pars)  
    ODE.compile_C()

    return ODE

def solve_ODE(ODE, y0, tspan, pars=False, step=1e-3, rtol=1e-3, atol=1e-6, display=False, \
                method='dopri5'):
    # start clock
    if display:
        start = timer.time()
    # import must be within function (or else t will not be caught)
    from jitcode import jitcode, y

    # check if parameter array given correctly
    if pars is False:
        pars = np.array([])
    else:
        pars = np.array( pars )
    num_par = pars.size

    # set integration parameters
    ODE.set_integrator(method,rtol=rtol,atol=atol)

    # set past history
    ODE.set_initial_value(y0, 0.0)

    # set model parameters (only if set by user)
    if num_par:
        try:
            ODE.set_parameters(pars)
        except:
            print(f'\nThe number of implicit parameters is {num_par}. Make sure that this is reflected in the JiTCODE compilation.\n')
            return None, None

    # solve
    data = []
    t = []
    for time in np.arange(ODE.t, ODE.t+tspan[1],  step):
        data.append( ODE.integrate(time) )
        t.append(time)

    # organize data
    data = np.array(data).T  # variables in rows, time points in columns
    t = np.array(t)

    # display simulation time
    if display:
        end = timer.time()
        print(f'\nElapsed time for DDE simulations: {end-start} seconds')
    
    # we're done
    return data, t

# -----------------------------------------
# compile wilson-cowan model into
# C++ wrapper
# INPUT:
# Wilson Cowan parameters (parameters to be changed must be symengine variables)
# control_pars - list of symengine variables (parameters that can be changed)
# OUTPUT:
# DDE - JiTCDDE object
# y0 - numpy array (initial conditions)
# -----------------------------------------
def compile_wilson_cowan(N, P=1.0, Q=-2.0, delays=False, taux=0.013, tauy=0.013, \
        Cxx=24, Cxy=-20, Cyy=0, Cyx=40, h=1, Sa=1, theta=4, kappa=0.4, \
        random_init=True, osc_freqs=False, control_pars=()):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t

    # construct adjacency matrix of symbols
    W = [[sym.var(f'W_{i}_{j}') for j in range(N)] for i in range(N)]

    # specify node frequency if applicable
    tauxs = []
    tauys = []
    for k in range(N):
        if osc_freqs:
            tauxs.append(0.346 / osc_freqs[k])
            tauys.append(0.346 / osc_freqs[k])
        else:
            tauxs.append(taux)
            tauys.append(tauy)

    # if P or Q not list then make list (list necessary for symengine variables)
    if not isinstance(P,list):
        P_val = P
        P = [P_val for _ in range(N)]
    if not isinstance(Q,list):
        Q_val = Q
        Q = [Q_val for _ in range(N)]
    
    def neural_mass():
        for k in range(N):
            # define input to node
            aff_inp = kappa*sum( W[j][k] * y(2*j+0, t-delays[j,k]) for j in range(N) )

            x_inp = (Cxx*y(2*k+0) + Cxy*y(2*k+1) + P[k] + aff_inp)
            y_inp = (Cyx*y(2*k+0) + Cyy*y(2*k+1) + Q[k])

            Sx = h*(1+sym.exp(-Sa*(x_inp - theta)))**-1
            Sy = h*(1+sym.exp(-Sa*(y_inp - theta)))**-1

            yield 1/tauxs[k] * (-y(2*k+0) + Sx)
            yield 1/tauys[k] * (-y(2*k+1) + Sy)

    # set up initial conditions
    if random_init:
        y0 = np.random.uniform(0,0.2, size=2*N)
    else:
        y0 = np.full((2*N),0.1)

    # flatten symbolic adjacency matrix as list
    flat_W = list(np.array(W).flatten())

    # include symbolic adjacency matrix as implicit parameters
    control_pars = [*flat_W, *control_pars]

    # compile DDE, set integration parameters, and store number of nodes
    DDE = jitcdde(neural_mass, n=2*N, control_pars=control_pars)  
    DDE.compile_C(do_cse=True, verbose=False)

    # add number of nodes and initial conditions to DDE object
    DDE.N = N
    DDE.y0 = y0

    return DDE

# -----------------------------------------
# compile a stochastic 3-node wilson-cowan model
# without delay into C++ wrapper
# INPUT:
# Wilson Cowan parameters (parameters to be changed must be symengine variables)
# control_pars - list of symengine variables (parameters that can be changed)
# OUTPUT:
# SDE - JiTCSDE object
# y0 - numpy array (initial conditions)
# -----------------------------------------
def compile_slow_fast_wilson_cowan(N, P=1.0, Q=-2.0, R=-10.0, taux=0.013, tauy=0.013, \
        tauz=0.267, Cxx=24, Cxy=-20, Cxz=0, Cyy=0, Cyx=40, Cyz=0, Czx=0, Czy=0, Czz=0, \
        mu=0.05, sigma=0.05, tau=0.05, h=1, Sa=1, theta=4, kappa=0.4, \
        random_init=True, osc_freqs=False, control_pars=()):
    # import must be within function
    from jitcsde import jitcsde, y
    from math import e

    # construct adjacency matrix of symbols
    W = [[sym.var(f'W_{i}_{j}') for j in range(N)] for i in range(N)]

    # specify node frequency if applicable
    tauxs = []
    tauys = []
    tauzs = []
    for k in range(N):
        if osc_freqs:
            factor = 25  # picked per Wang (2012)
            tauxs.append(0.346 / osc_freqs[k])
            tauys.append(0.346 / osc_freqs[k])
            tauzs.append(factor * 0.346 / osc_freqs[k])
        else:
            tauxs.append(taux)
            tauys.append(tauy)
            tauzs.append(tauz)

    # if P or Q not list then make list (list necessary for symengine variables)
    if not isinstance(P,list):
        P_val = P
        P = [P_val for _ in range(N)]
    if not isinstance(Q,list):
        Q_val = Q
        Q = [Q_val for _ in range(N)]
    if not isinstance(R,list):
        R_val = R
        R = [R_val for _ in range(N)]
    
    # deterministic right-hand side
    def neural_mass():
        for k in range(N):
            # define afferent input to excitatory subnode
            aff_inp = kappa*sum( W[j][k] * y(6*j+0) for j in range(N) )
    
            # define inputs into each subnode
            x_inp = Cxx*y(6*k+0) + Cxy*y(6*k+1) + Cxz*y(6*k+2) + P[k] + aff_inp
            y_inp = Cyx*y(6*k+0) + Cyy*y(6*k+1) * Cyz*y(6*k+2) + Q[k]
            z_inp = Czx*y(6*k+0) + Czy*y(6*k+1) * Czz*y(6*k+2) + R[k]
            
            Sx = h*(1+sym.exp(-Sa*(x_inp - theta)))**-1
            Sy = h*(1+sym.exp(-Sa*(y_inp - theta)))**-1
            Sz = h*(1+sym.exp(-Sa*(z_inp - theta)))**-1

            # yield
            yield 1/tauxs[k] * (-y(6*k+0) + Sx) + y(6*k+3)
            yield 1/tauys[k] * (-y(6*k+1) + Sy) + 0*y(6*k+4)
            yield 1/tauzs[k] * (-y(6*k+2) + Sz) + 0*y(6*k+5)
            yield -(y(6*k+3)-mu)/tau
            yield -(y(6*k+4)-mu)/tau
            yield -(y(6*k+5)-mu)/tau

    # stochastic right hand side
    def noise():
        for k in range(N):
            yield 0
            yield 0
            yield 0
            yield sigma * (2/tau)**(1/2)
            yield sigma * (2/tau)**(1/2)
            yield sigma * (2/tau)**(1/2)

    # set up initial conditions
    if random_init:
        R0 = [np.random.uniform(0, 0.2) for _ in range(N)]
    else:
        R0 = [0.1 for _ in range(N)]
    y0 = np.zeros((6*N))
    for k in range(0,N):
        # subnodes initial values
        y0[6*k+0] = R0[k] 
        y0[6*k+1] = R0[k]
        y0[6*k+2] = R0[k]
        # noise initial values
        y0[6*k+3] = 0 
        y0[6*k+4] = 0
        y0[6*k+5] = 0

    # flatten symbolic adjacency matrix as list
    flat_W = list(np.array(W).flatten())

    # include symbolic adjacency matrix as implicit parameters
    control_pars = [*flat_W, *control_pars]

    # compile DDE, set integration parameters, and store number of nodes
    SDE = jitcsde(neural_mass, noise, n=6*N, control_pars=control_pars)  
    SDE.compile_C(do_cse=False, verbose=True)

    # add number of nodes and initial conditions to DDE object
    SDE.N = N
    SDE.y0 = y0

    return SDE

# -----------------------------------------
# compile hopf normal form model into
# C++ wrapper
# INPUT:
# Hopf normal form parameters (parameters to be changed must be symengine variables)
# control_pars - list of symengine variables (parameters that can be changed)
# OUTPUT:
# DDE - JiTCDDE object
# y0 - numpy array (initial conditions)
# -----------------------------------------
def compile_hopf(N, a=False, b=False, delays=False, t_span=(0,10), \
             kappa=10, h=1, w=False, decay=-0.01, inter_idx=[], inter_c=1,  \
             random_init=True, delay_c=1, max_delay=None, decay0=0, decay1=1, \
             only_a=False, control_pars=()):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t

    # set default parameter values
    if delays is False:
        delays = np.zeros((N,N))
    if not a:
        a = 1
    if not b:
        b = 1

    # construct adjacency matrix of symbols
    W = [[sym.var(f'W_{i}_{j}') for j in range(N)] for i in range(N)]

    # interhemispheric coupling matrix (scales interhemispheric coupling by inter_c)
    inter_mat = [ [1 for _ in range(N)] for _ in range(N) ]
    for e1, e2 in inter_idx:
        inter_mat[e1][e2] = inter_c

    # if a or b not list then make list (list necessary for symengine variables)
    if not isinstance(a,list):
        a_val = a
        a = [a_val for _ in range(N)]
    if not isinstance(b,list):
        b_val = b
        b = [b_val for _ in range(N)]
    if not isinstance(decay,list):
        decay_val = decay
        decay = [decay_val for _ in range(N)]
    if not isinstance(h,list):
        h_val = h
        h = [h_val for _ in range(N)]

    # TEST DISCARDING B SEMIAXIS
    if only_a:
        b = a

    # define generator of rhs
    def neural_mass():
        for k in range(N):
            # define input to node
            afferent_input = kappa * sum( inter_mat[j][k] * W[j][k] * y(2*j+0, t-delay_c*delays[j,k]) for j in range(N) )

            # transform decays
            decay[k] = decay1*(decay[k]-decay0)

            # dynamics of node k
            yield decay[k]*y(2*k+0) - w[k]*(a[k]/b[k])*y(2*k+1) \
                     - y(2*k+0)*(y(2*k+0)**2/a[k]**2 + y(2*k+1)**2/b[k]**2) \
                         + h[k] * sym.tanh(afferent_input)
            yield decay[k]*y(2*k+1) + w[k]*(b[k]/a[k])*y(2*k+0)  \
                     - y(2*k+1)*(y(2*k)**2/a[k]**2 + y(2*k+1)**2/b[k]**2)

    # set up initial conditions
    if random_init:
        theta0 = np.random.uniform(0, 2*3.14, N)
        R0 = np.random.uniform(0,1,N)
    else:
        R0 = np.full((N),1)
        theta0 = np.full((N),0)
    y0 = np.zeros((2*N)) 
    y0[::2] = R0 * np.cos(theta0)
    y0[1::2] = R0 * np.sin(theta0)
    
    # flatten symbolic adjacency matrix as list
    flat_W = list(np.array(W).flatten())

    # include symbolic adjacency matrix as implicit parameters
    control_pars = [*flat_W, *control_pars]

    # compile DDE, set integration parameters, and store number of nodes
    DDE = jitcdde(neural_mass, n=2*N, control_pars=control_pars, max_delay=max_delay)  
    DDE.compile_C(do_cse=True, chunk_size=int(N*2))  # after vacation this is suddenly slow

    # add number of nodes and initial conditions to DDE object
    DDE.N = N
    DDE.y0 = y0

    return DDE

# -----------------------------------------
# compile diffusive hopf normal form model into
# C++ wrapper
# INPUT:
# Hopf normal form parameters (parameters to be changed must be symengine variables)
# control_pars - list of symengine variables (parameters that can be changed)
# OUTPUT:
# DDE - JiTCDDE object
# y0 - numpy array (initial conditions)
# -----------------------------------------
def compile_hopf_diff(N, a=False, b=False, delays=False, t_span=(0,10), \
             kappa=10, w=False, decay=-0.01, inter_idx=[], inter_c=1,  \
             random_init=True, delay_c=1, max_delay=None, \
             noise_sf=11, noise_avg=0, noise_std=11, control_pars=()):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t, jitcdde_input, input
    from chspy import CubicHermiteSpline

    # construct adjacency matrix of symbols
    W = [[sym.var(f'W_{i}_{j}') for j in range(N)] for i in range(N)]

    # interhemispheric coupling matrix (scales interhemispheric coupling by inter_c)
    inter_mat = [ [1 for _ in range(N)] for _ in range(N) ]
    for e1, e2 in inter_idx:
        inter_mat[e1][e2] = inter_c
    W = np.multiply(W,inter_mat) 

    # if a or b not list then make list (list necessary for symengine variables)
    if not isinstance(a,list):
        a_val = a
        a = [a_val for _ in range(N)]
    if not isinstance(b,list):
        b_val = b
        b = [b_val for _ in range(N)]

    # scale delay matrix
    delays = delay_c * delays


    # create noise function
    input_n = noise_sf * 2*t_span[1]
    input_times = np.linspace(t_span[0], 2*t_span[1], input_n)
    input_data = np.sqrt(1/noise_sf) * noise_std * np.random.normal(loc=noise_avg, \
                     scale=1, size=(input_n,2*N))
    input_spline = CubicHermiteSpline.from_data(input_times,input_data)

    # define generator of rhs
    def neural_mass():
        for k in range(N):
            # define input to node
            exc_input = kappa * sum( W[j][k] * (y(2*j+0, t-delays[j,k]) - y(2*k+0)) for j in range(N))
            inh_input = kappa * sum( W[j][k] * (y(2*j+1, t-delays[j,k]) - y(2*k+1)) for j in range(N))

            # dynamics of node k
            yield decay*y(2*k+0) - w[k]*(a[k]/b[k])*y(2*k+1) \
                     - y(2*k+0)*(y(2*k+0)**2/a[k]**2 + y(2*k+1)**2/b[k]**2) + exc_input #+ input(2*k+0)
            yield decay*y(2*k+1) + w[k]*(b[k]/a[k])*y(2*k+0) \
                     - y(2*k+1)*(y(2*k+0)**2/a[k]**2 + y(2*k+1)**2/b[k]**2) + inh_input #+ input(2*k+1)


    # set up initial conditions
    if random_init:
        theta0 = np.random.uniform(0, 2*3.14, N)
        R0 = np.random.uniform(0,1,N)
    else:
        R0 = np.full((N),1)
        theta0 = np.full((N),0)
    y0 = np.zeros((2*N)) 
    y0[::2] = R0 * np.cos(theta0)
    y0[1::2] = R0 * np.sin(theta0)
    
    # flatten symbolic adjacency matrix as list
    flat_W = list(np.array(W).flatten())

    # include symbolic adjacency matrix as implicit parameters
    control_pars = [*flat_W, *control_pars]

    # compile DDE, set integration parameters, and store number of nodes
    #DDE = jitcdde(neural_mass, n=2*N, control_pars=control_pars, max_delay=max_delay)  
    DDE = jitcdde_input(neural_mass, input_spline, n=2*N, control_pars=control_pars, max_delay=max_delay)  
    DDE.compile_C(do_cse=True, chunk_size=int(2*N), verbose=True)

    # add number of nodes and initial conditions to DDE object
    DDE.N = N
    DDE.y0 = y0

    return DDE

# ----------------------------------------------------------------
# solve DDE 
# INPUT:
# DDE - a jitcdde object 
# y0 - numpy array (initial conditions)
# parameterss -  numpy array shape: (#runs, #parameters)
# -> each row is a parameter setting with a parameter in each
# column
# OUTPUT:
#   sols: (#runs) array with solutions stored as dictionaries
# ----------------------------------------------------------------
def solve_dde(DDE, y0, W, t_span=(0,10), step=10**-4, atol=10**-6, rtol=10**-4, parameterss=False, display=False, discard_y=False, cutoff=0):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t

    # check if parameter array given
    if parameterss is False:
        parameterss = np.array([[]])
        parN, num_par = (1, 0)
    else:
        parameterss = np.array( parameterss )
        parN, num_par = parameterss.shape

    # initialize 
    sols = np.empty((parN), dtype='object')

    # set number of nodes and flatten values of adjacency matrix
    N = W.shape[0]
    flat_num_W = list(W.flatten())

    # set integration parameters
    DDE.set_integration_parameters(rtol=rtol,atol=atol)
    #DDE.set_integration_parameters(rtol=1e12,atol=1e12, first_step=10**-4, max_step=10**-4, min_step=10**-4)  # test fixed step size

    # start clock
    if display:
        start = timer.time()

    # loop over parameters
    for i in range(parN):
        # add numeric adj. matrix and add model parameters
        parameters = [*flat_num_W, *parameterss[i,:]]

        # set past history
        DDE.constant_past(y0, time=0.0)

        # set model parameters (only if set by user)
        try:
            DDE.set_parameters(parameters)
        except:
            print(f'\nThe number of implicit parameters is {num_par}. Make sure that this is reflected in the JiTCDDE compilation.\n')
            return None, None

        # handle initial discontinuities
        DDE.adjust_diff()
        #DDE.step_on_discontinuities(propagations=1)
        #DDE.integrate_blindly(0.01, step=step)

        # solve
        data = []
        t = []
        for time in np.arange(DDE.t, DDE.t+t_span[1],  step):
            data.append( DDE.integrate(time) )
            t.append(time)

        # organize data
        data = np.array(data)
        data = np.transpose(data)
        t = np.array(t)

        # store solution as dictionary, potentially discard y and cut off transients
        sol = {}
        sol['x'] = data[0:2*N:2,t>cutoff]
        if discard_y:
            sol['y'] = []
        else:
            sol['y'] = data[1:2*N:2,t>cutoff]
        sol['t'] = t[t>cutoff]

        # purge past history
        DDE.purge_past()

        # store solution in grid array
        sols[i] = sol

    # display simulation time
    if display:
        end = timer.time()
        print(f'\nElapsed time for all DDE simulations: {end-start} seconds\nElapsed time per DDE simulation: {round((end-start)/parN,4)}')
    
    # we're done
    return sols 

# ----------------------------------------------------------------
# solve DDE 
# INPUT:
# SDE - a jitcsde object 
# y0 - numpy array (initial conditions)
# parameterss -  numpy array shape: (#runs, #parameters)
# -> each row is a parameter setting with a parameter in each
# column
# OUTPUT:
#   sols: (#runs) array with solutions stored as dictionaries
# ----------------------------------------------------------------
def solve_sde(SDE, y0, W, t_span=(0,10), step=10**-4, atol=10**-6, rtol=10**-4, parameterss=False, display=False, discard_y=False, cutoff=0):
    # import must be within function
    from jitcsde import jitcsde, y

    # check if parameter array given
    if parameterss is False:
        parameterss = np.array([[]])
        parN, num_par = (1, 0)
    else:
        parameterss = np.array( parameterss )
        parN, num_par = parameterss.shape

    # initialize 
    sols = np.empty((parN), dtype='object')

    # set number of nodes and flatten values of adjacency matrix
    N = SDE.N
    flat_num_W = list(W.flatten())

    # set integration parameters
    SDE.set_integration_parameters(rtol=rtol,atol=atol)

    # start clock
    if display:
        start = timer.time()

    # loop over parameters
    for i in range(parN):
        # add numeric adj. matrix and add model parameters
        parameters = [*flat_num_W, *parameterss[i,:]]

        # set past history
        SDE.set_initial_value(y0,0.0)

        # set model parameters (only if set by user)
        try:
            SDE.set_parameters(parameters)
        except:
            print(f'\nThe number of implicit parameters is {num_par}. Make sure that this is reflected in the JiTCSDE compilation.\n')
            return None, None

        # solve
        data = []
        t = []
        for time in np.arange(SDE.t, SDE.t+t_span[1],  step):
            data.append( SDE.integrate(time) )
            t.append(time)

        # organize data
        data = np.array(data)
        data = np.transpose(data)
        t = np.array(t)

        # store solution as dictionary, potentially discard y and cut off transients
        sol = {}
        sol['x'] = data[0:6*N:6,t>cutoff]
        if discard_y:
            sol['y'] = []
        else:
            sol['y'] = data[2:6*N:6,t>cutoff]
        sol['t'] = t[t>cutoff]

        # store solution in grid array
        sols[i] = sol

    # display simulation time
    if display:
        end = timer.time()
        print(f'\nElapsed time for all SDE simulations: {end-start} seconds\nElapsed time per SDE simulation: {round((end-start)/parN,4)}')
    
    # we're done
    return sols 

def threshold_matrix(A, perc):
    Acopy = copy.copy(A)
    A_flat = Acopy.flatten()
    A_flat = np.sort(A_flat)  # sort lowest to highest
    threshold_val = A_flat[int( perc * A_flat.size ) ]
    A[A < threshold_val] = 0
    return A

import numpy as np

def jaccard_index(arr1, arr2, perc):
    """
    Computes the Jaccard index between two 1D NumPy arrays.

    Args:
        arr1 (numpy.ndarray): A 1D NumPy array.
        arr2 (numpy.ndarray): A 1D NumPy array of the same length as `arr1`.

    Returns:
        The Jaccard index between `arr1` and `arr2`.
    """
    arr1 = threshold_matrix(arr1, perc)
    arr2 = threshold_matrix(arr2, perc)
    intersection = np.intersect1d(arr1, arr2)
    union = np.union1d(arr1, arr2)
    jaccard = len(intersection) / len(union)
    return jaccard

# --------------------------------------------------------------------------
# Objective function using a jitc*de object and solve_dde()
# to minimize the error between exp and simulated
# PLI functional connectomes
# INPUT
#   var -   np array of variables to optimize over
#   DE  -   jitc*de object
#   W   -   np array adjacency matrix
#   tspan - 2-tuple, timespan for dynamical system
#   atol    -   float, absolute tolerance for dde solver
#   rtol    -   float, relative tolerance for dde solver
#   cutoff  -   float, transient time to cut
#   band    -   2-tuple, frequency band for functional connectivity
#   exp_PLI -   np array, experimental functional connectivity to compare
# --------------------------------------------------------------------------
def error_FC(var, DE, W, tspan, step, atol, rtol, cutoff, band, exp_PLI, \
                normalize_exp, threshold_exp, normalize_sim, threshold_sim, zero_scale, y0, inds, objective, freq_normal, mean_coherence, par_coherence):
    # find N
    N = np.array(W).shape[0]
    # pack variable for solve_dde
    var = np.array([var])
    if threshold_exp == -1:
        var_dde = var[:,0:-1]
        threshold_exp = var[:,-1]
    elif freq_normal:
        var_dde = var[:,0:-2]
        mean_freq = var[:,-2]
        var_freq = var[:,-1]
        freqs = 2*pi*np.random.normal(loc=mean_freq, scale=var_freq, size=N)
        freqs[freqs<0] = 0
        var_dde = np.concatenate((var_dde, freqs.reshape(1,-1)), axis=1)
    else:
        var_dde = var

    # set generic initial condition if not set
    if y0 is False:
        y0 = DE.y0

    # solve DE
    sol = solve_dde(DE, y0, W, t_span=tspan, step=step, atol=atol, rtol=rtol, \
             parameterss=var_dde, discard_y=False, cutoff=cutoff)

    # extract solution
    x = sol[0]['x']
    y = sol[0]['y']
    t = sol[0]['t']
    compl_signal = x + 1j * y

    # sampling rate
    fs = 1/(t[1]-t[0])

    # bandpass
    x = butter_bandpass_filter(x, band[0], band[1], fs)

    # compute PLI matrix, normalize, and flatten
    #sim_PLI = PLI(x)
    sim_PLI = PLI_from_complex(compl_signal)
    if not objective == 'jaccard':
        if normalize_sim:
            sim_max = np.amax(sim_PLI)
            if sim_max > 0:
                sim_PLI = sim_PLI / sim_max
        if threshold_sim:
            sim_PLI_flat = sim_PLI.flatten()
            sim_PLI_flat = np.sort(sim_PLI_flat)  # sort lowest to highest
            threshold_val = sim_PLI_flat[int( threshold_sim * sim_PLI_flat.size ) ]
            sim_PLI[sim_PLI < threshold_val] = 0
        if normalize_exp:
            exp_PLI = exp_PLI / np.amax(exp_PLI)
        if threshold_exp:
            exp_PLI_flat = exp_PLI.flatten()
            exp_PLI_flat = np.sort(exp_PLI_flat)  # sort lowest to highest
            threshold_val = exp_PLI_flat[int( threshold_exp * exp_PLI_flat.size ) ]
            exp_PLI[exp_PLI < threshold_val] = 0
        
    if np.any(inds):
        sim_PLI = np.delete(sim_PLI,inds,0)  # submatrix if wanted
        sim_PLI = np.delete(sim_PLI,inds,1)  # submatrix if wanted
    flat_sim_PLI = sim_PLI.flatten()

    # flatten experimental PLI 
    if np.any(inds):
        exp_PLI = np.delete(exp_PLI,inds,0)  # submatrix if wanted
        exp_PLI = np.delete(exp_PLI,inds,1)  # submatrix if wanted
    flat_exp_PLI = exp_PLI.flatten()

    # zero penalty
    if zero_scale > 0:
        zero_inds = np.where(flat_exp_PLI == 0)
        zeros = zero_scale * np.mean(flat_sim_PLI[zero_inds])**(1/2)
    else:
        zeros = 0
    
    # compute sample pearson corelation
    if objective == 'pearson':
        r, _ = pearsonr(flat_sim_PLI, flat_exp_PLI)

        # try only optimizing the nonzero indices
        #inds_int = np.argwhere(flat_exp_PLI > 0)
        #r, _ = pearsonr(flat_sim_PLI[inds_int][:,0], flat_exp_PLI[inds_int][:,0])

    # binarize nonzero indices
    #inds_int = np.argwhere(flat_exp_PLI > 0)
    #flat_exp_PLI[inds_int] = 1
    #flat_sim_PLI = threshold_matrix(flat_sim_PLI, 0.1)
    #inds_int_sim = np.argwhere(flat_sim_PLI > 0)
    #flat_sim_PLI[inds_int_sim] = 1
    ##r = np.sum((flat_sim_PLI - flat_exp_PLI)**2)
    ## Hamming distance
    #r = hamming(flat_sim_PLI, flat_exp_PLI)
    
    # cosine similarity
    if objective == 'cosine':
        #inds_int = np.argwhere(flat_exp_PLI > 0)
        #r = np.dot(flat_sim_PLI[inds_int][:,0], flat_exp_PLI[inds_int][:,0])/ (np.linalg.norm(flat_sim_PLI[inds_int][:,0])*np.linalg.norm(flat_exp_PLI[inds_int][:,0]))
        r = np.dot(flat_sim_PLI, flat_exp_PLI)/ (np.linalg.norm(flat_sim_PLI)*np.linalg.norm(flat_exp_PLI))
    
    # jaccard index
    if objective == 'jaccard':
        r = jaccard_index(flat_sim_PLI, flat_exp_PLI, threshold_exp)    

    #coherence_error = par_coherence * np.abs(np.mean(compute_phase_coherence(x)) - mean_coherence)
    #coherence_error = par_coherence * np.abs(np.mean(compute_phase_coherence_from_complex(compl_signal)) - mean_coherence)
    coherence_error = par_coherence * np.abs(np.mean(compute_phase_coherence_old(x)) - mean_coherence)

    # return negative pearson correlation (maximization)
    return -r+zeros+coherence_error

# ----------------------------------------------------------------
# simulate multi-timescale glioma model 
# INPUT:
#   W0 - numpy array (N,N), initial adjacency matrix
#   DE - a JiTC*DE object, has to be compiled with
#           2*N implicit parameters
#   dyn_y0 - numpy array (#trials, #variables), initial values for DE
#   optional arguments are spreading parameters and
#       integration parameters
# OUTPUT:
#   spread_sol: dictionary, solutions of spreading model
#   dyn_sols: array of dictionaries, solutions of dynamical model
#               at different time points
# ----------------------------------------------------------------
def glioma(W0, DE, dyn_y0, seed=False, seed_amount=0.1, t_spread=False, spread_tspan=False, \
        spread_y0=False, a0=0.75, ai=1, api=1, aii=1, k0=1, c0=1, gamma=0, delta=0.95, \
        rho=10**(-3), a_min=False, a_max=False, b_min=False, a_init=1, b_init=1, \
        degen=False, degen_c=False, method='RK45', spread_max_step=0.125, as_dict=True, \
        spread_atol=10**-6, spread_rtol=10**-3, dyn_atol=10**-6, dyn_rtol=10**-4, \
        dyn_step=1/100, dyn_tspan=(0,10), display=False, trials=1, dyn_cutoff=0):

    # set degen_c to default degradation constant if not set
    if not degen_c:
        degen_c = c0

    # set t_spread if not provided, and add end points if not inluded by user
    if t_spread.size == 0:
        t_spread = [0,spread_tspan[-1]]
    else:
        if 0 not in t_spread:
            t_spread = [0] + t_spread

    # initialize dynamical solutions
    dyn_sols = np.empty((len(t_spread)), dtype='object')

    # if only one initial condition given, repeat it for all trials
    if len(dyn_y0.shape) == 1:
        n_vars = dyn_y0.shape[0]
        new_dyn_y0 = np.empty((trials,n_vars))
        for l in range(trials):
            new_dyn_y0[l,:] = dyn_y0
        dyn_y0 = new_dyn_y0

    # construct laplacian, list of edges, and list of neighours
    N = W0.shape[0]  
    M = 0
    edges = []
    neighbours = [[] for _ in range(N)]
    w0 = []
    for i in range(N):
        for j in range(i+1, N):
            if W0[i,j] != 0:
                M += 1
                edges.append((i,j))
                neighbours[i].append(j)
                neighbours[j].append(i)
                w0.append(W0[i,j])

    # construct spreading initial values, spread_y0
    if not spread_y0:
        u = np.array([a0/ai for _ in range(N)])
        qu = np.array([0 for _ in range(1*N, 2*N)])
        a = np.array([a_init for _ in range(2*N, 3*N)])
        b = np.array([b_init for _ in range(3*N, 4*N)])
        up = np.array([0 for _ in range(4*N, 5*N)])
        spread_y0 = [*u, *qu, *a, *b, *up, *w0]

    # seed tau and beta
    if seed:
        for index in seed:
            seed_index = 4*N+index 
            spread_y0[seed_index] = seed_amount

    # define a and b limits
    if delta:
        a_max = 1 + delta
        a_min = 1 - delta
        b_min = 1 - delta
    elif a_max is not False and a_min is not False and b_min is not False:
        pass
    else:
        print("\nError: You have to either provide a delta or a_min, a_max, and b_min\n")

    # initialize spreading solution
    t0 = t_spread[0]
    empty_array = np.array([[] for _ in range(N)])
    empty_arraym = np.array([[] for _ in range(M)])
    spread_sol = {'t': np.array([]), 'u':empty_array, 'qu':empty_array, \
            'a':empty_array, 'b':empty_array, 'up':empty_array, 'w':empty_arraym, 'w_map': edges, 'rhythms':[(w0, [1 for _ in range(N)], [1 for _ in range(N)], t0)]}

    # spreading dynamics
    def rhs(t, y):
        # set up variables as lists indexed by node k
        u = np.array([y[i] for i in range(N)])
        qu = np.array([y[i] for i in range(1*N, 2*N)])
        a = np.array([y[i] for i in range(2*N, 3*N)])
        b = np.array([y[i] for i in range(3*N, 4*N)])
        up = np.array([y[i] for i in range(4*N, 5*N)])

        # update laplacian from m weights
        w = np.array([y[i] for i in range(5*N, 5*N+M)])
        L = np.zeros((N,N))
        for i in range(M):
            n, m = edges[i]
            # set (n,m) in l
            L[n,m] = -w[i]
            L[m,n] = L[n,m]
            # update (n,n) and (m,m) in l
            L[n,n] += w[i]
            L[m,m] += w[i]

        # check if l is defined correctly
        for i in range(N):
            if abs(sum(L[i,:])) > 10**-10:
                print('L is ill-defined')
                print(sum(L[i,:]))

        L = rho*L
        
        # nodal dynamics
        du, dqu, da, db, dup = [[] for _ in range(5)]
        for k in range(N):
            # index list of node k and its neighbours
            neighbours_k = neighbours[k] + [k]

            # heterodimer dynamics
            duk = sum([-L[k,l]*u[l] for l in neighbours_k]) + a0 - ai*u[k] - aii*u[k]*up[k]
            dupk = sum([-L[k,l]*up[l] for l in neighbours_k]) - api*up[k] + aii*u[k]*up[k]
            ## append
            du.append(duk)
            dup.append(dupk)

            # damage dynamics
            dquk = k0*up[k]*(1-qu[k])
            ## append
            dqu.append(dquk)

            # excitatory-inhibitory dynamics
            if degen and seed[0] == k:
                dak = -degen_c*qu[k]*(a[k]-a_min)
                dbk = -degen_c*qu[k]*(b[k]-b_min)
            else:
                dak = -c0*qu[k]*(a[k]-a_max)
                dbk = -c0*qu[k]*(b[k]-b_min)
            ## append
            da.append(dak)
            db.append(dbk)

        # connecctivity dynamics
        dw = []
        for i in range(M):
            # extract edge
            n, m = edges[i]
            
            # axonopathy dynamcs
            dwi = -gamma*w[i]*(qu[n] + qu[m])
            ## append
            dw.append(dwi)

        # pack right-hand side
        rhs = [*du, *dqu, *da, *db, *dup, *dw]

        return rhs
    
    # SOLVE DYNAMICAL MODEL AT TIME 0
    # measure computational time
    if display:
        start = timer.time()

    # set dynamical model parameters
    dyn_pars = [[*a,*b]]

    # initialize storage for trial simulations
    dyn_x = []
    dyn_y = []

    # solve dynamical model for each trial
    for l in range(trials):
        # set initial values
        dyn_y0_l = dyn_y0[l,:]

        # solve dynamical model at time 0
        print(f'\tSolving dynamical model at time 0 (trial {l+1} of {trials}) ...')
        dyn_sol = solve_dde(DE, dyn_y0_l, W0, t_span=dyn_tspan, step=dyn_step, atol=dyn_atol, rtol=dyn_rtol, parameterss=dyn_pars, cutoff=dyn_cutoff)
        print('\tDone')

        # store each trial
        dyn_x_l = dyn_sol[0]['x'] 
        dyn_y_l = dyn_sol[0]['y'] 
        dyn_x.append(dyn_x_l)
        dyn_y.append(dyn_y_l)

    # store all trials in tuple and add to dyn_sols
    dyn_t = dyn_sol[0]['t']
    dyn_x = np.array( dyn_x )
    dyn_y = np.array( dyn_y )
    dyn_sol_tup = (dyn_t, dyn_x, dyn_y)
    dyn_sols[0] = dyn_sol_tup  
    
    # if only first time-point is given, return empty spread
    if t_spread.size == 1:
        return {}, dyn_sols

    # SOLVE MULTI-SCALE MODEL FOR TIME>0
    for i in range(1,len(t_spread)):
        # SPREADING MODEL TO TIME T
        # set time interval to solve
        t = t_spread[i]
        spread_tspan = (t0, t)

        # solve spreading from time t_(i-1) to t_(i)
        print(f'\n\tSolving spread model for {spread_tspan} ...')
        sol = solve_ivp(rhs, spread_tspan, spread_y0, method=method, max_step=spread_max_step, atol=spread_atol, rtol=spread_rtol)
        print('\tDone.')

        # append spreading solution
        spread_sol['t'] = np.concatenate((spread_sol['t'], sol.t))
        spread_sol['u'] = np.concatenate((spread_sol['u'], sol.y[0:N,:]), axis=1)
        spread_sol['qu'] = np.concatenate((spread_sol['qu'], sol.y[1*N:2*N,:]), axis=1)
        spread_sol['a'] = np.concatenate((spread_sol['a'], sol.y[2*N:3*N,:]), axis=1)
        spread_sol['b'] = np.concatenate((spread_sol['b'], sol.y[3*N:4*N,:]), axis=1)
        spread_sol['up'] = np.concatenate((spread_sol['up'], sol.y[4*N:5*N,:]), axis=1)
        spread_sol['w'] = np.concatenate((spread_sol['w'], sol.y[5*N:5*N+M,:]), axis=1)

        # extract the parameters for the dynamic model
        a = sol.y[2*N:3*N,-1]
        b = sol.y[3*N:4*N,-1]
        w = sol.y[5*N:5*N+M,-1]

        # construct adjacency matrix at time t
        W_t = np.zeros((N,N))
        for j in range(M):
            n, m = edges[j]
            weight = w[j]
            W_t[n,m] = weight
            W_t[m,n] = weight

        # append dynamic model parameters to rhythms list
        rhythms_i = (W_t, a, b, t)
        spread_sol['rhythms'].append(rhythms_i)

        # DYNAMICAL MODEL AT TIME T
        # set dynamical model parameters
        dyn_pars = [[*a,*b]]

        # initialize storage for trial simulations
        dyn_x = []
        dyn_y = []

        # solve dynamical model for each trial
        for l in range(trials):
            # set initial values
            dyn_y0_l = dyn_y0[l,:]

            # solve dynamical model at time 0
            print(f'\tSolving dynamical model at time {t} (trial {l+1} of {trials}) ...')
            dyn_sol = solve_dde(DE, dyn_y0_l, W0, t_span=dyn_tspan, step=dyn_step, atol=dyn_atol, rtol=dyn_rtol, parameterss=dyn_pars, cutoff=dyn_cutoff)
            print('\tDone.')

            # store each trial
            dyn_x_l = dyn_sol[0]['x'] 
            dyn_y_l = dyn_sol[0]['y'] 
            dyn_x.append(dyn_x_l)
            dyn_y.append(dyn_y_l)

        # store all trials in tuple and add to dyn_sols
        dyn_t = dyn_sol[0]['t']
        dyn_x = np.array( dyn_x )
        dyn_y = np.array( dyn_y )
        dyn_sol_tup = (dyn_t, dyn_x, dyn_y)
        dyn_sols[i] = dyn_sol_tup  

        # in the future, potential feedback changes from dynamics to spreading here
        # ->

        # update spreading initial values, spread_y0, and start of simulation, t0
        spread_y0 = sol.y[:,-1]
        t0 = t

    # display computational time
    if display:
        end = timer.time()
        print(f'\nElapsed time for glioma simulations: {end-start} seconds\nElapsed time per time step: {round((end-start)/len(t_spread),4)}')

    # done
    return spread_sol, dyn_sols
    

# ----------------------------------------------------------------
# simulate multi-timescale alzheimer's model 
# INPUT:
#   W0 - numpy array (N,N), initial adjacency matrix
#   DE - a JiTC*DE object, has to be compiled with
#           2*N implicit parameters
#   dyn_y0 - numpy array (#trials, #variables), initial values for DE
#   optional arguments are spreading parameters and
#       integration parameters
# OUTPUT:
#   spread_sol: dictionary, solutions of spreading model
#   dyn_sols: array of dictionaries, solutions of dynamical model
#               at different time points
# ----------------------------------------------------------------
def alzheimer(W0, DE, dyn_y0, tau_seed=False, beta_seed=False, seed_amount=0.1, t_spread=False, \
        spread_tspan=False, \
        spread_y0=False, a0=0.75, ai=1, api=1, aii=1, b0=1, bi=1, bii=1, biii=1, gamma=0, delta=0.95, \
        bpi=1, c1=1, c2=1, c3=1, k1=1, k2=1, k3=1, c_init=0, c_min=0,
        rho=10**(-3), a_min=False, a_max=False, b_min=False, a_init=1, b_init=1, \
        freqss=np.empty([1,1]), method='RK45', spread_max_step=0.125, as_dict=True, \
        spread_atol=10**-6, spread_rtol=10**-3, dyn_atol=10**-6, dyn_rtol=10**-4, \
        dyn_step=1/100, dyn_tspan=(0,10), display=False, trials=1, SDE=False,  \
        normalize_row=False, dyn_cutoff=0, feedback=False, kf=1, bii_max=2, adaptive=False):
    # imports
    from math import e

    # set t_spread if not provided, and add end points if not inluded by user
    if t_spread.size == 0:
        t_spread = [0,spread_tspan[-1]]
    else:
        if 0 not in t_spread:
            t_spread = [0] + t_spread
    Ts_final = t_spread[-1]

    # initialize dynamical solutions
    #dyn_sols = np.empty((len(t_spread)), dtype='object')
    dyn_sols = []  

    # if only one initial condition given, repeat it for all trials
    if len(dyn_y0.shape) == 1:
        n_vars = dyn_y0.shape[0]
        new_dyn_y0 = np.empty((trials,n_vars))
        for l in range(trials):
            new_dyn_y0[l,:] = dyn_y0
        dyn_y0 = new_dyn_y0

    # construct laplacian, list of edges, and list of neighours
    N = W0.shape[0]  
    M = 0
    edges = []
    neighbours = [[] for _ in range(N)]
    w0 = []
    for i in range(N):
        for j in range(i+1, N):
            if W0[i,j] != 0:
                M += 1
                edges.append((i,j))
                neighbours[i].append(j)
                neighbours[j].append(i)
                w0.append(W0[i,j])

    # construct spreading initial values, spread_y0
    if not spread_y0:
        u = np.array([a0/ai for _ in range(N)])
        up = np.array([0 for _ in range(N)])
        v = np.array([b0/bi for _ in range(N)])
        vp = np.array([0 for _ in range(N)])
        qu = np.array([0 for _ in range(N)])
        qv = np.array([0 for _ in range(N)])
        a = np.array([a_init for _ in range(N)])
        b = np.array([b_init for _ in range(N)])
        c = np.array([c_init for _ in range(N)])
        spread_y0 = [*u, *up, *v, *vp, *qu, *qv, *a, *b, *c, *w0]

    # seed tau and beta
    if beta_seed:
        for index in beta_seed:
            beta_index = N+index
            if seed_amount:
                spread_y0[beta_index] = seed_amount
            else:
                spread_y0[beta_index] = (10**(-2)/len(beta_seed))*a0/ai
    if tau_seed:
        for index in tau_seed:
            tau_index = 3*N+index 
            if seed_amount:
                spread_y0[tau_index] = seed_amount
            else:
                spread_y0[tau_index] = (10**(-2)/len(tau_seed))*b0/bi

    # define a and b limits
    if delta:
        a_max = 1 + delta
        a_min = 1 - delta
        b_min = 1 - delta
    elif a_max is not False and a_min is not False and b_min is not False:
        pass
    else:
        print("\nError: You have to either provide a delta or a_min, a_max, and b_min\n")

    # make pf a list (necessary, in case of feedback)
    pf = np.ones((N))

    # initialize spreading solution
    t0 = t_spread[0]
    empty_array = np.array([[] for _ in range(N)])
    empty_arraym = np.array([[] for _ in range(M)])
    spread_sol = {'t': np.array([]), 'u':empty_array, 'up':empty_array, 'v':empty_array, \
                     'vp':empty_array, 'qu':empty_array, 'qv':empty_array, 'a':empty_array, \
                     'b':empty_array, 'c':empty_array, 'w':empty_arraym, 'w_map': edges, \
                     'rhythms':[(w0, [1 for _ in range(N)], [1 for _ in range(N)], t0)], \
                     'pf':np.transpose(np.array([pf])), 'disc_t':[0]}

    # spreading dynamics
    def rhs(t, y):
        # set up variables as lists indexed by node k
        u = np.array([y[i] for i in range(N)])
        up = np.array([y[i] for i in range(N, 2*N)])
        v = np.array([y[i] for i in range(2*N, 3*N)])
        vp = np.array([y[i] for i in range(3*N, 4*N)])
        qu = np.array([y[i] for i in range(4*N, 5*N)])
        qv = np.array([y[i] for i in range(5*N, 6*N)])
        a = np.array([y[i] for i in range(6*N, 7*N)])
        b = np.array([y[i] for i in range(7*N, 8*N)])
        c = np.array([y[i] for i in range(8*N, 9*N)])

        # update laplacian from m weights
        w = np.array([y[i] for i in range(9*N, 9*N+M)])
        L = np.zeros((N,N))
        for i in range(M):
            n, m = edges[i]
            # set (n,m) in l
            L[n,m] = -w[i]
            L[m,n] = L[n,m]
            # update (n,n) and (m,m) in l
            L[n,n] += w[i]
            L[m,m] += w[i]

        # check if l is defined correctly
        for i in range(N):
            if abs(sum(L[i,:])) > 10**-10:
                print('L is ill-defined')
                print(sum(L[i,:]))
    
        # scale Laplacian by diffusion constant
        L = rho*L
        
        # nodal dynamics
        du, dup, dv, dvp, dqu, dqv, da, db, dc = [[] for _ in range(9)]
        for k in range(N):
            # index list of node k and its neighbours
            neighbours_k = neighbours[k] + [k]

            # heterodimer dynamics
            duk = sum([-L[k,l]*u[l] for l in neighbours_k]) + a0 - ai*u[k] - aii*u[k]*up[k]
            dupk = sum([-L[k,l]*up[l] for l in neighbours_k]) - api*up[k] + aii*u[k]*up[k]
            dvk = pf[k]*sum([-L[k,l]*v[l] for l in neighbours_k]) + b0 - bi*v[k] \
                     - bii*v[k]*vp[k] - biii*up[k]*v[k]*vp[k]
            dvpk = pf[k]*sum([-L[k,l]*vp[l] for l in neighbours_k]) - bpi*vp[k] \
                     + bii*v[k]*vp[k] + biii*up[k]*v[k]*vp[k]
            ## append
            du.append(duk)
            dup.append(dupk)
            dv.append(dvk)
            dvp.append(dvpk)

            # damage dynamics
            dquk = k1*up[k]*(1-qu[k])
            dqvk = k2*vp[k]*(1-qv[k]) + k3*up[k]*vp[k]
            ## append
            dqu.append(dquk)
            dqv.append(dqvk)

            # excitatory-inhibitory dynamics
            dak = c1*qu[k]*(a_max-a[k])*(a[k]-a_min) - c2*qv[k]*(a[k]-a_min)
            dbk = -c3*qu[k]*(b[k]-b_min)
            dck = -c3*qu[k]*(c[k]-c_min)
            ## append
            da.append(dak)
            db.append(dbk)
            dc.append(dck)

        # connecctivity dynamics
        dw = []
        for i in range(M):
            # extract edge
            n, m = edges[i]
            
            # axonopathy dynamcs
            dwi = -gamma*w[i]*(qv[n] + qv[m])
            ## append
            dw.append(dwi)

        # pack right-hand side
        rhs = [*du, *dup, *dv, *dvp, *dqu, *dqv, *da, *db, *dc, *dw]

        return rhs 

    # measure computational time
    if display:
        start = timer.time()

    # set initial dynamical model parameters
    W_t = W0

    # SOLVE MULTI-SCALE MODEL FOR TIME>0
    t = 0;  i = 0 
    while t < Ts_final + 1:
        # SOLVE DYNAMICAL MODEL AT T0
        # initialize storage for trial simulations
        dyn_x = []
        dyn_y = []

        # solve dynamical model for each trial
        for l in range(trials):
            # set initial values
            dyn_y0_l = dyn_y0[l,:]
    
            # update dynamical parameters
            if freqss.size > 0:
                freqs = freqss[:,l] 
                dyn_pars = [[*a, *b, *freqs]]
            else:
                dyn_pars = [[*a, *b]]

            # if told, normalize adj. matrix
            if normalize_row:
                for n in range(N):
                    W_t[n,:] = W_t[n,:] / np.sum(W_t[n,:])

            # solve dynamical model at time 0
            print(f'\tSolving dynamical model at time {t0} (trial {l+1} of {trials}) ...')
            if SDE:
                dyn_sol = solve_sde(DE, dyn_y0_l, W_t, t_span=dyn_tspan, step=dyn_step,  \
                     atol=dyn_atol, rtol=dyn_rtol, parameterss=dyn_pars, cutoff=dyn_cutoff)
            else:
                dyn_sol = solve_dde(DE, dyn_y0_l, W_t, t_span=dyn_tspan, step=dyn_step, \
                     atol=dyn_atol, rtol=dyn_rtol, parameterss=dyn_pars, cutoff=dyn_cutoff)
            print('\tDone')

            # store each trial
            dyn_x_l = dyn_sol[0]['x'] 
            dyn_y_l = dyn_sol[0]['y'] 
            dyn_x.append(dyn_x_l)
            dyn_y.append(dyn_y_l)

        # store all trials in tuple and add to dyn_sols
        dyn_t = dyn_sol[0]['t']
        dyn_x = np.array( dyn_x )
        dyn_y = np.array( dyn_y )
        dyn_sol_tup = (dyn_t, dyn_x, dyn_y)
        dyn_sols.append(dyn_sol_tup)
        #dyn_sols[i] = dyn_sol_tup  

        # SPREADING MODEL FROM T0 to T
        # if only one time-point, return the spreading initial conditions
        if len(t_spread) == 1:
            print('\tOnly one time point in spreading simulation')
            spread_sol['t'] = np.concatenate((spread_sol['t'], [0]))
            spread_sol['u'] = np.concatenate((spread_sol['u'], np.reshape(spread_y0[0:N], (N,1))), \
                                                 axis=1)
            spread_sol['up'] = np.concatenate((spread_sol['up'], np.reshape(spread_y0[N:2*N], (N,1))), \
                                                 axis=1)
            spread_sol['v'] = np.concatenate((spread_sol['v'], np.reshape(spread_y0[2*N:3*N], (N,1))), \
                                                 axis=1)
            spread_sol['vp'] = np.concatenate((spread_sol['vp'], np.reshape(spread_y0[3*N:4*N], \
                                                 (N,1))), axis=1)
            spread_sol['qu'] = np.concatenate((spread_sol['qu'], np.reshape(spread_y0[4*N:5*N], \
                                                (N,1))), axis=1)
            spread_sol['qv'] = np.concatenate((spread_sol['qv'], np.reshape(spread_y0[5*N:6*N], \
                                                (N,1))), axis=1)
            spread_sol['a'] = np.concatenate((spread_sol['a'], np.reshape(spread_y0[6*N:7*N], \
                                                (N,1))), axis=1)
            spread_sol['b'] = np.concatenate((spread_sol['b'], np.reshape(spread_y0[7*N:8*N], \
                                                (N,1))), axis=1)
            spread_sol['c'] = np.concatenate((spread_sol['c'], np.reshape(spread_y0[8*N:9*N], \
                                                (N,1))), axis=1)
            spread_sol['w'] = np.concatenate((spread_sol['w'], np.reshape(spread_y0[9*N:9*N+M], \
                                                (M,1))), axis=1)
        # end simulation at last time point
        if t >= Ts_final:
            break

        # set time interval to solve (if adaptive, analyze dynamics here)
        if feedback:
            mods = (dyn_x_l**2 + dyn_y_l**2)**(1/2) 
            avg_mod = np.mean(mods, axis=1) 
            if t0==0:
                mod0 = np.mean(avg_mod)
                pf_0 = pf - 1e-5
            if adaptive:
                eqs = kf*(-mod0+avg_mod-pf+pf_0)
                funcs = 1 / (kf*(mod0 + pf - pf_0))
                step_size = np.amin( funcs )
                t = t + step_size
                print(f'\t\tAdaptive step size = {step_size}')
        if not adaptive:
            t = t_spread[i+1]
        spread_tspan = (t0, t)

        # solve spreading from time t_(i-1) to t_(i)
        print(f'\n\tSolving spread model for {spread_tspan} ...')
        sol = solve_ivp(rhs, spread_tspan, spread_y0, method=method, \
                         max_step=spread_max_step, atol=spread_atol, rtol=spread_rtol)
        print('\tDone.')

        # append spreading solution
        spread_sol['t'] = np.concatenate((spread_sol['t'], sol.t))
        spread_sol['u'] = np.concatenate((spread_sol['u'], sol.y[0:N,:]), axis=1)
        spread_sol['up'] = np.concatenate((spread_sol['up'], sol.y[N:2*N,:]), axis=1)
        spread_sol['v'] = np.concatenate((spread_sol['v'], sol.y[2*N:3*N,:]), axis=1)
        spread_sol['vp'] = np.concatenate((spread_sol['vp'], sol.y[3*N:4*N,:]), axis=1)
        spread_sol['qu'] = np.concatenate((spread_sol['qu'], sol.y[4*N:5*N,:]), axis=1)
        spread_sol['qv'] = np.concatenate((spread_sol['qv'], sol.y[5*N:6*N,:]), axis=1)
        spread_sol['a'] = np.concatenate((spread_sol['a'], sol.y[6*N:7*N,:]), axis=1)
        spread_sol['b'] = np.concatenate((spread_sol['b'], sol.y[7*N:8*N,:]), axis=1)
        spread_sol['c'] = np.concatenate((spread_sol['c'], sol.y[8*N:9*N,:]), axis=1)
        spread_sol['w'] = np.concatenate((spread_sol['w'], sol.y[9*N:9*N+M,:]), axis=1)
        spread_sol['disc_t'].append(t)

        # extract the parameters for the dynamic model
        a = sol.y[6*N:7*N,-1]
        b = sol.y[7*N:8*N,-1]
        w = sol.y[9*N:9*N+M,-1]

        # construct adjacency matrix at time t
        W_t = np.zeros((N,N))
        for j in range(M):
            n, m = edges[j]
            weight = w[j]
            W_t[n,m] = weight
            W_t[m,n] = weight

        # append dynamic model parameters to rhythms list
        rhythms_i = (W_t, a, b, t)
        spread_sol['rhythms'].append(rhythms_i)
        
        # in the future, potential feedback changes from dynamics to spreading here
        if feedback:
            # update parameters
            t_res = t - t0
            # euler
            #pf = pf + t_res * kf * (pf - pf_0) * ((avg_mod - mod0) - (pf - pf_0))
            # RK4
            rk1 = kf * (pf - pf_0) * ((avg_mod - mod0) - (pf - pf_0))
            rk2 = kf * ((pf+t_res*rk1/2) - pf_0) * ((avg_mod - mod0) - ((pf+t_res*rk1/2) - pf_0))
            rk3 = kf * ((pf+t_res*rk2/2) - pf_0) * ((avg_mod - mod0) - ((pf+t_res*rk2/2) - pf_0))
            rk4 = kf * ((pf+t_res*rk3) - pf_0) * ((avg_mod - mod0) - ((pf+t_res*rk3) - pf_0))
            pf = pf + 1/6 * (rk1 + 2*rk2 + 2*rk3 + rk4) * t_res
            print(pf)
            # append parameters
            pf_save = np.transpose(np.array([pf]))  # need correct np dimensions
            spread_sol['pf'] = np.concatenate((spread_sol['pf'], pf_save), axis=1)

        # update spreading initial values, spread_y0, and start of simulation, t0
        spread_y0 = sol.y[:,-1]
        t0 = t
        i += 1

    # display computational time
    if display:
        end = timer.time()
        print(f'\nElapsed time for alzheimer simulations: {end-start} seconds\nElapsed time per time step: {round((end-start)/len(t_spread),4)}')

    # done
    return spread_sol, dyn_sols


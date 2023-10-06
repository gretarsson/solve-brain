from scipy.integrate import solve_ivp, simps, trapz
from scipy.signal import hilbert
from scipy.fftpack import fft, ifft, rfft, rfftfreq
from scipy.interpolate import interp1d
from math import e, cos, sin, pi, ceil, log10, floor
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pprint import pprint
import random
import seaborn as sns
import numpy as np
import copy
from numba import jit

# -------------------------------------------------------
# Here we model the spreading of A-beta and tau,
# and model its effect on nodal dynamics (ak, bk).
# 
#
# u - amyloid beta, up - pathological amyloid beta
# v - tau, vp - pathological tau, a - excitatory semiaxis
# b - inhibitory semiaxis, q_u - amyloid beta damage,
# q_v - tau damage, w - link weight
#
# Input:
# W0 - numpy array, weighted adjacency matrix (symmetric)
# tau_seed - integer list designating nodes where tau seeds
# beta_seed - integer list designating nodes where A-beta seeds
# t_stamps - times at which W_i is stored
# y0 - 8*N + M list, initial values, if not set
# it resorts to stationary healthy protein levels.
#
# Output:
# The solution per solve_ivp
#
# If as_dict=True (default), a dictionary is
# returned with keys:
# t, u, up, v, vp, qu, qv, a, b, w, w_map, L
# if t_stamps is provided then a list of tuples
# is included with key 'rhythms' containing
# (W, a, b, t) where t is the timestamp
#
# Each variable element is a 2-D list (rows = nodes,
# column = time point). Rows of w are edges, 
# that are mapped with w_map (element i contains
# the tuple associated with row i of w). L is the
# final Laplacian built from w.
# -------------------------------------------------------

def network_spreading(W0, tau_seed=False, beta_seed=False, t_span=(0, 100), t_stamps=False, y0=False, \
        a0=0.75, ai=1, aii=1, api=0.6, b0=0.75, bi=1, bii=1, biii=1, bpi=0.6, a_min=False, a_max=False, \
        b_min=False, c_min=0, gamma=0.125, k1=0.25, k2=0.25, k3=0, rho=0.01, delta=0.95, c1=0.2, \
        c2=0.4, c3=0.2, \
        seed_amount=False, method='radau', max_step=0.125, as_dict=True, atol=10**-6, rtol=10**-3, \
        a_init=1, b_init=1, c_init=0):
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

    # set t_stamps if not provided, and add end points if not inluded by user
    if t_stamps.size == 0:
        t_stamps = [0,t_span[-1]]
    else:
        if 0 not in t_stamps:
            t_stamps = [0] + t_stamps

    # construct initial values, y0
    if not y0:
        u = np.array([a0/ai for _ in range(N)])
        up = np.array([0 for _ in range(N)])
        v = np.array([b0/bi for _ in range(N)])
        vp = np.array([0 for _ in range(N)])
        qu = np.array([0 for _ in range(N)])
        qv = np.array([0 for _ in range(N)])
        a = np.array([a_init for _ in range(N)])
        b = np.array([b_init for _ in range(N)])
        c = np.array([c_init for _ in range(N)])
        y0 = [*u, *up, *v, *vp, *qu, *qv, *a, *b, *c, *w0]

    # seed tau and beta
    if beta_seed:
        for index in beta_seed:
            beta_index = N+index
            if seed_amount:
                y0[beta_index] = seed_amount
            else:
                y0[beta_index] = (10**(-2)/len(beta_seed))*a0/ai
    if tau_seed:
        for index in tau_seed:
            tau_index = 3*N+index 
            if seed_amount:
                y0[tau_index] = seed_amount
            else:
                y0[tau_index] = (10**(-2)/len(tau_seed))*b0/bi

    # define a and b limits
    if delta:
        a_max = 1 + delta
        a_min = 1 - delta
        b_min = 1 - delta
    elif a_max is not False and a_min is not False and b_min is not False:
        pass
    else:
        print("\nError: You have to either provide a delta or a_min, a_max, and b_min\n")

    # system dynamics
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

        L = rho*L

        
        # nodal dynamics
        du, dup, dv, dvp, dqu, dqv, da, db, dc = [[] for _ in range(9)]
        for k in range(N):
            # index list of node k and its neighbours
            neighbours_k = neighbours[k] + [k]

            # heterodimer dynamics
            duk = sum([-L[k,l]*u[l] for l in neighbours_k]) + a0 - ai*u[k] - aii*u[k]*up[k]
            dupk = sum([-L[k,l]*up[l] for l in neighbours_k]) - api*up[k] + aii*u[k]*up[k]
            dvk = sum([-L[k,l]*v[l] for l in neighbours_k]) + b0 - bi*v[k] - bii*v[k]*vp[k] - biii*up[k]*v[k]*vp[k]
            dvpk = sum([-L[k,l]*vp[l] for l in neighbours_k]) - bpi*vp[k] + bii*v[k]*vp[k] + biii*up[k]*v[k]*vp[k]
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
    
    # solve system (if t_stamps provided then the solution procedure is done step-wise)
    # initialize
    t0 = t_stamps[0]
    empty_array = np.array([[] for _ in range(N)])
    if M > 1:  # if all weights zero, we get an error 
        empty_arraym = np.array([[] for _ in range(M)])
    else:
        empty_arraym = np.array([[] for _ in range(1)])
    y_dict = {'t': np.array([]), 'u':empty_array, 'up':empty_array, 'v':empty_array, 'vp':empty_array, 'qu':empty_array, \
            'qv':empty_array, 'a':empty_array, 'b':empty_array, 'c':empty_array, 'w':empty_arraym, 'w_map': edges, 'rhythms':[(w0, [1 for _ in range(N)], [1 for _ in range(N)], t0)]}
    for i in range(1,len(t_stamps)):
        # solve from time t_(i-1) to t_(i)
        t = t_stamps[i]
        t_span = (t0, t)
        sol = solve_ivp(rhs, t_span, y0, method=method, max_step=max_step, atol=atol, rtol=rtol)

        # append solution
        y_dict['t'] = np.concatenate((y_dict['t'], sol.t))
        y_dict['u'] = np.concatenate((y_dict['u'], sol.y[0:N,:]), axis=1)
        y_dict['up'] = np.concatenate((y_dict['up'], sol.y[N:2*N,:]), axis=1)
        y_dict['v'] = np.concatenate((y_dict['v'], sol.y[2*N:3*N,:]), axis=1)
        y_dict['vp'] = np.concatenate((y_dict['vp'], sol.y[3*N:4*N,:]), axis=1)
        y_dict['qu'] = np.concatenate((y_dict['qu'], sol.y[4*N:5*N,:]), axis=1)
        y_dict['qv'] = np.concatenate((y_dict['qv'], sol.y[5*N:6*N,:]), axis=1)
        y_dict['a'] = np.concatenate((y_dict['a'], sol.y[6*N:7*N,:]), axis=1)
        y_dict['b'] = np.concatenate((y_dict['b'], sol.y[7*N:8*N,:]), axis=1)
        if c_init != c_min:
            y_dict['c'] = np.concatenate((y_dict['c'], sol.y[8*N:9*N,:]), axis=1)
        if M > 1:  # if all weights zero we get an error
            y_dict['w'] = np.concatenate((y_dict['w'], sol.y[9*N:9*N+M,:]), axis=1)
        else:
            y_dict['w'] = np.concatenate((y_dict['w'], np.zeros([1,sol.t.size])), axis=1)

        # construct w and add to list ws
        a = sol.y[6*N:7*N,-1]
        b = sol.y[7*N:8*N,-1]
        c = sol.y[8*N:9*N,-1]
        #qu = sol.y[4*n:5*n,-1]
        #qu = sol.y[5*n:6*n,-1]
        w = sol.y[9*N:9*N+M,-1]

        w_t = np.zeros((N,N))
        for i in range(M):
            n, m = edges[i]
            weight = w[i]
            w_t[n,m] = weight
            w_t[m,n] = weight

        # check if we include c or not
        if c_init == c_min:
            rhythms_i = (w_t, a, b, t)
        else:
            rhythms_i = (w_t, a, b, c, t)
        y_dict['rhythms'].append(rhythms_i)

        # update initial values, y0, and start of simulation, t0
        y0 = sol.y[:,-1]
        t0 = t

    # done
    return y_dict

# ------------------------------------------------------------------------------
def glioma_spreading(W0, seed=False, seed_amount=0.1, t_span=(0, 100), t_stamps=False, \
        y0=False, a0=0.75, ai=1, api=1, aii=1, k0=1, c0=1, gamma=0, delta=0.95, rho=10**(-3), a_min=False, a_max=False, b_min=False, a_init=1, b_init=1, \
        degen=False, degen_c=False, method='radau', max_step=0.125, as_dict=True, atol=10**-6, rtol=10**-3):

    # set degen_c to default degrataoin constant if not set
    if not degen_c:
        degen_c = c0
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


    # set t_stamps if not provided, and add end points if not inluded by user
    if t_stamps.size == 0:
        t_stamps = [0,t_span[-1]]
    else:
        if 0 not in t_stamps:
            t_stamps = [0] + t_stamps
        #if t_span[-1] not in t_stamps:
        #    t_stamps = t_stamps + [t_span[-1]]

    # construct initial values, y0
    if not y0:
        u = np.array([a0/ai for _ in range(N)])
        qu = np.array([0 for _ in range(1*N, 2*N)])
        a = np.array([a_init for _ in range(2*N, 3*N)])
        b = np.array([b_init for _ in range(3*N, 4*N)])
        up = np.array([0 for _ in range(4*N, 5*N)])
        y0 = [*u, *qu, *a, *b, *up, *w0]

    # seed tau and beta
    if seed:
        for index in seed:
            seed_index = 4*N+index 
            y0[seed_index] = seed_amount

    # define a and b limits
    if delta:
        a_max = 1 + delta
        a_min = 1 - delta
        b_min = 1 - delta
    elif a_max is not False and a_min is not False and b_min is not False:
        pass
    else:
        print("\nError: You have to either provide a delta or a_min, a_max, and b_min\n")

    # system dynamics
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
    
    # solve system (if t_stamps provided then the solution procedure is done step-wise)
    # initialize
    t0 = t_stamps[0]
    empty_array = np.array([[] for _ in range(N)])
    empty_arraym = np.array([[] for _ in range(M)])
    y_dict = {'t': np.array([]), 'u':empty_array, 'qu':empty_array, \
            'a':empty_array, 'b':empty_array, 'up':empty_array, 'w':empty_arraym, 'w_map': edges, 'rhythms':[(w0, [1 for _ in range(N)], [1 for _ in range(N)], t0)]}
    for i in range(1,len(t_stamps)):
        # solve from time t_(i-1) to t_(i)
        t = t_stamps[i]
        t_span = (t0, t)
        sol = solve_ivp(rhs, t_span, y0, method=method, max_step=max_step, atol=atol, rtol=rtol)

        # append solution
        y_dict['t'] = np.concatenate((y_dict['t'], sol.t))
        y_dict['u'] = np.concatenate((y_dict['u'], sol.y[0:N,:]), axis=1)
        y_dict['qu'] = np.concatenate((y_dict['qu'], sol.y[1*N:2*N,:]), axis=1)
        y_dict['a'] = np.concatenate((y_dict['a'], sol.y[2*N:3*N,:]), axis=1)
        y_dict['b'] = np.concatenate((y_dict['b'], sol.y[3*N:4*N,:]), axis=1)
        y_dict['up'] = np.concatenate((y_dict['up'], sol.y[4*N:5*N,:]), axis=1)
        y_dict['w'] = np.concatenate((y_dict['w'], sol.y[5*N:5*N+M,:]), axis=1)

        # construct w and add to list ws
        a = sol.y[2*N:3*N,-1]
        b = sol.y[3*N:4*N,-1]
        w = sol.y[5*N:5*N+M,-1]

        w_t = np.zeros((N,N))
        for i in range(M):
            n, m = edges[i]
            weight = w[i]
            w_t[n,m] = weight
            w_t[m,n] = weight

        rhythms_i = (w_t, a, b, t)
        #rhythms_i = (w_t, qu, qv, t)
        y_dict['rhythms'].append(rhythms_i)

        # update initial values, y0, and start of simulation, t0
        y0 = sol.y[:,-1]
        t0 = t

    # done
    return y_dict
# -----------------------------------------------------------------------
# Here we input a weighted adjacency matrix and performs a
# graphic simulation of elliptic neural mass models
#
# Required input:
# W - N-by-N array-like matrix, weigthed adjacency matrix
# a, b - lists of semiaxes parameters
# delays - N-by-N array-like matrix, with delay from node n to m
# -----------------------------------------------------------------------

def network_rhythms(W, a=False, b=False, delays=False, t_span=(0,100), kappa=10, w=False, decay=-0.01, step=10**-4, random_init=True, atol=10**-6):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t
    import symengine as sym

    # find threshold
    W_max = np.amax(W)
    threshold = 0.000*W_max

    # find neighbours of each node
    N = W.shape[0]
    neighbours = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            #if W[i,j] != 0:
            #    neighbours[i].append(j)
            #    neighbours[j].append(i)
            if W[i,j] > threshold:
                neighbours[i].append(j)
                neighbours[j].append(i)
    # debug
    n_of_links = 0
    for j in neighbours:
        n_of_links += len(j)

    # check unset parameters
    if not w.any():
        w = [10*(2*pi) for _ in range(N)]

    # define sigmoidal function
    S = lambda x: 1/(1+e**(-x))

    # organize variables
    Re = np.array([y(i) for i in range(N)])
    Im = np.array([y(i) for i in range(N, 2*N)])

    # right-hand side for each node
    dRe = []
    dIm = []
    for k in range(N):
        # define input to node
        afferent_input = 0
        count = 0
        for j in neighbours[k]:
            afferent_input += W[j,k]*y(j, t-delays[j,k])
            #count += 1
            #if count == 40:
            #    break
            #afferent_input += W[j,k]*y(j, t-0.001)
            #afferent_input += W[j,k]*y(j, t)
        
        # dynamics of node k
        dRe_k = decay*Re[k] - w[k]*(a[k]/b[k])*Im[k] - Re[k]*(Re[k]**2/a[k]**2 + Im[k]**2/b[k]**2) + kappa*S(afferent_input)
        dIm_k = decay*Im[k] + w[k]*(b[k]/a[k])*Re[k] - Im[k]*(Re[k]**2/a[k]**2 + Im[k]**2/b[k]**2)

        # append to rhs
        dRe.append(dRe_k)
        dIm.append(dIm_k)

    # pack together the right-hand side
    dydt = [*dRe, *dIm]

    # feed the rhs to the delay-DDE solver
    DDE = jitcdde(dydt) 

    # set up initial values at random positions at their intrinsic ellipse
    if random_init:
        theta0 = [random.uniform(0, 2*3.14) for _ in range(N)]
        R0 = [random.uniform(0, 1) for _ in range(N)]
    else:
        theta0 = [0 for _ in range(N)]
    # my own intial condition
    #y0_Re = [a[k]*abs(decay)**0.5*sym.cos(theta0[k]) for k in range(N)]
    #y0_Im = [b[k]*abs(decay)**0.5*sym.sin(theta0[k]) for k in range(N)]
    ## Bick Goriely initial conditions
    #y0_Re = [R0[k]*sym.cos(theta0[k]) for k in range(N)]
    #y0_Im = [R0[k]*sym.sin(theta0[k]) for k in range(N)]
    y0_Re = [1 for k in range(N)]
    y0_Im = [0 for k in range(N)]
    y0 = [*y0_Re, *y0_Im]

    # use constant past
    DDE.constant_past(y0)

    # integration parameters
    DDE.set_integration_parameters(rtol=0,atol=atol, first_step=10**-5, max_step=10**-2)

    # numerical issue handling
    DDE.step_on_discontinuities()
    #DDE.adjust_diff()
    #DDE.integrate_blindly(np.amax(delays), step=0)

    # solve system
    data = []
    t = []
    for time in np.arange(DDE.t, DDE.t+t_span[1],  step):
        data.append( DDE.integrate(time) )
        t.append(time)
    data = np.array(data)
    data = np.transpose(data)
    t = np.array(t)

    # organize solution as dictionary
    sol = {}
    sol['t'] = t
    sol['x'] = data[0:N,:]
    sol['y'] = data[N:2*N,:]

    return sol

# -----------------------------------------
# redo network rhythms symbolically
# ----------------------------------------
def network_rhythms_symbolically(W, a=False, b=False, delays=False, t_span=(0,100), kappa=10, w=False, decay=-0.01, step=10**-4, random_init=True, atol=10**-6, rtol=10**-4):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t
    #from jitcsde import jitcsde, y
    import symengine as sym

    # find neighbours of each node
    N = W.shape[0]
    neighbours = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] != 0:
                neighbours[i].append(j)
                neighbours[j].append(i)

    # define generator of rhs
    def neural_mass():
        for k in range(N):
            # define input to node
            #afferent_input = sum( W[j,k] * y(2*j, t-delays[j,k]) for j in neighbours[k] )
            #c = 1
            #o = 0
            ##taux = 1 + 0.5*(1-b[k])
            ##tauy = 1 + 0.5*(1-a[k])
            #
            ## dynamics of node k
            #yield decay*y(2*k+0) - w[k]*(a[k]/b[k])*y(2*k+1) - y(2*k+0)*(y(2*k+0)**2/a[k]**2 + y(2*k+1)**2/b[k]**2) + kappa * 1 / (1 + e**(-c*(afferent_input-o)))
            #yield decay*y(2*k+1) + w[k]*(b[k]/a[k])*y(2*k+0) - y(2*k+1)*(y(2*k)**2/a[k]**2 + y(2*k+1)**2/b[k]**2)

            #yield 1/taux * (decay*y(2*k) - w[k]*y(2*k+1) - y(2*k)*(y(2*k)**2 + y(2*k+1)**2) + kappa * 1 / (1 + e**(-afferent_input)))
            #yield 1/tauy * (decay*y(2*k+1) + w[k]*y(2*k) - y(2*k+1)*(y(2*k)**2 + y(2*k+1)**2))
            ## --------------------------------------------------------------------
            # Wilson-Cowan
            taux = 0.041  # alpha wave 
            tauy = 0.041 
            tauz = 0.267  

            Cxx = 24
            Cxy = -20 
            Cxz = 0

            Cyy = 0 
            Cyx = 35
            Cyz = 0

            Czx = 0
            Czy = 0
            Czz = 0

            P = 0.5 - 0.5*(1-a[k])
            Q = -2 - 0.5*(1-b[k])
            R = 0

            # epilepsy
            #taux = 0.03  # alpha wave 
            #tauy = 0.03 
            #tauz = 0.61  

            #Cxx = 24
            #Cxy = -20 
            #Cxz = -20

            #Cyy = 0 
            #Cyx = 35
            #Cyz = 0

            #Czx = 35
            #Czy = 0
            #Czz = 0

            #P = 0.5 - 0.5*(1-a[k])
            #Q = -2 - 0.5*(1-b[k])
            #R = -2
            ## --

            h = 1
            aS = 1
            theta = 4

            aff_inp = kappa*sum( W[j,k] * y(3*j+0, t-delays[j,k]) for j in neighbours[k] )
            #aff_inp = kappa*sum( W[j,k] * y(3*j+0) for j in neighbours[k] )
            x_inp = (Cxx*y(3*k+0) + Cxy*y(3*k+1) + Cxz*y(3*k+2) + P + aff_inp)
            y_inp = (Cyx*y(3*k+0) + Cyy*y(3*k+1) + Cyz*y(3*k+2) + Q)
            z_inp = (Czx*y(3*k+0) + Czy*y(3*k+1) + Czz*y(3*k+2) + R)

            Sx = h*(1+e**(-aS*(x_inp - theta)))**-1
            Sy = h*(1+e**(-aS*(y_inp - theta)))**-1
            Sz = h*(1+e**(-aS*(z_inp - theta)))**-1

            yield 1/taux * (-y(3*k+0) + Sx)
            yield 1/tauy * (-y(3*k+1) + Sy)
            #yield 1/tauz * (-y(3*k+2) + Sz)
            yield 0
            ## --------------------------------------------------------------------
            
            ## --------------------------------------------------------------------
            # define input to node
            #afferent_input = sum( 1/2 * W[j,k] * (y(2*j, t-delays[j,k]) + y(2*j, t-delays[j,k]))for j in neighbours[k] )
            #ra = (y(4*k+0)**2/a[k]**2 + y(4*k+1)**2/b[k]**2)**(1/2)
            #rd = (y(4*k+2)**2/a[k]**2 + y(4*k+3)**2/b[k]**2)**(1/2)
            #alpha = -1
            #alphal= 0.5
            #decay_a = decay - alphal*rd            
            #decay_d = decay - alphal*ra            
            #wd = w[k] - 6*(2*pi)
            #c=1
            #o=0
            #
            ## dynamics of node k
            #yield y(4*k+0)*(decay_a+alpha*ra**2) - w[k]*(a[k]/b[k])*y(4*k+1) + kappa * 1 / (1 + e**(-c*(afferent_input - o)))
            #yield y(4*k+1)*(decay_a+alpha*ra**2) + w[k]*(b[k]/a[k])*y(4*k+0)

            #yield y(4*k+2)*(decay_d+alpha*rd**2) - wd*(a[k]/b[k])*y(4*k+3) + kappa * 1 / (1 + e**(-c*(afferent_input - o)))
            #yield y(4*k+3)*(decay_d+alpha*rd**2) + wd*(b[k]/a[k])*y(4*k+2)
            # --------------------------------------------------------------------
            # --------------------------------------------------------------------
            # saddle node v1
            #afferent_input = sum( W[j,k] * y(2*j, t-delays[j,k]) for j in neighbours[k] )
            #p = 2.5
            #R = 1
            #c = 1
            #o = 0
            #r = (y(2*k)**2/a[k]**2 + y(2*k+1)**2/b[k]**2)**(1/2)
            #yield y(2*k)*(decay-r**2)*((r-p)**2 - R) - w[k]*(a[k]/b[k])*y(2*k+1) + kappa * 1 / (1 + e**(-c*(afferent_input - o)))
            #yield y(2*k+1)*(decay-r**2)*((r-p)**2 - R) + w[k]*(b[k]/a[k])*y(2*k)
            # ----------------------------------------------------------------------
            # --------------------------------------------------------------------
            # saddle node v2
            #afferent_input = sum( W[j,k] * y(3*j, t-delays[j,k]) for j in neighbours[k] )
            #afferent_input = sum( W[j,k] * y(3*j) for j in neighbours[k] )
            #p = 3
            #s = 1.5
            #R = 5*sym.sin(y(3*k+2)) - 0
            #c = 1
            #o = 0
            #r = (y(3*k)**2/a[k]**2 + y(3*k+1)**2/b[k]**2)**(1/2)
            #I0 = 0.1

            #yield y(3*k)*(decay-(r-s)**2)*((r-p)**2 - R) - w[k]*(a[k]/b[k])*y(3*k+1) + kappa * 1 / (1 + e**(-c*(afferent_input - o)))
            #yield y(3*k+1)*(decay-(r-s)**2)*((r-p)**2 - R) + w[k]*(b[k]/a[k])*y(3*k)
            #yield 5*(1 - sym.cos(y(3*k+2)) + (1 + sym.cos(y(3*k+2))) * ( 1 / (1 + e**(-c*(afferent_input - o))) - I0))
            # ----------------------------------------------------------------------
            # --------------------------------------------------------------------
            # saddle node v3
            #afferent_input = sum( W[j,k] * y(4*j, t-delays[j,k]) for j in neighbours[k] )
            ##afferent_input = sum( W[j,k] * y(4*j) for j in neighbours[k] )
            #c = 1
            #o = 0
            #r = (y(4*k)**2/a[k]**2 + y(4*k+1)**2/b[k]**2)**(1/2)
            #alpha =  -e**(-1*(y(4*k+2)+1))
            #decay = 0.1
            ##alpha = +1 + y(4*k+2)

            #af = 1
            #bf = 0.1
            ##af = 0.7
            ##bf = 0.8
            #tauv = 0.01
            #tauw = 10*tauv
            #I0 = 5
            #Is = 0
            #R = 1

            #yield y(4*k)*(decay+alpha*r**2) - w[k]*(a[k]/b[k])*y(4*k+1) + kappa * 1 / (1 + e**(-c*(afferent_input - o)))
            #yield y(4*k+1)*(decay+alpha*r**2) + w[k]*(b[k]/a[k])*y(4*k)
            #yield 1/tauv * (y(4*k+2) - y(4*k+2)**3 / 3 - y(4*k+3) + R*(1 / (1 + e**(-c*(afferent_input - o))) + Is - I0))
            #yield 1/tauw * (y(4*k+2) + af - bf*y(4*k+3))
            # ----------------------------------------------------------------------
            # ----------------------------------------------------------------------
            # Jansen-Rit
            #A = 3.25 - 3*(1-a[k])
            #B = 22 - 3*(1-b[k])
            ##A = 3.25 
            ##B = 22  
            #a0 = 100 #100 * (1+0.75*(1-b[k]))
            #b0 = 50  #50 * (1+0.75*(1-a[k]))
            #C = 135
            #C1 = 1*C
            #C2 = 0.8*C
            #C3 = 0.25*C
            #C4 = 0.25*C
            #v_max = 5
            #v0 = 6
            #r = 0.56
            #alpha = 20  # 20
            #f0 = 200  # 110

            #aff_input = alpha * sum ( W[j,k] * v_max * 1 / (1+e**(r*(v0-(y(2*j+1, t-delays[j,k])-y(2*j+2, t-delays[j,k]))))) for j in neighbours[k])


            #yield y(6*k+3)
            #yield y(6*k+4)
            #yield y(6*k+5)
            #yield A*a0*v_max * 1 / (1+e**(r*(v0-(y(6*k+1)-y(6*k+2))))) - 2*a0*y(6*k+3) - a0**2 * y(6*k)
            #yield A*a0* ( f0 + aff_input + C2 * v_max * 1 / (1+e**(r*(v0-C1*y(6*k)))) ) - 2*a0*y(6*k+4) - a0**2 * y(6*k+1)
            #yield B*b0*C4 * v_max * 1 / (1+e**(r*(v0-C3*y(6*k)))) - 2*b0*y(6*k+5) - b0**2 * y(6*k+2) 
            # ----------------------------------------------------------------------
    # stochastic version WC
    mu = 0.01
    def noise(): 
        for k in range(N):
            yield mu #* y(3*k + 0)
            yield mu #* y(3*k + 1)
            yield mu #* y(3*k + 2)

    # set up initial values at random positions at their intrinsic ellipse
    #if random_init:
    #    theta0 = [random.uniform(0, 2*3.14) for _ in range(N)]
    #    R0 = [random.uniform(0, 1) for _ in range(N)]
    #else:
    #    R0 = [1 for _ in range(N)]
    #    theta0 = [0 for _ in range(N)]
    #y0 = np.zeros((2*N))  # normal form
    #for k in range(0,N):
    #    y0[2*k] = R0[k] * cos(theta0[k])
    #    y0[2*k+1] = R0[k] * sin(theta0[k])
    # --------------------------------------------------------------
    # wilson-cowan
    if random_init:
        init = np.random.uniform(0,0.5, size=3*N)
    else:
        init = [0.1 for _ in range(3*N)]
    y0 = np.zeros((3*N))  # normal form
    for k in range(0,N):
        y0[3*k+0] = init[3*k+0]
        y0[3*k+1] = init[3*k+1]
        #y0[3*k+2] = init[3*k+2] * 0.1
        y0[3*k+2] = 0
    #    # --------------------------------------------------------------
    # normal form plus theta
    #if random_init:
    #    theta0 = [random.uniform(0, 2*3.14) for _ in range(N)]
    #    R0 = [random.uniform(0, 1) for _ in range(N)]
    #else:
    #    R0 = [1 for _ in range(N)]
    #    theta0 = [0 for _ in range(N)]
    #y0 = np.zeros((3*N))  # normal form
    #for k in range(0,N):
    #    y0[3*k] = R0[k] * cos(theta0[k])
    #    y0[3*k+1] = R0[k] * sin(theta0[k])
    #    y0[3*k+2] = -sym.pi/2  # plus theta
    # ---------------------------------------------------------
    # --------------------------------------------------------------
    # normal form plus fitzhugh
    #if random_init:
    #    theta0 = [random.uniform(0, 2*3.14) for _ in range(N)]
    #    R0 = [random.uniform(0, 1) for _ in range(N)]
    #else:
    #    R0 = [1 for _ in range(N)]
    #    theta0 = [0 for _ in range(N)]
    #y0 = np.zeros((4*N))  # normal form
    #for k in range(0,N):
    #    y0[4*k] = R0[k] * cos(theta0[k])
    #    y0[4*k+1] = R0[k] * sin(theta0[k])
    #    y0[4*k+2] = 0.1  # plus theta
    #    y0[4*k+3] = 0.5  # plus theta
    # ---------------------------------------------------------
    # --------------------------------------------------------------
    # normal form dual
    #if random_init:
    #    theta0 = [random.uniform(0, 2*3.14) for _ in range(2*N)]
    #    R0 = [random.uniform(0, 1) for _ in range(2*N)]
    #else:
    #    R0 = [1 for _ in range(N)]
    #    theta0 = [0 for _ in range(N)]
    #y0 = np.zeros((4*N))  # normal form
    #for k in range(0,N):
    #    y0[4*k+0] = R0[2*k+0] * cos(theta0[2*k+0])
    #    y0[4*k+1] = R0[2*k+0] * sin(theta0[2*k+0])
    #    y0[4*k+2] = R0[2*k+1] * cos(theta0[2*k+1])
    #    y0[4*k+3] = R0[2*k+1] * sin(theta0[2*k+1])
    # ---------------------------------------------------------
    # jansen rit initial values
    #y0 = np.zeros((6*N))
    #for k in range(N):
    #    if not random_init:
    #        y0[6*k+0] = 0.1
    #        y0[6*k+1] = 0.1
    #        y0[6*k+2] = 0.1
    #        y0[6*k+3] = 0.1
    #        y0[6*k+4] = 0.1
    #        y0[6*k+5] = 0.1
    #    else:
    #        y0[6*k+0] = random.uniform(-5, 5)
    #        y0[6*k+1] = random.uniform(-5, 5)
    #        y0[6*k+2] = random.uniform(-5, 5)
    #        y0[6*k+3] = random.uniform(-5, 5)
    #        y0[6*k+4] = random.uniform(-5, 5)
    #        y0[6*k+5] = random.uniform(-5, 5)
    ## ---------------------------------------------------------

    # set-up DDE configurations
    #DDE = jitcdde(neural_mass, n=2*N)  # normal form
    DDE = jitcdde(neural_mass, n=3*N)  # normal form with theta or WC
    ##DDE = jitcdde(neural_mass, n=4*N)  # normal form with fitzhugh
    ##DDE = jitcdde(neural_mass, n=6*N)  # jansen-rit
    DDE.compile_C(do_cse=True, chunk_size=int(N*2))
    DDE.set_integration_parameters(rtol=rtol,atol=atol)
    DDE.constant_past(y0, time=0.0)
    #DDE.step_on_discontinuities(propagations=0)
    #DDE.integrate_blindly(np.amax(delays), step=step)
    DDE.adjust_diff()

    # stochastic version
    #DDE = jitcsde(neural_mass, noise, n=3*N)  # normal form with theta or WC
    #DDE.set_integration_parameters(rtol=rtol,atol=atol)
    #DDE.set_initial_value(y0,0.0)

    # solve
    data = []
    t = []
    for time in np.arange(DDE.t, DDE.t+t_span[1],  step):
        data.append( DDE.integrate(time) )
        t.append(time)

    data = np.array(data)
    data = np.transpose(data)
    t = np.array(t)

    # organize solution as dictionary (normal form style)
    #sol = {}
    #sol['t'] = t
    #sol['x'] = data[0:2*N:2,:]
    #sol['y'] = data[1:2*N:2,:]

    # organize solution as dictionary (normal form plus theta style, or WC)
    sol = {}
    sol['t'] = t
    sol['x'] = data[0:3*N:3,:]
    sol['y'] = data[1:3*N:3,:]

    # organize solution as dictionary (normal form plus fitzhugh style)
    #sol = {}
    #sol['t'] = t
    #sol['x'] = data[0:4*N:4,:]
    #sol['y'] = data[2:4*N:4,:]

    # organize solution (Jansen-Rit style)
    #sol = {}
    #sol['t'] = t
    #sol['x'] = data[1:6*N:6,:] - data[2:6*N:6,:] 
    #sol['y'] = data[5:6*N:6,:]

    return sol

# --------------------------------------
# solve network of WC symbolically
# ----------------------------------------
def wilson_cowan(W, P=False, Q=False, delays=False, t_span=(0,100), taux=0.013, tauy=0.013, Cxx=24, \
        Cxy=-20, Cyy=0, Cyx=40, h=1, Sa=1, theta=4, kappa=0.4, \
        step=10**-4, random_init=True, atol=10**-6, rtol=10**-4, osc_freqs=False):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t
    import symengine as sym

    # find neighbours of each node
    N = W.shape[0]
    neighbours = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] != 0:
                neighbours[i].append(j)
                neighbours[j].append(i)

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

    # if P or Q not list then make constant array
    if isinstance(P,float) or isinstance(P,int):
        P_val = P
        P = np.full((N),P_val)
    if isinstance(Q,float) or isinstance(Q,int):
        Q_val = Q
        Q = np.full((N),Q_val)

    # define generator of rhs
    def neural_mass():
        for k in range(N):
            aff_inp = kappa*sum( W[j,k] * y(2*j+0, t-delays[j,k]) for j in neighbours[k] )

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

    # set-up DDE configurations
    DDE = jitcdde(neural_mass, n=2*N)  # normal form
    DDE.compile_C(do_cse=True)
    DDE.set_integration_parameters(rtol=rtol,atol=atol)
    DDE.constant_past(y0, time=0.0)
    #DDE.step_on_discontinuities(propagations=0)
    #DDE.integrate_blindly(np.amax(delays), step=step)
    DDE.adjust_diff()

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

    # store solution as dictionary (normal form plus theta style, or WC)
    sol = {}
    sol['t'] = t
    sol['x'] = data[0:2*N:2,:]
    sol['y'] = data[1:2*N:2,:]

    return sol

# -----------------------------------------
# Compute network simulations of hopf normal forms
# ----------------------------------------
def hopf(W, a=False, b=False, delays=False, t_span=(0,100), kappa=10, w=False, decay=-0.01, step=10**-4, random_init=True, atol=10**-6, rtol=10**-4, c=1, o=0):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t
    import symengine as sym

    # find neighbours of each node
    N = W.shape[0]
    neighbours = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] != 0:
                neighbours[i].append(j)
                neighbours[j].append(i)

    # define generator of rhs
    def neural_mass():
        for k in range(N):
            # define input to node
            afferent_input = sum( W[j,k] * y(2*j, t-delays[j,k]) for j in neighbours[k] )

            # dynamics of node k
            yield decay*y(2*k+0) - w[k]*(a[k]/b[k])*y(2*k+1) - y(2*k+0)*(y(2*k+0)**2/a[k]**2 + y(2*k+1)**2/b[k]**2) + kappa * 1 / (1 + e**(-c*(afferent_input-o)))
            yield decay*y(2*k+1) + w[k]*(b[k]/a[k])*y(2*k+0) - y(2*k+1)*(y(2*k)**2/a[k]**2 + y(2*k+1)**2/b[k]**2)

    # set up initial values at random positions at their intrinsic ellipse
    if random_init:
        theta0 = [random.uniform(0, 2*3.14) for _ in range(N)]
        R0 = [random.uniform(0, 1) for _ in range(N)]
    else:
        R0 = [1 for _ in range(N)]
        theta0 = [0 for _ in range(N)]
    y0 = np.zeros((2*N)) 
    for k in range(0,N):
        y0[2*k] = R0[k] * cos(theta0[k])
        y0[2*k+1] = R0[k] * sin(theta0[k])

    # set-up DDE configurations
    DDE = jitcdde(neural_mass, n=2*N)  # normal form
    DDE.compile_C(do_cse=True, chunk_size=int(N*2))
    DDE.set_integration_parameters(rtol=rtol,atol=atol)
    DDE.constant_past(y0, time=0.0)
    #DDE.step_on_discontinuities(propagations=0)
    #DDE.integrate_blindly(np.amax(delays), step=step)
    DDE.adjust_diff()

    # solve
    data = []
    t = []
    for time in np.arange(DDE.t, DDE.t+t_span[1],  step):
        data.append( DDE.integrate(time) )
        t.append(time)

    data = np.array(data)
    data = np.transpose(data)
    t = np.array(t)

    # organize solution as dictionary (normal form style)
    sol = {}
    sol['t'] = t
    sol['x'] = data[0:2*N:2,:]
    sol['y'] = data[1:2*N:2,:]

    return sol

# -------------------------------------------------------
# this function applies the above in a series provided
# by the rhythms output of network_spreading
# -------------------------------------------------------
def network_rhythms_series(rhythms, t_span=(0,50), delays=False, decay=-0.01, w=False, kappa=10, step=10**-3, random_init=True):
       # initialize solutions
       solutions = [None for _ in range(len(rhythms))]

       # iterate through time-stamps in rhythms
       for i in range(len(rhythms)):
            # unpack
            W, a, b, t = rhythms[i]

            # solve
            nnm = network_rhythms(W, a=a, b=b, t_span=t_span, delays=delays, decay=decay, step=step, w=w, kappa=kappa, random_init=random_init)

            # extract and store solution
            solutions[i] = (nnm['t'], nnm['x'], nnm['y'])

       # we are done
       return solutions
      

# ---------------------------------------------------------------------
# this function returns the average (over L times) for the
# network_rhythms_series. The stochasticity lies in the
# generation of intrinsic frequencies (gaussian)
#
# Return
# -----
# solutions : list of 3-tuples
#       Tuples include (time, x (for each run), y (for each run))
# ------------------------------------------------------------------
def mean_network_rhythms_series(L, rhythms, t_span=(0,50), delays=False, decay=-0.01, mean_w=[10], dw=[0], kappa=10, step=10**-3, random_init=True, atol=10**-6, rtol=10**-4):
       # initialize solutions
       import scipy as sp
       solutions = np.array([None for _ in range(len(rhythms))])

       # get number of nodes
       W0, a0, b0, t0 = rhythms[0]
       N = W0.shape[0]

       # initialize output
       x_collected = [[] for _ in range(len(rhythms))]
       y_collected = [[] for _ in range(len(rhythms))]

       # iterate through time-stamps in rhythms
       for l in range(L):
           # construct frequency parameters
           w = []
           if len(mean_w) > 1:
               for n in range(N):
                   w.append(np.random.normal(loc=mean_w[n], scale=dw[n], size=1)[0])
           else:
               w = np.random.normal(loc=mean_w[0], scale=dw[0], size=N)


           for i in range(len(rhythms)):
                # unpack
                W, a, b, t = rhythms[i]

                # solve
                nnm = network_rhythms_symbolically(W, a=a, b=b, t_span=t_span, delays=delays, decay=decay, step=step, w=w, kappa=kappa, random_init=random_init, atol=atol, rtol=rtol)

                # extract and store solution
                x_collected[i].append(nnm['x'])
                y_collected[i].append(nnm['y'])

                # print update to screen
                print(f'\tCompleted simulation {i+1} of {len(rhythms)} for trial {l+1} of {L}')
       t = nnm['t']

       # package output similarly to above
       for i in range(len(rhythms)):
           solutions[i] = (t, np.array(x_collected[i]), np.array(y_collected[i]))

       # we're done
       return solutions

# ---------------------------------------------------------------------
# this function returns the average (over L times) for the
# canonical Wilson-Cowan model. The stochasticity lies in the
# initial conditions only.
#
# Return
# -----
# solutions : list of 3-tuples
#       Tuples include (time, x (for each run), y (for each run))
# ------------------------------------------------------------------
def wilson_cowan_trajectory(L, rhythms, t_span=(0,50), delays=False, taux=0.013, tauy=0.013, Cxx=24, \
        Cxy=-20, Cyy=0, Cyx=40, P=1, Q=-2, h=1, Sa=1, theta=4, kappa=0.4, osc_freqs=False, \
        step=10**-3, random_init=True, atol=10**-6, rtol=10**-4):
       # initialize solutions
       import scipy as sp
       solutions = np.array([None for _ in range(len(rhythms))])

       # get number of nodes
       W0, P0, Q0, t0 = rhythms[0]
       N = W0.shape[0]

       # initialize output
       x_collected = [[] for _ in range(len(rhythms))]
       y_collected = [[] for _ in range(len(rhythms))]

       # iterate through time-stamps in rhythms
       for l in range(L):
           for i in range(len(rhythms)):
                # unpack
                W, P, Q, t = rhythms[i]

                # if osc_freqs a number then make a list with that number
                if osc_freqs and not isinstance(osc_freqs, list):
                    #osc_freqs = [osc_freqs for _ in range(N)]
                    osc_freqs = np.full((N), osc_freqs)

                # solve
                nnm = wilson_cowan(W, P=P, Q=Q, t_span=t_span, delays=delays, taux=taux, tauy=tauy, \
                        Cxx=Cxx, Cxy=Cxy, Cyy=Cyy, Cyx=Cyx, h=h, Sa=Sa, theta=theta, \
                        kappa=kappa, osc_freqs=osc_freqs, random_init=random_init, atol=atol, \
                        rtol=rtol, step=step)

                # extract and store solution
                x_collected[i].append(nnm['x'])
                y_collected[i].append(nnm['y'])

                # print update to screen
                print(f'\tCompleted simulation {i+1} of {len(rhythms)} for trial {l+1} of {L}')
       t = nnm['t']

       # package output similarly to above
       for i in range(len(rhythms)):
           solutions[i] = (t, np.array(x_collected[i]), np.array(y_collected[i]))

       # we're done
       return solutions

# ---------------------------------------------------------------------
# this function returns the average (over L times) for the
# Hopf normal form. The stochasticity lies in the
# generation of intrinsic frequencies (gaussian) and initial conditions.
#
# Return
# -----
# solutions : list of 3-tuples
#       Tuples include (time, x (for each run), y (for each run))
# ------------------------------------------------------------------
def hopf_trajectory(L, rhythms, t_span=(0,50), delays=False, decay=-0.01, mean_w=[10], dw=[0], \
        kappa=10, c=1, o=0, step=10**-3, random_init=True, atol=10**-6, rtol=10**-4):
       # initialize solutions
       import scipy as sp
       solutions = np.array([None for _ in range(len(rhythms))])

       # get number of nodes
       W0, a0, b0, t0 = rhythms[0]
       N = W0.shape[0]

       # initialize output
       x_collected = [[] for _ in range(len(rhythms))]
       y_collected = [[] for _ in range(len(rhythms))]

       # iterate through time-stamps in rhythms
       for l in range(L):
           # construct frequency parameters
           w = []
           if len(mean_w) > 1:
               for n in range(N):
                   w.append(np.random.normal(loc=mean_w[n], scale=dw[n], size=1)[0])
           else:
               w = np.random.normal(loc=mean_w[0], scale=dw[0], size=N)


           for i in range(len(rhythms)):
                # unpack
                W, a, b, t = rhythms[i]

                # solve
                nnm = hopf(W, a=a, b=b, t_span=t_span, delays=delays, decay=decay, step=step, w=w, \
                        kappa=kappa, c=c, o=o, random_init=random_init, atol=atol, rtol=rtol)

                # extract and store solution
                x_collected[i].append(nnm['x'])
                y_collected[i].append(nnm['y'])

                # print update to screen
                print(f'\tCompleted simulation {i+1} of {len(rhythms)} for trial {l+1} of {L}')
       t = nnm['t']

       # package output similarly to above
       for i in range(len(rhythms)):
           solutions[i] = (t, np.array(x_collected[i]), np.array(y_collected[i]))

       # we're done
       return solutions


# -------------------------------
# solve slow fast wilson-cowan
# -------------------------------
def SF_wilson_cowan(W, taux=1, tauy=1, tauz=1, Cxx=1, Cxy=1, Cxz=1, Cyx=1, Cyy=1, Cyz=1, Czx=1, Czy=1, Czz=1, kappa=1, h=1, Sa=1, theta=4, R=-10, P=1, Q=-2, mu=0.05, sigma=0.05, tau=0.05, step=10**-4, random_init=True, atol=10**-6, rtol=10**-4, t_span=(0,50), a=[], b=[], c=[]):
    # import must be within function (or else t will not be caught)
    from jitcsde import jitcsde, y
    import symengine as sym

    # find neighbours of each node
    N = W.shape[0]
    neighbours = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] != 0:
                neighbours[i].append(j)
                neighbours[j].append(i)

    # define generator of rhs
    def neural_mass():
        for k in range(N):
            # update parameters 
            P = a[k]
            Q = b[k]
            R = c[k]

            # define input to node
            afferent_input = kappa * sum( W[j,k] * y(6*j+0) for j in neighbours[k] )
            Sx = h*(1+e**(-Sa*((Cxx*y(6*k+0) + Cxy*y(6*k+1) + Cxz*y(6*k+2) + P + afferent_input) - theta)))**-1
            Sy = h*(1+e**(-Sa*((Cyx*y(6*k+0) + Cyy*y(6*k+1) + Cyz*y(6*k+2) + Q) - theta)))**-1
            Sz = h*(1+e**(-Sa*((Czx*y(6*k+0) + Czy*y(6*k+1) + Czz*y(6*k+2) + R) - theta)))**-1

            
            # deterministic right hand side
            yield 1/taux * (-y(6*k+0) + Sx) + y(6*k+3)          
            yield 1/tauy * (-y(6*k+1) + Sy) + y(6*k+4)
            yield 1/tauz * (-y(6*k+2) + Sz) + y(6*k+5)
            yield -(y(6*k+3)-mu)/tau
            yield -(y(6*k+4)-mu)/tau
            yield -(y(6*k+5)-mu)/tau
    
    def noise():
        for k in range(N):
            # stochastic right hand side
            yield 0
            yield 0
            yield 0
            yield sigma * (2/tau)**(1/2)
            yield sigma * (2/tau)**(1/2)
            yield sigma * (2/tau)**(1/2)


    # set up initial values at random positions at their intrinsic ellipse
    if random_init:
        R0 = [np.random.uniform(0, 0.2) for _ in range(N)]
    else:
        R0 = [0.1 for _ in range(N)]
    y0 = np.zeros((6*N))
    for k in range(0,N):
        # subpopulations IC
        y0[6*k+0] = R0[k] 
        y0[6*k+1] = R0[k]
        y0[6*k+2] = R0[k]

        # noise IC
        y0[6*k+3] = 0 
        y0[6*k+4] = 0
        y0[6*k+5] = 0

    # set-up SDE configurations
    DDE = jitcsde(neural_mass, noise, n=6*N)
    DDE.set_integration_parameters(rtol=rtol,atol=atol)
    DDE.set_initial_value(y0, 0.0)

    # solve
    data = []
    t = []
    for time in np.arange(DDE.t, DDE.t+t_span[1],  step):
        data.append( DDE.integrate(time) )
        t.append(time)

    data = np.array(data)
    data = np.transpose(data)
    t = np.array(t)

    # organize solution as dictionary
    sol = {}
    sol['t'] = t
    sol['x'] = data[0:6*N:6,:]
    sol['y'] = data[1:6*N:6,:]
    sol['z'] = data[2:6*N:6,:]
    sol['x_noise'] = data[3:6*N:6,:]
    sol['y_noise'] = data[4:6*N:6,:]
    sol['z_noise'] = data[5:6*N:6,:]

    return sol


# ---------------------------------------------------------------------
# this function returns the average (over L times) for the
# slow/fast Wilson-Cowan. The stochasticity lies in the
# initial conditions and noise terms.
#
# Return
# -----
# solutions : list of 3-tuples
#       Tuples include (time, x (for each run), y (for each run), z (for each run))
# ------------------------------------------------------------------
def SF_wilson_cowan_trajectory(L, rhythms, taux=0.03, tauy=0.03, tauz=0.267, Cxx=24, Cxy=-20, Cxz=-20, Cyx=40, Cyy=0, Cyz=0, Czx=40, Czy=0, Czz=0, kappa=4.4, h=1, Sa=1, theta=4, P=1, Q=-2, R=-10, mu=0, sigma=0.5, tau=0.005, t_span=(0,50), step=0.001, random_init=True, atol=10**-6, rtol=10**-3):
       # initialize solutions
       import scipy as sp
       solutions = np.array([None for _ in range(len(rhythms))])

       # check if c is included or not
       try:
           W0, a0, b0, c0, t0 = rhythms[0]
       except:
           W0, a0, b0, t0 = rhythms[0]
           N = W0.shape[0]
           c0 = np.array( [[R for _ in range(N)] for _ in range(t0.size)] )

       # get number of nodes
       N = W0.shape[0]

       # initialize output
       x_collected = [[] for _ in range(len(rhythms))]
       y_collected = [[] for _ in range(len(rhythms))]
       z_collected = [[] for _ in range(len(rhythms))]

       # iterate through time-stamps in rhythms
       for l in range(L):
           for i in range(len(rhythms)):
                # unpack, try first with c, if not create constant c vector
                try:
                    W, a, b, c, t = rhythms[i]
                except:
                    W, a, b, t = rhythms[i]
                    c = np.array( [R for _ in range(N)] )


                # solve
                nnm = SF_wilson_cowan(W, taux=taux, tauy=tauy, tauz=tauz, Cxx=Cxx, Cxy=Cxy, Cxz=Cxz, Cyx=Cyx, Cyy=Cyy, Cyz=Cyz, Czx=Czx, Czy=Czy, Czz=Czz, kappa=kappa, h=h, Sa=Sa, theta=theta, P=P, Q=Q, R=R, mu=mu, sigma=sigma, tau=tau, t_span=t_span, step=step, random_init=random_init, atol=atol, rtol=rtol, a=a, b=b, c=c)

                # extract and store solution
                x_collected[i].append(nnm['x'])
                y_collected[i].append(nnm['y'])
                z_collected[i].append(nnm['z'])

                # print update to screen
                print(f'\tCompleted simulation {i+1} of {len(rhythms)} for trial {l+1} of {L}')
       t = nnm['t']

       # package output similarly to above
       for i in range(len(rhythms)):
           solutions[i] = (t, np.array(x_collected[i]), np.array(y_collected[i]))

       # we're done
       return solutions


# --------------------------------------------------
# produce rhythms with desired time points
# from spreading solution
# -------------------------------------------------

def create_rhythms(spread_sol, start, end, n_of_points):
    # extract stuff
    a = spread_sol['a']
    b = spread_sol['b']
    # try to see if c is included (wilson-cowan)
    if 'c' in spread_sol and spread_sol['c'].size > 0:
        c = spread_sol['c']
        c_exist = True
    else:
        c_exist = False
    w = spread_sol['w']
    t_spread = spread_sol['t']
    edges = spread_sol['w_map']
    N = a.shape[0]
    M = w.shape[0]

    # find times in spread closest to desired time points
    t_desired = np.linspace(start, end, n_of_points)
    stamps_ind = []
    for t_d in t_desired:
        index = np.argmin(np.abs(t_spread - t_d))
        stamps_ind.append(index)

    # create rhythms
    rhythms = []
    for ind in stamps_ind:
        W_t = np.zeros((N,N))
        if M > 1:  # check that adj matrix not empty
            for i in range(M):
                n, m = edges[i]
                weight = w[i,ind]
                W_t[n,m] = weight
                W_t[m,n] = weight
        if c_exist:
            rhythms.append((W_t, a[:,ind], b[:,ind], c[:,ind], t_spread[ind]))
        else:
            rhythms.append((W_t, a[:,ind], b[:,ind], t_spread[ind]))

    # we're done
    return rhythms

# ---------------------------------------------------
# Compute mean phase coherence of signal
#
#   input:
#           signal - array-like (signals, time)
# ---------------------------------------------------
def MPC(signal):
    # find phases of signal (over time) using the Hilbert transform
    hil = hilbert(signal)
    phases = np.angle(hil)

    # initialize functional matrix (lower triangular)
    N, T = signal.shape  
    F = np.zeros((N,N))

    # compute MPCs for each node pair
    for c in range(N):
        for r in range(c+1,N):
            diff_phase = phases[r,:] - phases[c,:]

            R_rc = 0
            for i in range(T):
                R_rc += e**(complex(0,diff_phase[i]))
            R_rc = R_rc/T
            R_rc = abs(R_rc)

            F[r,c] = R_rc
            F[c,r] = R_rc
    
    # we're done
    return F

# ---------------------------------------------------
# Compute phase-lag index (PLI) of signal
#
#   input:
#           signal - array-like (signals, time)
# ---------------------------------------------------
def PLI(signal):
    # find phases of signal (over time) using the Hilbert transform
    hil = hilbert(signal)
    phases = np.angle(hil)

    # initialize functional matrix (lower triangular)
    N, T = signal.shape  
    F = np.zeros((N,N))

    # compute MPCs for each node pair
    for c in range(N):
        for r in range(c+1,N):
            diff_phase = phases[r,:] - phases[c,:]

            pli_i = np.sum(np.sign(np.sign(diff_phase))) / T
            pli_i = abs(pli_i)

            F[r,c] = pli_i
            F[c,r] = pli_i
    
    # we're done
    return F

# ---------------------------------------------------
# Compute mean phase coherence of signal
#
#   input:
#           signal - array-like (signals, time)
# ---------------------------------------------------
def im_coherence(signal, fs, bands, window_sec=10):
    from scipy.signal import welch, csd, periodogram
    # initialize functional matrix (lower triangular)
    N, T = signal.shape  
    Fs = [ np.zeros((N,N)) for _ in range(len(bands))]

    # compute MPCs for each node pair
    for c in range(N):
        for r in range(c+1,N):
            # compute cross-spectrums
            freq_c, Pcc = welch(signal[c,:], fs=fs, nperseg=fs*window_sec)
            freq_r, Prr = welch(signal[r,:], fs=fs, nperseg=fs*window_sec)
            freq_cr, Pcr = csd(signal[c,:],signal[r,:],fs=fs, nperseg=fs*window_sec)
            freq_res = freq_cr[1] - freq_cr[0]

            # compute coherence as a function of frequency
            Icoh_cr = np.imag(Pcr)/np.sqrt(Pcc*Prr)

            # integrate over frequency band
            for b, band in enumerate(bands):
                low, high = band
                idx_band = np.logical_and(freq_cr >= low, freq_cr <= high)
                Icoh_cr_b = 1/(high-low) * simps(np.abs(Icoh_cr[idx_band]), dx=freq_res)
                #Icoh_cr_b = np.arctanh(Icoh_cr_b)

                # debugging
                #print(f'band= {low} - {high}')
                #print(f'len idx = {len(idx_band)}')
                #print(f'idx_band = \n{idx_band}')
                #print(f'freq_res = {freq_res}')
                #print(f'freq_cr[idx_band] = {freq_cr[idx_band]}')
                #print(f' integral = {Icoh_cr_b}')
                #plt.figure()
                #plt.plot(freq_cr, Icoh_cr, alpha=0.5)
                #plt.plot(freq_cr[idx_band], Icoh_cr[idx_band], color='green')
                #plt.xlim([0,12])
                #plt.show()

                Fs[b][r,c] = Icoh_cr_b
                Fs[b][c,r] = Icoh_cr_b
        
    # we're done
    return Fs
           
# --------------------------------------------
# return average envelopes of oscillations
# --------------------------------------------

def envelopes(rhythms, solutions, n_of_points):
    from scipy.signal import find_peaks

    # find number of nodes and trials
    t0 , x0, _ = solutions[0]
    L = x0.shape[0]
    N = x0.shape[1]
    start = t0[0]
    end = t0[-1]

    # initialize
    t_avg = np.linspace(start, end, n_of_points)
    maxima_N = [[[[] for _ in range(len(t_avg)-1)] for _ in range(N)] for _ in range(len(rhythms))]
    minima_N = [[[[] for _ in range(len(t_avg)-1)] for _ in range(N)] for _ in range(len(rhythms))]


    for i in range(len(rhythms)):
            # extract solutions
            t, x, y = solutions[i]
            _, a, _, _ = rhythms[i]

            # discretize t into t_avg intervals
            tops = np.array([None for _ in range(len(t_avg)-1)])
            bottoms = np.array([None for _ in range(len(t_avg)-1)])
            for s in range(len(t_avg)-1):
                tops[s] = (np.abs(np.array(t) - t_avg[s+1])).argmin()
                bottoms[s]= (np.abs(np.array(t) - t_avg[s])).argmin()

            for j in range(N):
                for l in range(L):
                    xl = x[l]
                    # find mean of peaks in intervals of t_avg
                    for s in range(len(t_avg)-1):
                        top = tops[s]
                        bottom = bottoms[s]

                        # maxima
                        local_xl = xl[j, bottom:top]
                        local_max, _ = find_peaks(local_xl)
                        if len(local_max) == 0:
                            mean_max = max(local_xl)
                        else:
                            mean_max = np.mean(local_xl[local_max])

                        # minima
                        local_xl = xl[j, bottom:top]
                        local_min, _ = find_peaks(-local_xl)
                        if len(local_min) == 0:
                            mean_min = min(local_xl)
                        else:
                            mean_min = np.mean(local_xl[local_min])

                        # append amplitudes
                        maxima_N[i][j][s].append(mean_max)
                        minima_N[i][j][s].append(mean_min)
    return t_avg, maxima_N, minima_N

# -------------------------------------------------
# returns fig, axs object for plotting envelopes
# ------------------------------------------------
def plot_envelopes(t_stamps, t_avg, maxima_N, minima_N, title, node_labels, colours, par_lines=False, N=0):
    # find N
    if N == 0:
        N = len(maxima_N[0])
    else:
        pass
    len_rhythms = len(maxima_N)

    # initialize
    fig2, axs2 = plt.subplots(len_rhythms, N, sharex=True, sharey=True)

    # plot
    fig2.suptitle(title)
    for i in range(len_rhythms):
            # y-labels
            axs2[i,0].set_ylabel(f'$t_{{spread}} = {int(t_stamps[i])}$')
            
            # plot each node
            for j in range(N):
                # plot envelopes
                axs2[i,j].errorbar(t_avg[0:-1], np.mean(maxima_N[i][j], axis=1), c=colours[j], yerr=np.std(maxima_N[i][j], axis=1), capsize=4, capthick=1, errorevery=ceil(len(t_avg)/4), alpha=0.75, linewidth=1.5)
                axs2[i,j].errorbar(t_avg[0:-1], np.mean(minima_N[i][j], axis=1), c=colours[j], yerr=np.std(maxima_N[i][j], axis=1), capsize=4, capthick=1, errorevery=ceil(len(t_avg)/4), alpha=0.75, linewidth=1.5)

                if par_lines:
                    axs2[i,j].hlines(a[j]*decay**0.5, t_span[0], t_span[1], linestyles='dashed', colors=colours[j], alpha=0.5)
                    axs2[i,j].hlines(-a[j]*decay**0.5, t_span[0], t_span[1], linestyles='dashed', colors=colours[j], alpha=0.5)

            # limits
            #axs2[i,j].set_ylim([-2.1, 2.1])

    # titles and x-labels
    for i in range(N):
            axs2[-1,i].set_xlabel(f'$t_{{rhythms}}$')
            axs2[0,i].set_title(node_labels[i])

    # we're done
    return fig2, axs2

# -----------------------------------------------
# plot envelopes for connectome, one figure
# per t_spread
# -----------------------------------------------
def plot_envelopes_connectome(t_stamps, t_avg, maxima_N, minima_N, colours, rows=9, cols=10, regions=[], t_cutoff=None, wiggle=0):
    # find N, etc.
    N = len(maxima_N[0])
    S = len(maxima_N[0][0])
    len_rhythms = len(t_stamps)
    L = len(maxima_N[0][0][0])
    global_avg = 0

    # define x-axis for average plot
    wiggle = (t_stamps[-1] - t_stamps[0])*wiggle/(2*len(regions))
    wiggled = [np.array(t_stamps) + (i - len(regions)/2)*wiggle for i in range(len(regions)+1)]

    # convert to numpy array
    maxima_N = np.array(maxima_N)

    # figure for amplitudes over regions over time
    if regions:
        fig2 = plt.figure()
        axs2 = fig2.gca()
        plt.xlabel(f'$t_{{spread}}$')
        plt.ylabel(f'Average excitatory envelope')
        # initialization
        avg_region = [[[0 for _ in range(L)] for _ in range(len(t_stamps))] for _ in range(len(regions))]
        grand_avg = [[0 for _ in range(L)] for _ in range(len(t_stamps))]

        cutoff_ind = np.argmin(np.abs(t_avg - t_cutoff))

    # plot
    figs = []
    axss = []
    for i in range(len(t_stamps)):
            # get figure for t_spread
            fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False)

            # title and labels
            fig.suptitle(f'$t_{{spread}} = {int(t_stamps[i])}$')
            plt.tick_params(labelcolor="none", bottom=False, left=False)
            fig.text(0.5, 0.04, f'$t_{{rhythms}}$', ha='center')
            fig.text(0.04, 0.5, 'Excitatory envelopes', va='center', rotation='vertical')

            # plot each node
            for j in range(N):
                # add node's average amplitude to its region
                for r, region in enumerate(regions):
                    if j in region:
                        for l in range(L):
                            avg = 0
                            for s in range(cutoff_ind, S):
                                avg += maxima_N[i][j][s][l] 
                            avg /= S-cutoff_ind
                            avg_region[r][i][l] += avg/len(region)
                            grand_avg[i][l] += avg/N 
                        break

                # plot envelopes
                axs[j//10,j%10].errorbar(t_avg[0:-1], np.mean(maxima_N[i][j], axis=1), c=colours[r], yerr=np.std(maxima_N[i][j], axis=1), capsize=4, capthick=1, errorevery=ceil(len(t_avg)/4), alpha=0.75, linewidth=1.25)
                axs[j//10,j%10].errorbar(t_avg[0:-1], np.mean(minima_N[i][j], axis=1), c=colours[r], yerr=np.std(maxima_N[i][j], axis=1), capsize=4, capthick=1, errorevery=ceil(len(t_avg)/4), alpha=0.75, linewidth=1.25)

            # append node-envelope figure to return list
            figs.append(fig)
            axss.append(axs)

    # plot average amplitude over region
    #avg_global = 0
    #avgs = []
    for r in range(len(regions)):
        axs2.errorbar(wiggled[r], np.mean(avg_region[r][:], axis=1), c=colours[r], yerr=np.std(avg_region[r][:], axis=1), capsize=6, capthick=2, alpha=0.75, linestyle='--', marker='o')
        #axs2.plot(t_stamps, np.mean(avg_region[r][:], axis=1), c=colours[r])
        #avgs.append(np.mean(avg_region[r][:], axis=1))
        #avg_global += np.mean(avg_region[r][:], axis=1)
    #avg_global /= len(avgs)

    # plot average over all nodes
    axs2.errorbar(wiggled[-1], np.mean(grand_avg, axis=1), c='black', yerr=np.std(grand_avg, axis=1), capsize=6, capthick=2, alpha=0.75, linestyle='--', marker='o')

    #for r in range(len(regions)):
    #    avg_r = avg_region[r][:]
    #    axs2.errorbar(t_stamps, np.mean(avg_r, axis=1), c=colours[r], yerr=np.std(avg_r, axis=1), capsize=4, capthick=1, alpha=0.75)
    #    axs2.set_ylim([0,1.5])

    figs.append(fig2)

    # we're done
    return figs, axss

# -----------------------------------------------------
# Plot average oscillatory acitivity per Bick & Goriely
# ------------------------------------------------------

def plot_oscillatory_activity(t_stamps, solutions, regions=[], colours=[], legends=[], normalize=False, wiggle=0):
    # find N, etc.
    t, x0, y0 = solutions[0]
    N = x0[0].shape[0]
    L = len(x0)

    # initialize figure
    fig = plt.figure()
    axs = fig.gca()
    plt.xlabel(f'$t_{{spread}}$')
    plt.ylabel(f'Average oscillatory activity')

    # initialization for plotting input
    avg_region = [[[0 for _ in range(L)] for _ in range(len(t_stamps))] for _ in range(len(regions))]
    glob_l1 = []  # global average over first trial (used in normalization)

    # iterate through spreading time
    for i in range(len(t_stamps)):
            # get solution
            t, x, y = solutions[i]
            
            # iterate through trials
            for l in range(L):
                zl_mod = np.abs(x[l] + 1j * y[l])
                # add node's average amplitude to its region
                for n in range(N):
                    for r, region in enumerate(regions):
                        if n in region:
                            node_integral = trapz(zl_mod[n,:], t)
                            node_activity = 1/(t[-1]-t[0]) * node_integral
                            if l == 1:
                                glob_l1.append(node_activity)
                            avg_region[r][i][l] += node_activity / len(region)
                            break

    # wiggle
    wiggle = (t_stamps[-1] - t_stamps[0])*wiggle/(2*len(regions))
    wiggled = [np.array(t_stamps) + (i - len(regions)/2)*wiggle for i in range(len(regions)+1)]

    # find average for first trial to use in normalization
    regions_l1 = [0 for _ in range(len(regions))]
    for r in range(len(regions)):
        regions_l1[r] = np.mean(avg_region[r][:], axis=0)[0]

    # plot activity for each brain region, averaging over trials
    for r in range(len(regions)):
        if normalize:
            axs.errorbar(wiggled[r], np.mean(avg_region[r][:], axis=1)/regions_l1[r], c=colours[r], yerr=np.std(avg_region[r][:], axis=1)/regions_l1[r], capsize=4, capthick=1, alpha=0.75, label=legends[r], linestyle='--', marker='o')
        else:
            axs.errorbar(wiggled[r], np.mean(avg_region[r][:], axis=1), c=colours[r], yerr=np.std(avg_region[r][:], axis=1), capsize=4, capthick=1, alpha=0.75, label=legends[r], linestyle='--', marker='o')

        axs.legend()

    # we're done
    return fig, axs

# -----------------------------------------------
# compute power-spectal properties
# -----------------------------------------------

def spectral_properties(solutions, bands, fourier_cutoff, modified=False, functional=False, db=False, freq_tol=0, relative=False, window_sec=None):
    #from mne.connectivity import spectral_connectivity
    from fourier import power_spectrum, bandpower, frequency_peaks

    # find size of rhythms
    _, x0, _ = solutions[0]
    L = x0.shape[0]
    N = x0.shape[1]
    len_rhythms = len(solutions) 

    # find average power (and frequency peaks) in bands for each node over time stamps
    bandpowers = [[[[] for _ in range(len_rhythms)] for _ in range(N)] for _ in range(len(bands))]
    freq_peaks = [[[[] for _ in range(len_rhythms)] for _ in range(N)] for _ in range(len(bands))]

    # functional connectivity parameters and initializations
    if functional:
        functional_methods = ['coh', 'pli', 'plv']
        average_strengths = [[[[] for _ in range(len_rhythms)] for _ in range(len(functional_methods))] for _ in range(len(bands))]

    for b in range(len(bands)):
        for i in range((len_rhythms)):
            t, x, y = solutions[i]
            #print(t[-1])

            # iterate through trials
            for l in range(L):
                xl = x[l]

                # find last 10 seconds
                inds = [s for s in range(len(t)) if t[s]>fourier_cutoff]
                t = t[inds]
                x_cut = xl[:,inds]
                tot_t = t[-1] - t[0]
                sf = len(x_cut[0])/tot_t

                # compute spectral connectivity
                if functional:
                    functional_connectivity = spectral_connectivity([x_cut], method=functional_methods, sfreq=sf, fmin=bands[b][0], fmax=bands[b][1], mode='fourier', faverage=True, verbose=False)
                    
                    # get average link strength
                    for j in range(len(functional_methods)):
                        functional_matrix = functional_connectivity[0][j]  # lower triangular
                        n_rows, n_cols, _ = functional_matrix.shape

                        # compute average strength
                        average_strength = 0
                        for c in range(n_cols):
                                for r in range(c+1, n_cols): 
                                    average_strength += functional_matrix[r,c][0]
                        average_strength /= N*(N-1)/2

                        # append
                        average_strengths[b][j][i].append(average_strength)

                # find PSD and peak
                for j in range(N):
                    # PSD
                    bandpower_t = bandpower(x_cut[j], sf, bands[b], modified=modified, relative=relative, window_sec=window_sec)
                    if db:
                        bandpower_t = 10*log10(bandpower_t)
                    bandpowers[b][j][i].append(bandpower_t)

                    # frequency peaks
                    freq_peak_t = frequency_peaks(x_cut[j,:], sf, band=bands[b], tol=freq_tol, modified=modified, window_sec=window_sec)
                    freq_peaks[b][j][i].append(freq_peak_t)

    # package return value
    spectral_props = [bandpowers, freq_peaks]
    if functional:
        spectral_props.append(average_strengths)

    return spectral_props

# ---------------------------------------------
# plot spectral properties
# -------------------------------------------
def plot_spectral_properties(t_stamps, bandpowers, freq_peaks, bands, wiggle, title, legends, colours, bandpower_ylim = False, only_average=False, regions=[], n_ticks=5, relative=False):
    # find N and length of rhythms
    N = len(bandpowers[0])
    L = len(bandpowers[0][0][0])
    len_rhythms = len(bandpowers[0][0])

    # initialize
    figs_PSD = []
    figs_peaks = []
    axs_PSD = []
    axs_peaks = []
    for b in bands:
        fig_PSD = plt.figure()  # PSD
        plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = n_ticks) )
        figs_PSD.append(fig_PSD)

        fig_peaks = plt.figure()  # peaks
        plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = n_ticks) )
        #plt.xticks(np.arange(0, 30+1, 10))
        figs_peaks.append(fig_peaks)

    #  wiggle points in x-direction
    if regions:
        wiggle = (t_stamps[-1] - t_stamps[0])*wiggle/(2*len(regions))
        wiggled = [np.array(t_stamps) + (i - len(regions)/2)*wiggle for i in range(len(regions)+1)]
    else:
        wiggle = (t_stamps[-1] - t_stamps[0])*wiggle/(2*N)
        wiggled = [np.array(t_stamps) + (i - N/2)*wiggle for i in range(N+1)]

    # plot average power versus timestamps (one pipeline for regional input, one without)
    for b in range(len(bands)):
        if regions:
            # compute average over regions, and variance of region average over trials
            avg_powers = [[0 for _ in range(L)] for _ in range(len(t_stamps))]
            avg_peaks = [[0 for _ in range(L)] for _ in range(len(t_stamps))]

            for r in range(len(regions)):
                region = regions[r]
                powers = [[0 for _ in range(L)] for _ in range(len(t_stamps))]
                peaks = [[0 for _ in range(L)] for _ in range(len(t_stamps))]
                for ts in range(len(t_stamps)):
                    for l in range(L):
                        zero_peaks = 0
                        for node in region:
                            powers[ts][l] += bandpowers[b][node][ts][l]
                            node_peak = freq_peaks[b][node][ts][l]
                            if not np.isnan(node_peak):
                                peaks[ts][l] += freq_peaks[b][node][ts][l]
                            else: 
                                zero_peaks += 1

                        powers[ts][l] = powers[ts][l]/len(region)
                        if zero_peaks < len(region)/4:
                            peaks[ts][l] = peaks[ts][l]/(len(region)-zero_peaks)
                        else:
                            peaks[ts][l] = float("NaN")

                        # compute average over entire brain
                        if r == 0:
                            # average of trial
                            zero_peaks = 0
                            for n in range(N):
                                avg_powers[ts][l] += bandpowers[b][n][ts][l] 
                                
                                node_peak = freq_peaks[b][n][ts][l]
                                if not np.isnan(node_peak):
                                    avg_peaks[ts][l] += freq_peaks[b][n][ts][l] 
                                else:
                                    zero_peaks += 1
                            avg_powers[ts][l] /= N
                            if zero_peaks < N/4:
                                avg_peaks[ts][l] /= N - zero_peaks
                            else:
                                avg_peaks[ts][l] = float("NaN")


                # plot regions
                if not only_average:
                    # plot power
                    mean = np.mean(powers,axis=1)
                    std = np.std(powers,axis=1)
                    #figs_PSD[b].axes[0].plot(wiggled[r], mean, '-o', c=colours[r], \
                    #        alpha=0.75, label=legends[r])
                    #figs_PSD[b].axes[0].fill_between(wiggled[r], mean-std, mean+std, \
                    #        color=colours[r], alpha=0.2)
                    sns.despine()  # remove right and upper axis line

                    # plot peaks
                    mean = np.mean(peaks,axis=1)
                    std = np.std(peaks,axis=1)
                    #figs_peaks[b].axes[0].plot(wiggled[r], mean, '-o', c=colours[r], \
                    #        alpha=0.75, label=legends[r]) 
                    #figs_peaks[b].axes[0].fill_between(wiggled[r], mean-std, mean+std, \
                    #        color=colours[r], alpha=0.2)
                    sns.despine()  # remove right and upper axis line

                    figs_PSD[b].axes[0].spines['right'].set_visible(False)
                    figs_PSD[b].axes[0].spines['top'].set_visible(False)
                    figs_PSD[b].axes[0].errorbar(wiggled[r], np.mean(powers, axis=1), c=colours[r], \
                            label=legends[r], marker='o', linestyle='--', alpha=0.75, yerr=np.std(powers, axis=1), capsize=6, capthick=2)
                    sns.despine()
                    figs_peaks[b].axes[0].errorbar(wiggled[r], np.nanmean(peaks, axis=1), c=colours[r], \
                            label=legends[r], marker='o', linestyle='--', alpha=0.75, yerr=np.std(peaks,axis=1), capsize=6, capthick=2)
                    sns.despine()


            # plot average        
            # global power
            mean = np.mean(avg_powers,axis=1)
            std = np.std(avg_powers,axis=1)
            #figs_PSD[b].axes[0].plot(wiggled[-1], mean, '-o', c='black', \
            #        alpha=0.75, label='global mean')
            #figs_PSD[b].axes[0].fill_between(wiggled[r], mean-std, mean+std, color='black', alpha=0.2)
            #sns.despine()  # remove right and upper axis line

            # global peaks
            mean = np.mean(avg_peaks,axis=1)
            std = np.std(avg_peaks,axis=1)
            #figs_peaks[b].axes[0].plot(wiggled[-1], mean, '-o', c='black', \
            #        alpha=0.75, label='global mean')
            #figs_peaks[b].axes[0].fill_between(wiggled[r], mean-std, mean+std, color='black', alpha=0.2)
            #sns.despine()  # remove right and upper axis line

            figs_PSD[b].axes[0].errorbar(wiggled[-1], np.mean(avg_powers, axis=1), c='black', \
                    label='average', marker='o', linestyle='--', alpha=0.75, yerr=np.std(powers, axis=1), capsize=6, capthick=2)
            sns.despine()
            figs_peaks[b].axes[0].errorbar(wiggled[-1], np.nanmean(avg_peaks, axis=1), c='black', \
                    label='average', marker='o', linestyle='--', alpha=0.75, yerr=np.std(peaks,axis=1), capsize=6, capthick=2)
            sns.despine()

        else: 
            avg_power = np.array([[None for _ in range(N)] for _ in range(len_rhythms)])
            avg_peak = np.array([[None for _ in range(N)] for _ in range(len_rhythms)])
            for i in range(N):
                # power
                power = bandpowers[b][i]
                avg_power[:,i] = np.mean(power, axis=1) 
                if not only_average:
                    figs_PSD[b].axes[0].errorbar(wiggled[i], np.mean(power, axis=1), c=colours[i], label=legends[i], marker='o', linestyle='--', alpha=0.75, yerr=np.std(power, axis=1), capsize=6, capthick=2)

                # peak
                peak = freq_peaks[b][i]
                avg_peak[:,i] = np.mean(peak, axis=1) 
                if not only_average:
                    figs_peaks[b].axes[0].errorbar(wiggled[i], np.nanmean(peak, axis=1), c=colours[i], label=legends[i], marker='o', linestyle='--', alpha=0.75, yerr=np.std(peak,axis=1), capsize=6, capthick=2)

            # average power/peak over nodes
            avg_power = np.array(avg_power, dtype=np.float64)  # not included -> error due to sympy float values 
            figs_PSD[b].axes[0].errorbar(t_stamps, np.mean(avg_power, axis=1), c='black', label='Node average', marker='o', linestyle='--', alpha=0.75, yerr=np.std(avg_power, axis=1), capsize=6, capthick=2)
            
            avg_peak = np.array(avg_peak, dtype=np.float64)  # not included -> error due to sympy float values 
            figs_peaks[b].axes[0].errorbar(t_stamps, np.nanmean(avg_peak, axis=1), c='black', label='Node average', marker='o', linestyle='--', alpha=0.75, yerr=np.std(avg_peak, axis=1), capsize=6, capthick=2)

            
    # set labels 
    for b in range(len(bands)):
        # power
        figs_PSD[b].axes[0].set_title(title)
        if relative:
            figs_PSD[b].axes[0].set_ylabel(f'Relative power (${bands[b][0]} - {bands[b][1]}$ Hz)')
        else:
            figs_PSD[b].axes[0].set_ylabel(f'Absolute power (${bands[b][0]} - {bands[b][1]}$ Hz)')
        #figs_PSD[b].axes[0].set_xlabel('Speading time (years)')
        figs_PSD[b].axes[0].set_xlabel(f'$t_{{spread}}$')
        figs_PSD[b].axes[0].set_xlim([np.amin(wiggled)-wiggle, np.amax(wiggled)+wiggle])

        if bandpower_ylim:
            figs_PSD[b].axes[0].set_ylim([-0.05, bandpower_ylim])

        # peak
        figs_peaks[b].axes[0].set_title(title)
        figs_peaks[b].axes[0].set_ylabel(f'Peak frequency (${bands[b][0]} - {bands[b][1]}$ Hz)')
        #figs_peaks[b].axes[0].set_xlabel('Spreading time (years)')
        figs_peaks[b].axes[0].set_xlabel(f'$t_{{spread}}$')
        figs_peaks[b].axes[0].set_xlim([np.amin(wiggled)-wiggle, np.amax(wiggled)+wiggle])
        figs_peaks[b].axes[0].set_ylim([bands[b][0] - 0.5, bands[b][-1] + 0.5])
        figs_peaks[b].axes[0].set_ylim([8,11])

        # legends
        figs_PSD[b].axes[0].legend()
        plt.tight_layout()
        figs_peaks[b].axes[0].legend()
        plt.tight_layout()

    # find node with the most frequency slowing
    #freq_peaks[b][node][ts][l]
    #node_freqs = []
    #for node in range(N):
    #    freq = abs( np.mean(freq_peaks[0][node][0]) - np.mean(freq_peaks[0][node][-1]) ) 
    #    node_freqs.append(freq)
    #node_freqs = np.array(node_freqs)
    #print(f'node with most extreme slowing: {np.argmax(node_freqs)}')

    # we're done
    return figs_PSD, figs_peaks

# -----------------------------------------------
# computer power spectra over sums
# as done by Bick & Goriely
# --------------------------------------------------
def summed_power(t_stamps, solutions, band, fourier_cutoff, modified=False, regions=[], colours=[], legends=[]):
    from fourier import power_spectrum, bandpower, frequency_peaks

    # find size of rhythms
    _, x0, _ = solutions[0]
    L = x0.shape[0]
    N = x0.shape[1]
    len_rhythms = len(solutions) 
    import pickle


    # sum excitatory (real) solutions over regions
    regions_x = [[[None for _ in range(L)] for _ in range(len_rhythms)] for _ in range(len(regions))]
    for i in range(len_rhythms):
        t, x, y = solutions[i]
        # import from matlab
        #with open("../mat/t.pkl", 'rb') as fin:
        #    t = pickle.load(fin)[0]
        #with open("../mat/x.pkl", 'rb') as fin:
        #    xl = pickle.load(fin)
        for r, region in enumerate(regions):
            for l in range(L):
                xl = x[l]
                region = np.array(region)
                regions_xls = np.mean(xl[region,:], axis=0)
                regions_x[r][i][l] = regions_xls

    # initialize
    PSD_r = np.array([[[None for _ in range(L)] for _ in range(len_rhythms)] for _ in range(len(regions))])
    PSD_global = np.array([[None for _ in range(L)] for _ in range(len_rhythms)])

    # compute power spectral density over band
    for i in range(len_rhythms):
        t, x, y = solutions[i]
        t = t - t[0]
        for l in range(L):
            xl = x[l]

            # cut off transient
            print(f't(0) = {t[0]}, t(final) = {t[-1]}')
            print(f'fourier_cutoff = {fourier_cutoff}')

            # import from matlab
            #with open("../mat/t.pkl", 'rb') as fin:
            #    t = pickle.load(fin)[0]
            #with open("../mat/x.pkl", 'rb') as fin:
            #    xl = pickle.load(fin)
      
            inds = [i for i in range(len(t)) if t[i]>fourier_cutoff]
            t = t[inds]
            x_cut = xl[:,inds]
            tot_t = t[-1] - t[0]
            print(f'tot_t = {tot_t}')
            #sf = len(x_cut[0])/tot_t
            sf = 1000

            # average excitatory solution over all nodes
            xl_global = np.mean(x_cut, axis=0)

            ## interpolate signal as in Bick
            tt = np.linspace(t[0], t[-1], round(tot_t)*sf)
            xl_global = interp1d(t, xl_global)(tt) 

            ## calculate PSD using FFT as Bick
            Ns = len(xl_global)
            xdft = fft(xl_global)
            xdft = xdft[0:floor(Ns/2)]
            psdx = (1/(sf*Ns)) * np.square(np.abs(xdft))
            psdx[1:-1]= 2*psdx[1:-1]
            psdv = psdx
            freq = np.arange(0, sf/2, sf/Ns)

            # debug, using rfft instead
            #four = rfft(xl_global)
            #psdv = 1/(sf*len(xl_global)) * np.square(np.abs(four))
            #psdv[1:-1] = 2*psdv[1:-1]
            #freq = rfftfreq(len(xl_global), d=1/sf)

            # debug, using plt.psd()
            #psdv, freq = plt.psd(xl_global, Fs=sf, NFFT=2**10)


            idx = (freq>=band[0]) * (freq<=band[1])
            idx = np.where(idx)

            fig = plt.figure()

            plt.loglog(freq, psdv, c='black', label='average')

            PSD_global[i,l] = trapz(psdv[idx], freq[idx]) 


            # compute global PSD
            #bandpower_global_l = bandpower(xl_global, sf, band, modified=modified)
            #PSD_global[i, l] = bandpower_global_l

            # compute region-wise PSD
            for r, region in enumerate(regions):
                # cut off transients
                region_xl = regions_x[r][i][l]
                region_xl_cut = region_xl[inds]

                # regional PSD computation
                #bandpower_rl = bandpower(region_xl_cut, sf, band, modified=modified)
                #PSD_r[r][i][l] = bandpower_rl

                # compute PSD directly from FFT
                ## interpolate signal as in Bick
                tt = np.linspace(t[0], t[-1], round(tot_t)*sf)
                region_xl_cut = np.interp(tt, t, region_xl_cut)

                ## calculate PSD using FFT as Bick
                Ns = len(region_xl_cut)
                xdft = fft(region_xl_cut)
                xdft = xdft[0:floor(Ns/2)]
                psdx = (1/(sf*Ns)) * np.square(np.abs(xdft))
                psdx[1:-1]= 2*psdx[1:-1]
                psdv = psdx
                
                #freq = np.array([i for i in range(0,round(sf/2),round(sf/Ns))])
                freq = np.arange(0, sf/2, sf/Ns)
                
                idx = (freq>=band[0]) * (freq<=band[1])
                idx = np.where(idx)

                plt.loglog(freq, psdv, c=colours[r], label=legends[r])

                PSD_r[r][i][l] = trapz(psdv[idx], freq[idx]) 

    # plot
    axs = plt.gca()
    axs.legend()
    plt.show()

    #plot 
    fig = plt.figure()
    axs = plt.gca()

    # labels and axes limits
    plt.xlabel(f'$t_{{spread}}$')
    plt.ylabel(f'PSD')
    plt.ylim([0,2])

    PSD_global = np.array(PSD_global, dtype=np.float64) 
    PSD_r = np.array(PSD_r, dtype=np.float64) 
    print(np.mean(PSD_global, axis=1))
    
    # plot average
    axs.errorbar(t_stamps, np.mean(PSD_global, axis=1), c='black', label='average', marker='o', linestyle='--', alpha=0.75, yerr=np.std(PSD_global, axis=1), capsize=6, capthick=2)
    
    # plot regions
    for r in range(len(regions)):
        axs.errorbar(t_stamps, np.mean(PSD_r[r], axis=1), c=colours[r], label=legends[r], marker='o', linestyle='--', alpha=0.75, yerr=np.std(PSD_r[r], axis=1), capsize=6, capthick=2)
        print(np.mean(PSD_r[r], axis=1))

    plt.show()

    # we're done
    return fig, axs


# --------------------------------------
# compute average MPC
# --------------------------------------

def average_MPC(solutions, fourier_cutoff, extrema=False):
    # len_rhythms and number of trials 
    _, x0, _ = solutions[0]
    L = x0.shape[0]
    N = x0.shape[1]
    len_rhythms = len(solutions)

    average_MPC = [[] for i in range(len_rhythms)]
    maximas = [[] for i in range(len_rhythms)]
    minimas = [[] for i in range(len_rhythms)]
    for i in range(len_rhythms):
        t, x, y = solutions[i]
        for l in range(L):
            # cut off transients
            xl = x[l]
            inds = [i for i in range(len(t)) if t[i]>fourier_cutoff]
            x_cut = xl[:,inds]

            F = MPC(x_cut)
            n_rows, n_cols = F.shape


            # compute average strength
            average_strength = 0
            maxima = 0
            minima = 1
            for c in range(n_cols):
                    for r in range(c+1, n_cols): 
                        average_strength += F[r,c]
                        if extrema:
                            if F[r,c] > maxima:
                                maxima = F[r,c]
                            if F[r,c] < minima:
                                minima = F[r,c]
            average_strength /= N*(N-1)/2

            average_MPC[i].append(average_strength)

            if extrema:
                maximas[i].append(maxima)
                minimas[i].append(minima)
                
    # we're done
    if extrema:
        return average_MPC, maximas, minimas
    else:
        return average_MPC

# --------------------------------------
# compute average MPC matrices
# returns avg_F for which avg_F[b,i]
# conntains average (across trials)
# functional connectome for band b at
# progression i
# --------------------------------------
def functional_connectomes(solutions, fourier_cutoff=10, extrema=False, bands=False, \
         n_epochs=1, l_epochs=10, method="MPC"):
    from fourier import butter_bandpass_filter
    # len_rhythms and number of trials 
    if not bands:
        bands = [[]]
    _, x0, _ = solutions[0]
    L = x0.shape[0]
    N = x0.shape[1]
    len_rhythms = len(solutions)

    avg_F = np.empty((len(bands),len_rhythms, L, N, N))
    print(f'\nBeginning computing functional connectomes...')

    # We have to separate different methods as they distinguish bands in different ways
    if method == "MPC" or method == "PLI":
        for bi, b in enumerate(bands):
            # iterate through time points
            for i in range(len_rhythms):
                t, x, y = solutions[i]
                s_freq = 1/(t[1]-t[0])
                # iterate through trials
                for l in range(L):
                    # cut off transients 
                    xl = x[l]
                    total_length = n_epochs * l_epochs
                    inds = np.where(t > t[-1] - total_length)[0] 
                    x_cut = xl[:,inds]
                    if b:
                        x_cut = butter_bandpass_filter(x_cut, b[0], b[1], s_freq)

                    # iterate through epochs 
                    epochs = np.array_split(x_cut, n_epochs,axis=1)
                    F_epochs = np.empty((n_epochs, N, N))
                    for ee, epoch in enumerate(epochs):
                        # compute functional connectome
                        if method == "MPC":
                            F_epochs[ee] = MPC(epoch)
                        elif method == "PLI":
                            F_epochs[ee] = PLI(epoch)
                    print(f'\tcomputed functional connectome for band {b} Hz, time point {i}, trial {l}')
                    # store average over epoch
                    avg_F[bi,i,l] = np.mean(F_epochs,axis=0)

                # append average functional connectome (over trials)
                #avg_F[bi, i] = np.average(Fs, axis=0)
    elif method == "IC":
        for i in range(len_rhythms):
            t, x, y = solutions[i]
            s_freq = 1/(t[1]-t[0])
            Fss = [np.empty((L,N,N)) for _ in range(len(bands))]
            for l in range(L):
                # cut off transients
                xl = x[l]
                inds = [i for i in range(len(t)) if t[i]>fourier_cutoff]
                x_cut = xl[:,inds]

                # compute functional connectome for each band
                Fs = im_coherence(x_cut, s_freq, bands)
                for b, band in enumerate(bands):
                    Fss[b][l] = Fs[b]
                    avg_F[b,i,l] = Fs[b]
                    print(f'\tcomputed functional connectome for b={b}, i = {i}, l = {l}')

    # we're done
    return avg_F

# --------------------------------------
# plot average MPC matrices
# --------------------------------------
def plot_functional_connectomes(avg_F, t_stamps=False, bands=[], region_names=False, \
        colours=False, regions=False, coordinates=False, vmax=False, title=False, \
        edge_threshold='90.0%'):
    from itertools import chain
    from nilearn import plotting
    from matplotlib.colors import ListedColormap
    # check if we have a single connectome
    if len(avg_F.shape) == 2:
        avg_F = np.array([[[avg_F]]]) 

    # initialize
    B, I, L, N, N = avg_F.shape
    figs = []
    brain_figs = []
    
    # if colours, rearrange by node instead of region
    node_colours = []
    if colours is not False and regions is not False:
        for node in range(N):
            for r, region in enumerate(regions):
                if node in region:
                    node_colours.append(colours[r])
    else:
        node_colours = ['blue' for _ in range(N)]
        

    # if regions, reorganize matrices in the order of regions 2D list
    if regions is not False:
        node_map = list(chain(*regions))
        for b in range(B):
            for t in range(I):
                for l in range(L):
                    i = 0
                    for region in regions:
                        for node in region:
                            avg_F[b,t,l][[i,node], [i,node]] = avg_F[b,t,l][[node,i], [node,i]]
                            i += 1
                            if i == N:  # if node is in two regions, we need to break
                                break
    else:
        node_map = [n for n in range(N)]

    # rearrange region names after region
    if region_names is not False:
        new_region_names = []
        for n in range(N):
            new_region_names.append(region_names[node_map[n]])
    
    # iterate through each band and time point
    for b in range(B):
        if not vmax:
            vmax = np.amax(avg_F[b])
        for i in reversed(range(I)):
            # set plotting settings
            fig = plt.figure() 
            if title:
                plt.title(title)
            elif len(bands) and len(t_stamps):
                plt.title(f'band = {bands[b]}, t = {round(t_stamps[i],1)}')
            else:
                plt.title(f'b = {b}, i = {i}')

            # compute average functional matrix
            F = np.mean(avg_F[b,i], axis=0)

            # plot functional matrix as heatmap, either with regions names or without
            if region_names is not False:
                heatmap = sns.heatmap(F, xticklabels=new_region_names, yticklabels=new_region_names, \
                        vmin=0, vmax=vmax)
                heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 4)
                heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 4)
                if colours is not False:
                    for i, ticklabel in enumerate(heatmap.xaxis.get_majorticklabels()):
                        ticklabel.set_color(node_colours[node_map[i]])
                    for i, ticklabel in enumerate(heatmap.yaxis.get_majorticklabels()):
                        ticklabel.set_color(node_colours[node_map[i]])
            else:
                heatmap = sns.heatmap(F, vmin=0, vmax=vmax)

            # append figure to list of figures
            figs.append(fig)
            #plt.close()

	    # map functional connectome unto brain slices
            brain_map = None
            if coordinates is not False:
                # brain map settings
                node_size = 20
                cmap = ListedColormap(sns.color_palette("rocket"),1000)
                cmap = plt.get_cmap('magma')
                alpha_brain = 0.5
                alpha_edge = 0.5
                colorbar = True

                brain_map = plotting.plot_connectome(F, coordinates, edge_threshold=edge_threshold, \
                         node_color=node_colours, \
                        node_size=node_size, edge_cmap=cmap, edge_vmin=np.amin(F), edge_vmax=vmax, \
                        alpha=alpha_brain, colorbar=colorbar, edge_kwargs={'alpha':alpha_edge})
                #brain_map.close() 
            # append figure to list of figures
            brain_figs.append(brain_map)

    # we're done
    return figs, brain_figs

 
# --------------------------------------
# plot global functional connectivity
# using functional_connectome() output
# one for each frequency band
# --------------------------------------
def plot_global_functional(t_stamps, avg_F, bands, regions=False, lobe_names=False, colours=False, normalize=False, wiggle=0):
    import networkx as nx
    import graph_tool.all as gt
    from networkx.algorithms.community import greedy_modularity_communities, modularity
    from math import floor

    # initialize
    B, I, L, N, N = avg_F.shape
    global_functionals = [[[] for l in range(L)] for _ in range(len(bands))]
    avg_clsts = [[[] for l in range(L)] for _ in range(len(bands))]
    avg_paths = [[[] for l in range(L)] for _ in range(len(bands))]
    mods = [[[] for l in range(L)] for _ in range(len(bands))]
    syncs = [[[] for l in range(L)] for _ in range(len(bands))]
    if regions:
        regions_functionals = [[[[] for l in range(L)] for _ in range(len(regions))] for _ in range(len(bands))]
    figs = []
    plt.style.use("seaborn-muted")
    sns.despine()
    
    #  wiggle points in x-direction
    if regions:
        wiggle = (t_stamps[-1] - t_stamps[0])*wiggle/(2*len(regions))
        wiggled = [np.array(t_stamps) + (i - len(regions)/2)*wiggle for i in range(len(regions)+1)]
    else:
        wiggle = (t_stamps[-1] - t_stamps[0])*wiggle/(2*N)
        wiggled = [np.array(t_stamps) + (i - N/2)*wiggle for i in range(N+1)]

    # iterate through bands
    for b in range(B):
        # iterate through disease progression
        for i in range(I):
            print(f'Computing network metrics for band {b+1} of {B} for iterate {i+1} of {I}')
            for l in range(L):
                # compute global functional connectivity
                F = avg_F[b,i,l]
                global_functional = np.mean(F)
                global_functionals[b][l].append(global_functional)

                # compute global clustering coefficient
                ## Linda's version
                avg_clst = np.mean(clustering_coefficient(F))
                #print(f'original clst = {avg_clst}')

                ## graph-tool global clustering
                #G = to_graph_tool(F)
                #avg_clst = gt.global_clustering(G, weight=G.edge_properties['weight'])

                ## networkx version
                #G = nx.from_numpy_matrix(F)
                #G.edges(data=True)
                #avg_clst = nx.average_clustering(G, weight='weight')
                #print(f'networkx clust = {avg_clst}')


                # compute weighted average shortest path length
                G = nx.from_numpy_matrix(F)
                G.edges(data=True)
                M = G.number_of_edges()
                avg_path = nx.average_shortest_path_length(G, weight='weight')

                # normalization per random shuffling
                clst_norm = 1
                path_norm = 1
                surN = 100
                if normalize:
                    clst_norm = 0
                    path_norm = 0
                    F_triu = np.triu(F)
                    nonzero = np.nonzero(F_triu)                        
                    for j in range(surN):
                        #print(f'surrogate {j} of {surN}')
                        #nF = np.copy(F)
                        #indsM = np.nonzero(nF)
                        #
                        #inds = []
                        #for p in range(len(indsM[0])):
                        #    ind = [indsM[0][p], indsM[1][p]]
                        #    new_ind = [ min(ind), max(ind) ]
                        #    if new_ind not in inds:
                        #        inds.append(new_ind)
                        #inds = np.array(inds)
                        ##inds = np.unique(inds)
                        #
                        #np.random.shuffle(inds)
                        #for p in range(len(inds)-1):
                        #    n1,m1 = inds[p]
                        #    n2,m2 = inds[p+1]
                        #    nF1 = nF[n1,m1]
                        #    nF2 = nF[n2,m2]

                        #    nF[n1,m1] = nF2
                        #    nF[m1,n1] = nF2

                        #    nF[n2,m2] = nF1
                        #    nF[m2,n2] = nF1
                        #    

                        # randomly switch edge weights
                        indices = np.random.permutation(nonzero[0].size) 
                        
                        new_nonzero = (nonzero[0][indices], nonzero[1][indices])

                        nF_triu = np.copy(F_triu)
                        nF_triu[nonzero] = nF_triu[new_nonzero]
                        nF = np.maximum(nF_triu, nF_triu.transpose())
                        nG = to_graph_tool(nF)

                        # my own version
                        clst_sur = np.mean(clustering_coefficient(nF))
                        clst_norm += clst_sur
                        #print(f'clustering = {clst_sur}')
                        
                        ## networkx version clustering                       
                        #nG = nx.double_edge_swap(G, nswap=1, max_tries=N**2)
                        #clst_norm += nx.average_clustering(nG, weight='weight')
                        #path_norm += nx.average_shortest_path_length(nG, weight='weight')

                        edge_weights = nG.edge_properties['weight']
                        # graph-tool version clustering
                        #clst_norm += gt.global_clustering(nG, weight=edge_weights)[0]
                        
                        # shortest path
                        dist = gt.shortest_distance(nG, weights=edge_weights)
                        ave_path_length = sum([sum(i) for i in dist])/(nG.num_vertices()**2-nG.num_vertices())
                        path_norm += ave_path_length
                    clst_norm /= surN
                    path_norm /= surN
                    #print(f'clst_norm = {clst_norm}')
                avg_clsts[b][l].append(avg_clst/clst_norm)
                avg_paths[b][l].append(avg_path/path_norm)

                # compute optimal modularity score
                comms = greedy_modularity_communities(G, weight='weight')
                mod = modularity(G, comms, weight='weight')
                mods[b][l].append(mod)

                # compute synchronizability
                Lap = nx.laplacian_matrix(G, weight='weight').toarray()
                eigs, vecs = np.linalg.eig(Lap)
                eigs = sorted(eigs)
                sync = eigs[1] / eigs[-1]
                syncs[b][l].append(sync)

                # compute regional average
                if regions:
                    for r, region in enumerate(regions):
                        avg_region = 0
                        for node in region:
                            avg_region += np.mean(F[node,:])
                        avg_region /= len(region)
                        regions_functionals[b][r][l].append(avg_region)


        # plot band
        fig = plt.figure()
        mean = np.mean(global_functionals[b],axis=0)
        std = np.std(global_functionals[b],axis=0)
        plt.plot(wiggled[-1], mean, '-o', label='global mean', c='black', alpha=0.75)
        plt.fill_between(wiggled[-1], mean-std, mean+std, alpha=0.2, color='black')
        #plt.errorbar(t_stamps, np.mean(global_functionals[b],axis=0), marker='o', \
        #        yerr=np.std(global_functionals[b], axis=0), \
        #        label=f'{bands[b][0]} - {bands[b][1]} Hz',c='black', capsize=6, capthick=2)
        if regions:
            for r, region in enumerate(regions):
                mean = np.mean(regions_functionals[b][r],axis=0)
                std = np.std(regions_functionals[b][r],axis=0)
                plt.plot(wiggled[r], mean, '-o', label=lobe_names[r], c=colours[r], alpha=0.75)
                plt.fill_between(wiggled[r], mean-std, mean+std, alpha=0.2, color=colours[r])
                #plt.errorbar(wiggled[r], np.mean(regions_functionals[b][r],axis=0), \
                #        yerr=np.std(regions_functionals[b][r],axis=0), marker='o', \
                #        label=lobe_names[r], c=colours[r], capsize=6, capthick=2)
        # plot settings
        plt.ylabel(f"Global functional connectivity\n{bands[b][0]} - {bands[b][1]} Hz")
        plt.xlabel("Time (years)")
        plt.legend()
        figs.append(fig)

        # plot average clustering
        mean = np.mean(avg_clsts[b],axis=0)
        std = np.std(avg_clsts[b],axis=0)

        fig = plt.figure()
        plt.plot(t_stamps, mean, '-o', label=f'{bands[b][0]} - {bands[b][1]} Hz')
        plt.fill_between(t_stamps, mean-std, mean+std, alpha=0.2)
        #plt.errorbar(t_stamps, np.mean(avg_clsts[b],axis=0), yerr=np.std(avg_clsts[b],axis=0), \
                #marker='o', label=f'{bands[b][0]} - {bands[b][1]} Hz',c='black')
        # plot settings
        plt.ylabel(f"Global weighted clustering")
        plt.xlabel("Time (years)")
        plt.legend()
        figs.append(fig)

        # plot average path
        mean = np.mean(avg_paths[b],axis=0)
        std = np.std(avg_paths[b],axis=0)

        fig = plt.figure()
        plt.plot(t_stamps, mean, '-o', label=f'{bands[b][0]} - {bands[b][1]} Hz')
        plt.fill_between(t_stamps, mean-std, mean+std, alpha=0.2)
        #plt.errorbar(t_stamps, np.mean(avg_paths[b],axis=0), yerr=np.std(avg_paths[b],axis=0), \
        #        marker='o', label=f'{bands[b][0]} - {bands[b][1]} Hz',c='black')
        # plot settings
        plt.ylabel(f"Global weighted path length")
        plt.xlabel("Time (years)")
        plt.legend()
        figs.append(fig)

        # plot modularity
        mean = np.mean(mods[b],axis=0)
        std = np.std(mods[b],axis=0)

        fig = plt.figure()
        plt.plot(t_stamps, mean, '-o', label=f'{bands[b][0]} - {bands[b][1]} Hz')
        plt.fill_between(t_stamps, mean-std, mean+std, alpha=0.2)
        #plt.errorbar(t_stamps, np.mean(mods[b],axis=0), yerr=np.std(mods[b],axis=0), \
        #        marker='o', label=f'{bands[b][0]} - {bands[b][1]} hz',c='black')
        # plot settings
        plt.ylabel(f"optimal modularity")
        plt.xlabel("Time (years)")
        plt.legend()
        figs.append(fig)

        # plot synchronizability
        mean = np.mean(syncs[b],axis=0)
        std = np.std(syncs[b],axis=0)

        fig = plt.figure()
        plt.plot(t_stamps, mean, '-o', label=f'{bands[b][0]} - {bands[b][1]} Hz')
        plt.fill_between(t_stamps, mean-std, mean+std, alpha=0.2)
        #plt.errorbar(t_stamps, np.mean(syncs[b],axis=0), yerr=np.std(syncs[b],axis=0), \
        #        marker='o', label=f'{bands[b][0]} - {bands[b][1]} hz',c='black')
        # plot settings
        plt.ylabel(f"Synchronizability")
        plt.xlabel("Time (years)")
        plt.legend()
        figs.append(fig)
    # we're done
    return figs

# ---------------------------------------
# plot spreading
# Input
#   regions : list of lists, each list contains nodes to average over
# --------------------------------------

def plot_spreading(sol, colours, legends, xlimit=False, regions=[], averages=True, plot_c=False):
    # extract solution
    a = sol['a']
    b = sol['b']
    c = sol['c']
    qu = sol['qu']
    qv = sol['qv']
    u = sol['u']
    v = sol['v']
    up = sol['up']
    vp = sol['vp']
    w = sol['w']
    t = sol['t']

    # N of x-ticks
    nx = 5

    # find N
    N = a.shape[0]

    # if regions not given, plot all nodes
    if len(regions) == 0:
        regions = [[i] for i in range(N)]

    # plot 1-by-2 plot of all nodes'/regions a and b against time
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions tau and Abeta damage
    fig2, axs2 = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions toxic tau and Abeta concentration
    fig3, axs3 = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of average weight
    fig4, axs4 = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions healthy tau and Abeta concentration
    fig5, axs5 = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plot settings
    if xlimit:
        plt.xlim((0, xlimit))
    axs[0].set_xlabel('$t_{spread}$')
    axs[0].set_ylabel('Excitatory semiaxis, $a$')
    axs[1].set_xlabel('$t_{spread}$')
    axs[1].set_ylabel('Inhibitory semiaxis, $b$')

    axs2[0].set_xlabel('$t_{spread}$')
    axs2[0].set_ylabel('Amyloid-$\\beta$ damage, $q^{(\\beta)}$')
    axs2[1].set_xlabel('$t_{spread}$')
    axs2[1].set_ylabel('Tau damage, $q^{(\\tau)}$')
    axs2[0].set_ylim([-0.1, 1.1])
    axs2[1].set_ylim([-0.1, 1.1])

    axs3[0].set_xlabel('$t_{spread}$')
    axs3[0].set_ylabel('Toxic amyloid-$\\beta$ concentration, $\\tilde{u}$')
    axs3[1].set_xlabel('$t_{spread}$')
    axs3[1].set_ylabel('Toxic tau concentration, $\\tilde{v}$')

    axs5[0].set_xlabel('$t_{spread}$')
    axs5[0].set_ylabel('Healthy amyloid-$\\beta$ concentration, $u$')
    axs5[1].set_xlabel('$t_{spread}$')
    axs5[1].set_ylabel('Healthy tau concentration, $v$')

    # plot a, b, damage and concentrations against time
    for r in range(len(regions)):
        # initialize
        region = regions[r]
        avg_region_a = []
        avg_region_b = []
        avg_region_c = []
        avg_region_qu = []
        avg_region_qv = []
        avg_region_up = []
        avg_region_vp = []
        avg_region_u = []
        avg_region_v = []

        # compute averages over regions
        for node in region:
            avg_region_a.append(a[node,:])
            avg_region_b.append(b[node,:])
            avg_region_c.append(c[node,:])
            avg_region_qu.append(qu[node,:])
            avg_region_qv.append(qv[node,:])
            avg_region_up.append(up[node,:])
            avg_region_vp.append(vp[node,:])
            avg_region_u.append(u[node,:])
            avg_region_v.append(v[node,:])

        # convert lists to arrays
        avg_region_a = np.array(avg_region_a)
        avg_region_b = np.array(avg_region_b)
        avg_region_c = np.array(avg_region_c)
        avg_region_qu = np.array(avg_region_qu)
        avg_region_qv = np.array(avg_region_qv)
        avg_region_up = np.array(avg_region_up)
        avg_region_vp = np.array(avg_region_vp)
        avg_region_u = np.array(avg_region_u)
        avg_region_v = np.array(avg_region_v)

        # plot a, b
        axs[0].plot(t, np.mean(avg_region_a, axis=0), c=colours[r], label=legends[r])
        axs[1].plot(t, np.mean(avg_region_b, axis=0), c=colours[r], label=legends[r])
        if plot_c:
            axs[1].plot(t, np.mean(avg_region_c, axis=0), c=colours[r], label=legends[r])

        # plot damage
        axs2[0].plot(t, np.mean(avg_region_qu, axis=0), c=colours[r], label=legends[r])
        axs2[1].plot(t, np.mean(avg_region_qv, axis=0), c=colours[r], label=legends[r])

        # plot concentration
        axs3[0].plot(t, np.mean(avg_region_up, axis=0), c=colours[r], label=legends[r])
        axs3[1].plot(t, np.mean(avg_region_vp, axis=0), c=colours[r], label=legends[r])

        axs5[0].plot(t, np.mean(avg_region_u, axis=0), c=colours[r], label=legends[r])
        axs5[1].plot(t, np.mean(avg_region_v, axis=0), c=colours[r], label=legends[r])

    # plot averages over all nodes
    if averages:
        # a and b
        axs[0].plot(t, np.mean(a, axis=0), c='black', label='average')
        axs[1].plot(t, np.mean(b, axis=0), c='black', label='average')
        if plot_c:
            axs[1].plot(t, np.mean(c, axis=0), c='black', label='average')

        # damage
        axs2[0].plot(t, np.mean(qu, axis=0), c='black', label='average')
        axs2[1].plot(t, np.mean(qv, axis=0), c='black', label='average')

        # toxic concentratio
        axs3[0].plot(t, np.mean(up, axis=0), c='black', label='average')
        axs3[1].plot(t, np.mean(vp, axis=0), c='black', label='average')

        # healthy concentration
        axs5[0].plot(t, np.mean(u, axis=0), c='black', label='average')
        axs5[1].plot(t, np.mean(v, axis=0), c='black', label='average')


    # plot average weights over time
    axs4.plot(t, np.mean(w, axis=0), c='black')
    axs4.set_ylabel('Average link weight')
    axs4.set_xlabel('$t_{spread}$')

    # show
    axs[1].legend(loc='best')
    axs3[0].legend(loc='best')
    plt.tight_layout()

    # we're done
    figs = (fig, fig2, fig3, fig4)   
    axss = (axs, axs2, axs3, axs4)
    return figs, axss

# ---------------------------------------
# plot spreading
# Input
#   regions : list of lists, each list contains nodes to average over
# --------------------------------------

def plot_glioma_spreading(sol, colours, legends, xlimit=False, regions=[], averages=True):
    # extract solution
    a = sol['a']
    b = sol['b']
    u = sol['u']
    up = sol['up']
    qu = sol['qu']
    w = sol['w']
    t = sol['t']

    # N of x-ticks
    nx = 5

    # find N
    N = a.shape[0]

    # plot 1-by-2 plot of all nodes'/regions a and b against time
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions damage
    fig2, axs2 = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of average weight
    fig4, axs4 = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions tumor concentration
    fig5, axs5 = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plot settings
    if xlimit:
        plt.xlim((0, xlimit))
    axs[0].set_xlabel('$t_{spread}$')
    axs[0].set_ylabel('Excitatory semiaxis, $a$')
    axs[1].set_xlabel('$t_{spread}$')
    axs[1].set_ylabel('Inhibitory semiaxis, $b$')
    #axs[0].set_ylim([-0.1, 2])
    #axs[1].set_ylim([-0.1, 2])

    axs2.set_xlabel('$t_{spread}$')
    axs2.set_ylabel('Damage, $q$')
    axs2.set_ylim([-0.1, 1.1])

    axs5.set_xlabel('$t_{spread}$')
    axs5.set_ylabel('Tumor concentration, $u$')

    # plot a, b, damage and concentrations against time
    for r in range(len(regions)):
        # initialize
        region = regions[r]
        avg_region_a = []
        avg_region_b = []
        avg_region_qu = []
        avg_region_u = []
        avg_region_up = []

        # compute averages over regions
        for node in region:
            avg_region_a.append(a[node,:])
            avg_region_b.append(b[node,:])
            avg_region_qu.append(qu[node,:])
            avg_region_u.append(u[node,:])
            avg_region_up.append(up[node,:])

        # convert lists to arrays
        avg_region_a = np.array(avg_region_a)
        avg_region_b = np.array(avg_region_b)
        avg_region_qu = np.array(avg_region_qu)
        avg_region_u = np.array(avg_region_u)
        avg_region_up = np.array(avg_region_up)

        # plot a, b
        axs[0].plot(t, np.mean(avg_region_a, axis=0), c=colours[r], label=legends[r])
        axs[1].plot(t, np.mean(avg_region_b, axis=0), c=colours[r], label=legends[r])

        # plot damage
        axs2.plot(t, np.mean(avg_region_qu, axis=0), c=colours[r], label=legends[r])

        # plot concentration
        axs5.plot(t, np.mean(avg_region_up, axis=0), c=colours[r], label=legends[r])

    # plot averages over all nodes
    if averages:
        # a and b
        axs[0].plot(t, np.mean(a, axis=0), c='black', label='average')
        axs[1].plot(t, np.mean(b, axis=0), c='black', label='average')

        # damage
        axs2.plot(t, np.mean(qu, axis=0), c='black', label='average')

        # healthy concentration
        axs5.plot(t, np.mean(up, axis=0), c='black', label='average')


    # plot average weights over time
    axs4.plot(t, np.mean(w, axis=0), c='black')
    axs4.set_ylabel('Average link weight')
    axs4.set_xlabel('$t_{spread}$')

    # show
    axs[1].legend(loc='best')
    plt.tight_layout()

    # we're done
    figs = (fig, fig2, fig5, fig4)   
    axss = (axs, axs2, axs5, axs4)
    return figs, axss

# -----------------------------------------
# Jaccard index per Forrester's thesis
# ---------------------------------------

def jaccard(M1, M2):
    # minimum sum (num) and maximum sum (den)
    num = 0
    den = 0

    # make sure input is in numpy format
    M1 = np.array(M1)
    M2 = np.array(M2)

    # check if matrices have same shape
    if M1.shape != M2.shape:
        print('\nError: Cannot compute Jaccard index as array\'s shape are different.\n')
        return None
    
    # compute index
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            num += min(M1[i,j],M2[i,j])
            den += max(M1[i,j],M2[i,j])

    # we're done
    return num/den

# ----------------------------------------
# Matrix of delays. Input for rhythms series
# ----------------------------------------

def delay_matrix(distances, transmission_speed, N, discretize=40):
    # distances should be a list of 3-tuples like (node1, node2, distance)
    delay_matrix = np.zeros((N,N))
    for n,m,distance in distances:
        delay_matrix[n,m] = distance/transmission_speed
        delay_matrix[m,n] = delay_matrix[n,m]

    if discretize:
        nonzero_inds = np.nonzero(delay_matrix)
        max_delay = np.amax(delay_matrix)
        n_delays = np.count_nonzero(delay_matrix)

        lower_bounds = np.arange(0, discretize) * max_delay/discretize
        upper_bounds = np.zeros((discretize))
        for i in range(discretize):
            upper_bounds[i] = (i+1) * max_delay/discretize
        for l in range(len(nonzero_inds[0])):
            i = nonzero_inds[0][l]
            j = nonzero_inds[1][l]
            for k in range(discretize):
                w_ij = delay_matrix[i,j]
                if w_ij <= upper_bounds[k]: 
                    w_ij = upper_bounds[k]  # round to upper (GorielyBick)
                    delay_matrix[i,j] = w_ij
                    delay_matrix[j,i] = w_ij
                    break

    return delay_matrix
    
# ----------------------------------------
# Compute degree distribution of amplitudes
# return bins which is a T x L matrix
# where each entry is a list of all the
# amplitudes
# ----------------------------------------
def amplitude_distr(solutions, cutoff=10):
    from scipy.signal import argrelextrema

    # find size of rhythms
    _, x0, _ = solutions[0]
    L = x0.shape[0]
    N = x0.shape[1]
    spread_T = len(solutions) 

    # a matrix for each simulation, each entry a list of global amplitude values
    #bins = np.empty((spread_T,L))
    bins = [[[] for _ in range(L)] for _ in range(spread_T)]

    # iterate through each spread time
    for i, (t, xs, ys) in enumerate(solutions):
        # indices for cutoff
        inds = [i for i in range(len(t)) if t[i]>cutoff]
        t = t[inds]
        # iterate through each trial
        for j, x in enumerate(xs):
            # initialize collection of amplitude bin assortments
            extremas = np.array( [] )
            # iterate through each node
            for xi in x:
                # cut off transient time
                xi = xi[inds]

                # find extrema in node i
                peak_ind = argrelextrema(xi, np.greater)
                valley_ind = argrelextrema(xi, np.less)
                peaks = xi[peak_ind]
                valleys = xi[valley_ind]
                extrema = np.concatenate( (np.abs(peaks), np.abs(valleys)) )

                # concatenate extrema/peak values
                #bin_numbers = np.concatenate( (bin_numbers, extrema) )
                extremas = np.concatenate( (extremas, np.abs(peaks)) )  # only considering peaks in case oscillation off-centered


            # assert extrema/peak values into matrix
            bins[i][j] = extremas

    # we're done return bins
    return bins

def plot_amplitude_distr(bins, t_stamps=False, xlim=[0,30], binwidth=1):
    figs = []
    T = len(bins)
    L = len(bins[0])

    # iterate through disease progression (reversed for plt.show)
    for Ti in reversed(range(T)):
        metabin = np.array([])
        # iterate through trials and concatenate all trials
        for l in range(L):
            metabin = np.concatenate([metabin, bins[Ti][l]])

        # create histogram of amplitudes
        fig = plt.figure()
        sns.histplot(data=metabin, binwidth=binwidth, stat='probability')
        figs.append(fig)
        if t_stamps.any():
            plt.title(f't = {t_stamps[Ti]}')
        else:
            plt.title(f't = {Ti}')
        plt.ylabel('Portion of amplitudes')
        plt.xlabel('Amplitude')
        plt.xlim(xlim)
    
    # we're done
    return figs


# ------------------------------------------
# portion_ampl
# 
# find portion of amplitudes above p
# from bins in the above the function#
# above
# returns TxL matrix of real numbers
# denoting the portion
# -----------------------------------------

def portion_ampl(bins, p):
    # initialize
    T = len(bins)
    L = len(bins[0])
    portions = [[None for _ in range(L)] for _ in range(T)]

    # loop through spreading time and trials
    for t in range(T):
        for l in range(L):
            # find portion of elements in bin larger than p
            bini = np.array(bins[t][l])
            portion = np.count_nonzero(bini > p)
            portion *= 1/bini.size
            portions[t][l] = portion

    # we're done
    return portions

# ---------------------------------------
# plot_portions
# plot above portions
# ------------------------------------------

def plot_portions(portions, t_stamps):
    y = np.mean(portions, axis=1)
    var = np.std(portions, axis=1)
    x = t_stamps

    plt.style.use('seaborn-muted')
    sns.despine()
    fig = plt.errorbar(x, y, yerr=var)
    plt.ylabel(f'Portion of extrama above $p$')
    plt.xlabel('$t_{{spread}}$')
    
    return fig

# ----------------------------------------------------
# freq-distrb returns a T x L x N containing the
# global frequency peaks (0-80 Hz) 
# to be used for plotting frequency distributions
# ----------------------------------------------------

def freq_distr(sol, cutoff=10, freq_tol=0):
    from fourier import power_spectrum, bandpower, frequency_peaks
    _, x0, _ = sol[0]
    L = x0.shape[0]
    N = x0.shape[1]
    T = len(sol) 

    freq_peaks = [ [[] for _ in range(L)] for _ in range(T) ]

    # iterate disease progression
    for Ti in range(T):
        t, x, y= sol[Ti]

        # iterate through trials
        for l in range(L):
            xl = x[l]

            # cut off transient time
            inds = [i for i in range(len(t)) if t[i]>cutoff]
            t = t[inds]
            x_cut = xl[:,inds]
            tot_t = t[-1] - t[0]
            sf = len(x_cut[0])/tot_t

            # iterate through nodes
            for n in range(N):
                freq_peak = frequency_peaks(x_cut[n], sf, band=[0,80], tol=freq_tol)
                freq_peaks[Ti][l].append(freq_peak)

    return freq_peaks

def plot_freq_distr(freq_peaks, t_stamps=False, binwidth=1, xlim=[-1,15]):
    # initialize
    plt.style.use('seaborn-muted')
    sns.despine()
    figs = []
    T = len(freq_peaks)
    freq_peaks = np.array(freq_peaks)

    # iterate disease progression (reversed to plt.show() is in the correct order)
    for Ti in reversed(range(T)):
        # compute average over trials at time Ti
        freq_peak = np.nanmean(freq_peaks[Ti,:,:], axis=0)
        if np.isnan(freq_peak).all():
            freq_peak = []

        # create histogram of mean frequency distribution
        fig = plt.figure()
        sns.histplot(data=freq_peak, binwidth=binwidth, stat='probability')
        figs.append(fig)
        if t_stamps.any():
            plt.title(f't = {t_stamps[Ti]}')
        else:
            plt.title(f't = {Ti}')
        plt.ylabel('Portion of frequency peaks')
        plt.xlabel('Frequency (Hz)')
        plt.xlim(xlim)

    return figs

# ---------------------------
# Creates undirected graph-tool graph from weighted adjacency matrix
# -----------------------
def to_graph_tool(adj):
    import graph_tool.all as gt
    g = gt.Graph(directed=False)
    edge_weights = g.new_edge_property('double')
    g.edge_properties['weight'] = edge_weights
    nnz = np.nonzero(np.triu(adj,1))
    nedges = len(nnz[0])
    g.add_edge_list(np.hstack([np.transpose(nnz),np.reshape(adj[nnz],(nedges,1))]),eprops=[edge_weights])
    return g


# ----------------------
# computes clustering coefficient per
# VU MEG network analysis studies.
# input - symmetric (N,N) numpy matrix
# output - (N,) numpy array (each element is the clustering \
# coefficient of node N
# ----------------------------
def clustering_coefficient(W):
    # find nonzero indices
    N = W.shape[0]
    clusts = np.empty((N,))
    for n in range(N):
        # find neighbours of n)
        row = W[n,:]
        inds = np.nonzero(W[n,:])

        # clustering is 0 if only one or zero neighbours
        n_neighbours = inds[0].size
        if n_neighbours < 2:
            clusts[n] = 0
            continue

        # make link matrix, L, where L_ij = w_ni * w_nj
        neigh_mat = np.zeros((n_neighbours, n_neighbours))
        neigh_mat[0,:] = row[inds]
        link_mat = neigh_mat.T @ neigh_mat
        np.fill_diagonal(link_mat, 0)

        # sum of link matrix is the denominator
        denom = np.sum(link_mat)

        # multiply link matrix by the upper weighted adjacency matrix
        triangle_matrix = link_mat * W[np.ix_(inds[0], inds[0])]
        num = np.sum(triangle_matrix)

        # set the clustering
        clusts[n] = num / denom
    
    return clusts

# -------------------------------------------------------------------------
# Here we plot the incidence of epileptiform
# spikes and their duration in a connectome timeseries. 
# This function finds the center of the oscillations
# and defines a spike as an user-deined deviation from that
# center. Duration is defined as 67% percent decrease from spike peak.
# INPUT:
#   xs   numpy array (N,I), time series
#   t   numpy array (1,I), times of x above
#   factor  float/int, spikes > factor* relative_crest + center
# OUTPUT:
#   spike_density    numpy array of floats (N), spike density per signal 
#   spike_duration    numpy array of floats (N), spike density per signal 
# -------------------------------------------------------------------------
def spike_density_duration(xs, t, factor=2, distance=None, plot_check=False):
    from scipy.signal import find_peaks
    # total time length
    T = t[-1]
    N, _ = xs.shape

    # initialize spike density
    spike_density = np.empty((N))
    spike_duration = np.empty((N))  
    spike_duration[:] = np.nan  # nan is default duration value

    # iterate through signals
    for n in range(N):
        # extract each signal
        x = xs[n]

        # find the mean value (approximation for center of oscillation)
        center = np.mean(x)

        # find crests and throughs
        crests_ind, _ = find_peaks(x, distance=distance) 
        throughs_ind, _ = find_peaks(-x, distance=distance) 
        crests = x[crests_ind]
        throughs = x[throughs_ind]

        # find median crests and through relative to center
        crest = np.median(crests) - center
        through = center - np.median(throughs)

        # find peaks that are factor*crest higher than center (spikes)
        spike_def = factor * crest + center
        spike_inds, _ = find_peaks(x, height=spike_def)
        #spike_inds, _ = find_peaks(x, height=0.6)  # debug
        spike_density[n] = len(spike_inds) / T

        # if told, show plots to verify peak definition
        if plot_check and len(spike_inds)>0:
            plt.figure()
            plt.plot(t,x)
            plt.axhline(y=center, color='black')
            plt.axhline(y=crest+center, color='red')
            plt.axhline(y=-through+center, color='green')
            plt.axhline(y=spike_def, color='blue')

        # find duration of spikes
        durs = []
        for spike_ind in spike_inds:
            # cut x to after spike
            x_after = x[spike_ind:]

            # try to find the end of duration if not, skip
            try:
                dur_ind = np.argwhere( x_after < 0.33 * x[spike_ind] )[0][0]  # decrease by 67%
                dur_ind = np.argwhere( x_after < center )[0][0]  # pass through center
                dur_ind = np.argwhere( np.diff( x_after < center, prepend=False))[1,0]  # second pass through center
            except:
                print('\tCould not find spike duration')
                continue
            
            # compute time duration and store (for averaging later)
            dur = t[spike_ind+dur_ind] - t[spike_ind]
            durs.append(dur)

            # if told, plot the spike duration
            if plot_check:
                plt.plot(t[spike_ind:spike_ind+dur_ind], x[spike_ind:spike_ind+dur_ind], color='r')

        # store average of spike duration for signal n, if durations were found
        if len(durs) > 0:
            spike_duration[n] = np.mean( np.array(durs) )

        # if plot, show figure
        if plot_check and len(spike_inds)>0:
            plt.show()
            plt.close('all')

    # we're done
    return spike_density, spike_duration

# ----------------------------------------------------------------------
# Here we compute spike densities and durations of from a dynamical solution per 
# glioma() and alzheimer()
# INPUT:
#   sims    list (Ts) of dictionaries with keys 'x' and 't'
#               where 'x' contains numpy array (L,N,I)
#   factor  float/int, spikes > factor* relative_crest + center
# OUTPUT:
#   spike_densities     numpy array (Ts,L,N)
#   spike_durations     numpy array (Ts,L,N)
# ----------------------------------------------------------------------
def spike_density_duration_list(sims, factor=2, distance=None, plot_check=False):
    # extract some info
    Ts = len(sims)
    _, xs0, _ = sims[0]
    L, N, _ = xs0.shape  # number of trials and nodes
    
    # initialize
    spike_densities = np.empty((Ts,L,N))
    spike_durations = np.empty((Ts,L,N))
    spike_durations[:] = np.nan  # nan is default duration value

    # iterate through simulation entries (ex. disease progression)
    for ts in range(Ts):
        # extract trials of time series at ts
        t, xs, _ = sims[ts]    
        
        # iterate through trials
        for l in range(L):
            # time series at trial l
            x = xs[l]

            # compute and store spike density and durations
            spike_densities_i, spike_durations_i = spike_density_duration(x, t, factor=factor, \
                distance=distance, plot_check=plot_check)
            spike_densities[ts,l,:] = spike_densities_i
            spike_durations[ts,l,:] = spike_durations_i
            
    # we're done
    return spike_densities, spike_durations

# ----------------------------------------------------------------------
# Here we plot the spike density of a dynamical solution per 
# glioma() and alzheimer()
# INPUT:
#   spike_densities    numpy array (Ts,L,N)
# OUTPUT:
#   fig     matplotlib figure object
# ----------------------------------------------------------------------
def plot_spike_densities(spike_densities, t_spread, regions=False, region_names=False, colors=False, wiggle=0.05):
    # initialize figure
    fig = plt.figure()

    # axes and labels
    plt.ylabel('Spike density ($s^{-1}$)')
    plt.xlabel('Spreading time (years)')

    # compute global average over trials and nodes
    node_averaged = np.mean(spike_densities, axis=2)
    mean = np.mean(node_averaged, axis=1)
    std = np.std(node_averaged, axis=1)

    # plot average
    plt.plot(t_spread, mean, '-o', c='black', \
            alpha=0.75, label='global mean')
    plt.fill_between(t_spread, mean-std, mean+std, color='black', alpha=0.2)

    # if told, compute and plot averages over regions
    if regions is not False:
        #  wiggle points in x-direction
        if regions:
            i = 1  # used later for wiggling
            try:
                wiggle = (t_spread[-1] - t_spread[0])*wiggle/len(regions)
            except:
                print('Error: Wiggle was not provided')
                return fig

        # iterate through regions
        for r, region in enumerate(regions):
            # extract spikes from region
            region = np.array(region)
            region_spikes = spike_densities[:,:,region]

            # compute mean and std over trials in region
            region_averaged = np.mean(region_spikes,axis=2)
            mean_region = np.mean(region_averaged, axis=1)
            std_region = np.std(region_averaged, axis=1)

            # if region names and colors given, add it
            if region_names is not False:
                label = region_names[r]
            if colors is not False:
                color = colors[r]
        
            # position regions away from each other in plot
            wiggled = np.array(t_spread) + (-1)**(r)*i*wiggle
            if r%2:
                i += 1

            # plot regional spike densities and add legend
            plt.plot(wiggled, mean_region, '-o', c=color, \
                    alpha=0.75, label=label)
            plt.fill_between(wiggled, mean_region-std_region, mean_region+std_region, \
                 color=color, alpha=0.2)
            plt.legend()

    # we're done
    return fig
    
# ----------------------------------------------------------------------
# Here we plot the spike duration of a dynamical solution per 
# glioma() and alzheimer()
# INPUT:
#   spike_densities    numpy array (Ts,L,N)
# OUTPUT:
#   fig     matplotlib figure object
# ----------------------------------------------------------------------
def plot_spike_durations(spike_durations, t_spread, regions=False, region_names=False, colors=False, wiggle=0.05):
    # initialize figure
    fig = plt.figure()

    # axes and labels
    plt.ylabel('Average spike duration (seconds)')
    plt.xlabel('Spreading time (years)')

    # compute global average over trials and nodes
    node_averaged = np.nanmean(spike_durations, axis=2)
    mean = np.nanmean(node_averaged, axis=1)
    std = np.nanstd(node_averaged, axis=1)

    # plot average
    plt.plot(t_spread, mean, '-o', c='black', \
            alpha=0.75, label='global mean')
    plt.fill_between(t_spread, mean-std, mean+std, color='black', alpha=0.2)

    # if told, compute and plot averages over regions
    if regions is not False:
        #  wiggle points in x-direction
        if regions:
            i = 1  # used later for wiggling
            try:
                wiggle = (t_spread[-1] - t_spread[0])*wiggle/len(regions)
            except:
                print('Error: Wiggle was not provided')
                return fig

        # iterate through regions
        for r, region in enumerate(regions):
            # extract spikes from region
            region = np.array(region)
            region_spikes = spike_durations[:,:,region]

            # compute mean and std over trials in region
            region_averaged = np.nanmean(region_spikes,axis=2)
            mean_region = np.nanmean(region_averaged, axis=1)
            std_region = np.nanstd(region_averaged, axis=1)

            # if region names and colors given, add it
            if region_names is not False:
                label = region_names[r]
            if colors is not False:
                color = colors[r]
        
            # position regions away from each other in plot
            wiggled = np.array(t_spread) + (-1)**(r)*i*wiggle
            if r%2:
                i += 1

            # plot regional spike densities and add legend
            plt.plot(wiggled, mean_region, '-o', c=color, \
                    alpha=0.75, label=label)
            plt.fill_between(wiggled, mean_region-std_region, mean_region+std_region, \
                 color=color, alpha=0.2)
            plt.legend()

    # we're done
    return fig

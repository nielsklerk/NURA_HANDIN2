# imports
import numpy as np
import matplotlib.pyplot as plt


def n(
    x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> float | np.ndarray:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : float | ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    x = np.asarray(x)
    n = np.zeros_like(x)
    n[x>0] = A * Nsat * (x[x>0] / b) ** (a-3) * np.exp(-(x[x>0] / b) ** c) # REPLACE BOOL MASK WITH OWN CODE
    return n


##### Integrator block #####


# Below we provide a template for romberg integration
# You can implement this or use another integration method based on some form of Richardson extrapolation
def romberg_integrator(
    func: callable, bounds: tuple, order: int = 5, err: bool = False, args: tuple = ()
) -> float | tuple[float, float]:
    """
    Romberg integration method

    Parameters
    ----------
    func : callable
        Function to integrate.
    bounds : tuple
        Lower- and upper bound for integration.
    order : int, optional
        Order of the integration.
        The default is 5.
    err : bool, optional
        Whether to retun first error estimate.
        The default is False.
    args : tuple, optional
        Arguments to be passed to func.
        The default is ().

    Returns
    -------
    float
        Value of the integral. If err=True, returns the tuple
        (value, err), with err a first estimate of the (relative)
        error.
    """
    a, b = bounds
    h = b - a
    r = np.zeros(order)
    r[0] = 0.5 * h * (func(b, *args) - func(a, *args))
    N_p = 1
    for i in range(1, order):
        r[i] = 0
        delta = np.copy(h)
        h *= 0.5
        x = a + h
        for _ in range(N_p):
            r[i] += func(x, *args)
            x += delta
        r[i] = 0.5 * (r[i-1] + delta * r[i])
        N_p *= 2

    N_p = 1
    for i in range(1, order):
        N_p *= 4
        r[:order - i] = (N_p * r[1:order - i + 1] - r[:order - i]) / (N_p - 1)
    if err:
        return r[0], np.abs(r[0]-r[1])  # (value, error)
    return r[0] # value


#### Sampler block ####

def rng(N):
    global seed
    seed = np.uint(seed)
    a = 4294957665
    a_1 = 21
    a_2 = 35
    a_3 = 4
    rnds = np.zeros(N)
    for i in range(N):
        # MWC 32-bit
        seed = seed & (2**32 - 1)
        seed = a*(seed & (2**32 - 1)) + (seed >> 32)
        seed = seed & (2**32 - 1)

        # 64-bit XOR-shift
        seed = seed ^ (seed>>a_1)
        seed = seed ^ (seed<<a_2)
        seed = seed ^ (seed>>a_3)
        u = seed / np.float64(2**64)
        rnds[i] = u
    if N==1:
        return rnds[0]
    return rnds


def sampler(
    dist: callable,
    min: float,
    max: float,
    Nsamples: int,
    args: tuple = ()
) -> np.ndarray:
    """
    Sample a distribution using sampling method of your choice

    Parameters
    dist : callable
        Distribution to sample
    min :
        Minimum value for sampling
    max : float
        Maximum value for sampling
    Nsamples : int
        Number of samples
    args : tuple, optional
        Arguments of the distribution to sample, passed as args to dist

    Returns
    -------
    sample: ndarray
        Values sampled from dist, shape (Nsamples,)
    """
    sample = np.zeros(Nsamples)
    i = 0
    while i < Nsamples:
        random_x = min + (max-min)*rng(1)
        random_y = rng(1)
        if random_y < dist(random_x, *args):
            sample[i] = random_x
            i += 1
    return sample



#### Sorting block ####


def sort_array(
    arr: np.ndarray,
    inplace: bool = False,
    index = False
) -> np.ndarray:
    """
    Sort a 1D array using a sorting algorithm of your choice

    Parameters
    ----------
    arr : ndarray
        Input array to be sorted
    inplace : bool, optional
        If True, sort the array in-place
        If False, return a sorted copy

    Returns
    -------
    sorted_arr : ndarray
        Sorted array (same shape as arr)

    """
    if inplace:
        sorted_arr = arr
    else:
        sorted_arr = arr.copy()

    N = len(sorted_arr)
    step = 1
    index_array = np.arange(N)

    while step < N:
        for i in range(0, N, step*2):
            l = i
            r = min(i + step, N)
            end = min(i + 2*step, N)

            while l < r and r < end:
                if sorted_arr[l] <= sorted_arr[r]:
                    l += 1
                else:
                    sorted_arr[l:r+1] = np.roll(sorted_arr[l:r+1], 1) # REPLACE ROLL WITH OWN FUNCTION
                    index_array[l:r+1] = np.roll(index_array[l:r+1], 1)
                    l += 1
                    r += 1

        step *= 2
    if index:
        return index_array
    return sorted_arr


def choice(arr: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Choose given number of random elements from an array, without replacement

    Parameters
    ----------
    arr : ndarray
        Array to shuffle
    size : int, optional
        Number of elements to pick from array
        The default is 1

    Returns
    -------
    chosen : ndarray
        Randomly chosen elements from arr, shape (size,)
    """
    N = len(arr)
    random = rng(N)
    random_index = sort_array(random, index=True)
    return arr[random_index][:size].copy()


##### Derivative block #####


def dn_dx(
    x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> float | np.ndarray:
    """
    Analytical derivative of number density provide

    Parameters
    ----------
    x : ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Derivative of number density of
        satellite galaxies at given radius x.
    """
    fraction = x/b
    return  - A * Nsat * fraction**(a-4) * (c * fraction ** c - a + 3) * np.exp(-fraction**c) / b


def finite_difference(
    function: callable, x: float | np.ndarray, h: float
) -> float | np.ndarray:
    """
    A building block to compute derivative using finite differences

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h : float
        Step size for finite difference

    Returns
    -------
    dy : float | ndarray
        Derivative at x
    """
    return 0.5 * (function(x + h) - function(x - h)) / h


def compute_derivative(
    function: callable,
    x: float | np.ndarray,
    h_init: float,
    # For Ridders use parameters below:
    d: float, # Factor by which to decrease h_init every iteration
    eps: float, # Relative error
    max_iters: int = 10, # Maximum number of iterations before exiting
) -> float | np.ndarray:
    """
    Function to compute derivative

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h_init : float
        Initial step size for finite difference

    Returns
    -------
    df : float | ndarray
        Derivative at x
    """
    r = np.zeros(max_iters)
    r[0] = finite_difference(function, x, h_init)
    for i in range(1, max_iters):
        h_init /= d
        r[i] = finite_difference(function, x, h_init)
    N_p = 1
    error = np.inf
    for i in range(1, max_iters):
        N_p *= d**2
        r[:max_iters - i] = (N_p * r[1:max_iters - i + 1] - r[:max_iters - i]) / (N_p - 1)
        new_error =  np.abs(r[0]-r[1])
        if new_error > error:
            return r[0]
        else:
            error = new_error
        if np.abs(new_error/r[0]) < eps:
            return r[0]


def main():

    # Values from the hand-in
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    bounds = (0, 5)
    xmin, xmax = 10**-4, 5
    N_generate = 10000
    xx = np.linspace(xmin, xmax, N_generate)

    integrand = lambda x, a, b, c: 4 * np.pi * x**2 * n(x, 1, Nsat, a, b, c)
    integral, err = romberg_integrator(
        integrand, bounds, order=10, args=(a, b, c), err=True
    )

    # Normalisation
    A = Nsat / integral
    err_A = Nsat / (integral**2) * err
    integrand = lambda x, a, b, c: 4 * np.pi * x**2 * n(x, A, Nsat, a, b, c)
    integrated_Nsat = romberg_integrator(
        integrand, bounds, order=10, args=(a, b, c), err=False
    )
    with open("Calculations/satellite_A.txt", "w") as f:
        f.write(f"{A:.12g} & {err_A:.3e} & {np.abs(integrated_Nsat - Nsat):.3e}\n")

    p_of_x = (
        lambda x: 4*np.pi*x**2*n(x, A, 1, a, b, c))

    p_max = np.max(p_of_x(xx))
    p_of_x_norm = lambda x: p_of_x(x) / p_max
    
    global seed
    seed = 31415926535
    random_samples = sampler(p_of_x_norm, min=xmin, max=xmax, Nsamples=N_generate, args=())

    edges = 10 ** np.linspace(np.log10(xmin), np.log10(xmax), 21)

    hist = np.histogram(random_samples, bins=edges)[0]
    hist_scaled = hist / (edges[1:]-edges[:-1])

    fig = plt.figure()
    relative_radius = np.linspace(xmin,xmax, 1000)
    analytical_function = 4*np.pi*relative_radius**2*n(relative_radius, A, 1, a, b, c) * N_generate

    fig1b, ax = plt.subplots()
    ax.stairs(
        hist_scaled, edges=edges, fill=True, label="Satellite galaxies"
    )
    plt.plot(
        relative_radius, analytical_function, "r-", label="Analytical solution"
    )
    ax.set(
        xlim=(xmin, xmax),
        ylim=(10 ** (-3), 1e5),
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.savefig("Plots/my_solution_1b.png", dpi=600)

    # Cumulative plot of the chosen galaxies (1c)
    chosen = sort_array(choice(random_samples, 100), inplace=True)
    fig1c, ax = plt.subplots()
    ax.plot(chosen, np.arange(100))
    ax.set(
        xscale="log",
        xlabel="Relative radius",
        ylabel="Cumulative number of galaxies",
        xlim=(xmin, xmax),
        ylim=(0, 100),
    )
    plt.savefig("Plots/my_solution_1c.png", dpi=600)

    x_to_eval = 1
    func_to_eval = lambda x: n(x, A, Nsat, a, b, c)
    dn_dx_numeric = compute_derivative(func_to_eval, x_to_eval, h_init=0.5, d=1.2, eps=1e-12, max_iters=12)
    dn_dx_analytic = dn_dx(x_to_eval, A, Nsat, a, b, c)
    with open("Calculations/satellite_deriv_analytic.txt", "w") as f:
        f.write(f"{dn_dx_analytic:.12g}\n")

    with open("Calculations/satellite_deriv_numeric.txt", "w") as f:
        f.write(f"{dn_dx_numeric:.12g}\n")


if __name__ == "__main__":
    main()

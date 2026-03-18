# imports
import numpy as np
import matplotlib.pyplot as plt


def n(x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> np.ndarray:
    """
    Number density profile of satellite galaxies

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
    ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    # Create x and n arrays
    x = np.asarray(x)
    n = np.zeros_like(x)

    # Function is only defined for positive x
    n[x>0] = A * Nsat * (x[x>0] / b) ** (a-3) * np.exp(-(x[x>0] / b) ** c)

    return n


##### Integrator block #####


def romberg_integrator(
    func: callable, bounds: tuple, order: int = 5, err: bool = False, args: tuple = ()
) -> float:
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

    # Returns error if needed
    if err:
        return r[0], np.abs(r[0]-r[1])  # (value, error)
    return r[0] # value


#### Sampler block ####


def rng(N: int) -> np.ndarray:
    """
    Random number generator 

    Parameters
    ----------
    N: int
        Number of random numbers

    Returns
    -------
    np.ndarray
        Array containing N random numbers
        if N=1 returns float instead
    """
    global seed
    seed = np.uint(seed)

    # Parameters for rng
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

        # Calculating the float
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
    args: tuple = (),
) -> np.ndarray:
    """
    Sample a distribution using rejection sampling

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
        # Generate random x between min and max
        random_x = min + (max-min)*rng(1)

        # Generate random y between 0 and 1
        random_y = rng(1)

        # Store sample if random y is smaller than distribution
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
    Sort a 1D array using merge sort

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
    def roll(array, shift):
        """
        rolls array by shift indices
        """
        shifted_array = np.empty_like(array)
        shifted_array[:shift] = array[-shift:]
        shifted_array[shift:] = array[:-shift]
        return shifted_array
    
    # Make an copy if not inplace
    if inplace:
        sorted_arr = arr
    else:
        sorted_arr = arr.copy()

    N = len(sorted_arr)
    step = 1
    index_array = np.arange(N)

    while step < N:
        # Loop over pairs of subarrays
        for i in range(0, N, step*2):
            # Set pointers for 2 adjacent subarrays
            l = i
            r = min(i + step, N)
            end = min(i + 2*step, N)

            # Loop over the elements of both subarrays
            while l < r and r < end:
                # If the left <=  the right, the left is in the correct place
                if sorted_arr[l] <= sorted_arr[r]:
                    # increase left pointer
                    l += 1

                # If the left > the right, the array is rolled to the correct order
                else:
                    sorted_arr[l:r+1] = roll(sorted_arr[l:r+1], 1)
                    index_array[l:r+1] = roll(index_array[l:r+1], 1)
                    # Increase both pointers
                    l += 1
                    r += 1
        # Double step size
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
    # Generate array of float the same size as arr
    random = rng(len(arr))

    # Sort random array to obtain an index array
    random_index = sort_array(random, index=True)

    # Return the first size elements of the shuffled array
    return arr[random_index][:size].copy()


##### Derivative block #####


def dn_dx(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
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
    ndarray
        Same type and shape as x. Derivative of number density of
        satellite galaxies at given radius x.
    """
    fraction = x/b

    return  - A * Nsat * fraction**(a-4) * (c * fraction ** c - a + 3) * np.exp(-fraction**c) / b


def finite_difference(function: callable, x: np.ndarray, h: float) -> np.ndarray:
    """
    A building block to compute derivative using finite differences

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : ndarray
        Value(s) to evaluate derivative at
    h : float
        Step size for finite difference

    Returns
    -------
    dy : ndarray
        Derivative at x
    """
    # Return the central difference
    return 0.5 * (function(x + h) - function(x - h)) / h


def compute_derivative(
    function: callable,
    x: np.ndarray,
    h_init: float,
    # For Ridders use parameters below:
    d: float, # Factor by which to decrease h_init every iteration
    eps: float, # Relative error
    max_iters: int = 10, #3 Maximum number of iterations before exiting
) -> np.ndarray:
    """
    Function to compute derivative

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : ndarray
        Value(s) to evaluate derivative at
    h_init : float
        Initial step size for finite difference

    Returns
    -------
    df : ndarray
        Derivative at x
    """
    # Create starting array with initial esimates
    d_inverse = 1/d
    h = np.array([h_init * d_inverse ** n for n in range(0, max_iters)])
    r = finite_difference(function, x, h)

    N_p = 1
    error = np.inf
    # Iteratively improve the estimate using the previous ones
    for i in range(1, max_iters):
        N_p *= d**2

        # Calculate new estimates
        r[:max_iters - i] = (N_p * r[1:max_iters - i + 1] - r[:max_iters - i]) / (N_p - 1)

        new_error =  np.abs(r[0]-r[1])

        # If the error increases stop
        if new_error > error:
            break
        else:
            error = new_error

        # If the relative error is smaller than the desired error stop
        if np.abs(new_error/r[0]) < eps:
            break

    return r[0]


def main():
    #### 1a ####
    # Values from the hand-in
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    bounds = (0, 5)
    xmin, xmax = 10**-4, 5
    N_generate = 10000
    xx = np.linspace(xmin, xmax, N_generate)

    # Defining integrand with A=1
    integrand = lambda x, a, b, c: 4 * np.pi * x**2 * n(x, 1, Nsat, a, b, c)
    integral, err = romberg_integrator(
        integrand, bounds, order=10, args=(a, b, c), err=True
    )

    # Normalisation
    A = Nsat / integral
    err_A = Nsat / (integral**2) * err

    # Plug in value of A to check whether the value is correct
    integrand = lambda x, a, b, c: 4 * np.pi * x**2 * n(x, A, Nsat, a, b, c)
    integrated_Nsat = romberg_integrator(
        integrand, bounds, order=10, args=(a, b, c), err=False
    )
    with open("Calculations/satellite_A.txt", "w") as f:
        f.write(f"{A:.12g} & {err_A:.3e} & {np.abs(integrated_Nsat - Nsat):.3e}\n")

    #### 1b ####
    # Defining probability density
    p_of_x = (
        lambda x: 4*np.pi*x**2*n(x, A, 1, a, b, c))

    # Normalize probability density for rejection sampling
    p_max = np.max(p_of_x(xx))
    p_of_x_norm = lambda x: p_of_x(x) / p_max
    
    # Set seed
    global seed
    seed = 31415926535

    # Sample from distribution
    random_samples = sampler(p_of_x_norm, min=xmin, max=xmax, Nsamples=N_generate, args=())

    edges = 10 ** np.linspace(np.log10(xmin), np.log10(xmax), 21)

    hist = np.histogram(random_samples, bins=edges)[0]

    # Scale histogram by number of sources and bin width
    hist_scaled = hist / (N_generate*(edges[1:]-edges[:-1]))

    fig = plt.figure()
    relative_radius = np.linspace(xmin,xmax, 1000)
    analytical_function = 4*np.pi*relative_radius**2*n(relative_radius, A, 1, a, b, c)

    fig1b, ax = plt.subplots()
    ax.stairs(
        hist_scaled, edges=edges, fill=True, label="Satellite galaxies"
    )
    plt.plot(
        relative_radius, analytical_function, "r-", label="Analytical solution"
    )
    ax.set(
        xlim=(xmin, xmax),
        ylim=(10 ** (-3), 10),
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.savefig("Plots/my_solution_1b.png", dpi=600)

    #### 1c ####
    # Sort a selection of the sample to get cumaltive distribution
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

    #### 1d ####
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

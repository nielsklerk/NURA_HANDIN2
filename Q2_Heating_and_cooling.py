import numpy as np
import timeit

# Constants (mind the units!)

psi = 0.929
Tc = 1e4 # K
Z = 0.015
k = 1.38e-16  # erg/K
aB = 2e-13  # cm^3 / s
A = 5e-10
xi = 1e-15


# There's no need for nH nor ne as they cancel out
def equilibrium1(T, Z, Tc, psi):
    return psi * Tc * k - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T * k


def equilibrium2(T, Z, Tc, psi, nH, A, xi, aB):
    return (
        (
            psi * Tc
            - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T
            - 0.54 * (T / 1e4) ** 0.37 * T
        )
        * k
        * nH
        * aB
        + A * xi
        + 8.9e-26 * (T / 1e4)
    )


#### root finder ####


def root_finder(
    func: callable,  # add derivative if using Newton-Raphson
    bracket: tuple,
    atol: float = 1e-6,
    rtol: float = 1e-10,
    max_iters: int = 100,
    steps_method_swap: int = 35,
    return_num_step: bool = False
) -> tuple:
    """
    Find a root of a function

    Parameters
    ----------
    func : callable
        Function to find root of
    bracket : tuple
        Bracket for which to find first secant
    atol : float, optional
        Absolute tolerance.
        The default is 1e-6
    rtol : float, optional
        Relative tolerance.
        The default is 1e-6
    max_iters: int, optional
        Maximum number of iterations.
        The default is 100
    steps_method_swap: int, optional
        Number of steps after which the root finder
        switches from false position to secant method
        The defaul is 35
    return_num_step: bool, optional
        If True returns the number of 
        The defaul is False
    

    Returns
    -------
    root : float
        Approximate root
    aerr : float
        Absolute error
    rerr : float
        Relative error
    num_step: float
        Number of steps to find root (only returned when return_num_step is True)
    """
    # Extract bracket
    a, b = bracket

    # Set value for previous c
    previous_c = np.inf

    # Turning on false position
    false_position= not steps_method_swap<=0

    for num_step in range(max_iters):

        # Calculate the next point
        c = b - (b-a)/(func(b)-func(a))*func(b)

        # False position
        if false_position:
            # Choosing what point to replace
            if func(a)*func(c) < 0:
                b = np.copy(c)
            else:
                a = np.copy(c)

        # Secant method
        else:
            a = np.copy(b)
            b = np.copy(c)

        # Absolute and relative error
        aerr = np.abs(c - previous_c)
        rerr = np.abs(aerr /c)

        # Check if absolute error is below tolerance
        if aerr < atol:
            if return_num_step:
                return c, aerr, rerr, num_step + 1.0
            return c, aerr, rerr
        
        # Check if relative error is below tolerance
        if rerr < rtol:
            if return_num_step:
                return c, aerr, rerr, num_step + 1.0
            return c, aerr, rerr
        
        # Switching method after steps_method_swap 
        if num_step == steps_method_swap:
            false_position=False
        previous_c = np.copy(c)
    
    # Return after max_iters
    if return_num_step:
        return c, aerr, rerr, num_step + 1.0
    return c, aerr, rerr


def main():
    number = 10
    # Initial bracket
    bracket = (1, 1e7)

    time = (
            timeit.timeit(
                stmt=lambda: root_finder(lambda x: equilibrium1(x, Z, Tc, psi), bracket),
                number=number,
            )
            / number
        )
    
    # Finding root equilibrium 1
    root, aerr, rerr, steps = root_finder(lambda x: equilibrium1(x, Z, Tc, psi), bracket, steps_method_swap=1, return_num_step=True)  # replace with your root finder

    with open("Calculations/equilibrium_temp_simple.txt", "w") as f:
        f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e} & {int(steps)} & {time:.3e}")
    #### 2b ####

    # Initial bracket
    bracket = (1, 1e15)

    for nH in [1e-4, 1, 1e4]:
        
        time = (
            timeit.timeit(
                stmt=lambda: root_finder(lambda x: equilibrium2(x, Z, Tc, psi, nH, A, xi, aB), bracket),
                number=number,
            )
            / number
        )
        # Finding root equilibrium 1=1
        root, aerr, rerr, steps = root_finder(lambda x: equilibrium2(x, Z, Tc, psi, nH, A, xi, aB), bracket, steps_method_swap=31, return_num_step=True)
        if nH == 1e-4:
            with open("Calculations/equilibrium_low_density.txt", "w") as f:
                f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e} & {int(steps)} & {time:.3e}")
        elif nH == 1:
            with open("Calculations/equilibrium_mid_density.txt", "w") as f:
                f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e} & {int(steps)} & {time:.3e}")
        elif nH == 1e4:
            with open("Calculations/equilibrium_high_density.txt", "w") as f:
                f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e} & {int(steps)} & {time:.3e}")


if __name__ == "__main__":
    main()

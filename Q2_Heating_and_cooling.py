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
    max_iters: int = 1000,
    return_num_step: bool = False
) -> tuple[float, float, float] | tuple[float, float, float, float]:
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
    return_num_step: bool, optional
        If True returns the number of 
    

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
    # # False Position
    a, b = bracket  
    previous_c = np.inf
    false_position=True
    for num_step in range(max_iters):
        c = b - (b-a)/(func(b)-func(a))*func(b)
        if false_position:
            if func(a)*func(c) < 0:
                b = np.copy(c)
            else:
                a = np.copy(c)
        else:
            a = np.copy(b)
            b = np.copy(c)
        aerr = np.abs(c - previous_c)
        if num_step == 35:
            false_position=False
        if aerr < atol:
            if return_num_step:
                return c, aerr, rerr, num_step + 1.0
            return c, aerr, rerr
        rerr = np.abs(aerr /c)
        if rerr < rtol:
            if return_num_step:
                return c, aerr, rerr, num_step + 1.0
            return c, aerr, rerr
        previous_c = np.copy(c)
    if return_num_step:
        return c, aerr, rerr, num_step + 1.0
    return c, aerr, rerr


def main():

    # Initial bracket
    bracket = (1, 1e7)

    root, aerr, rerr = root_finder(lambda x: equilibrium1(x, Z, Tc, psi), bracket)  # replace with your root finder

    with open("Calculations/equilibrium_temp_simple.txt", "w") as f:
        f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e}")
    #### 2b ####

    # Initial bracket
    bracket = (1, 1e15)

    for nH in [1e-4, 1, 1e4]:
        number = 10
        t = (
            timeit.timeit(
                stmt=lambda: root_finder(lambda x: equilibrium2(x, Z, Tc, psi, nH, A, xi, aB), bracket),
                number=number,
            )
            / number
        )

        root, aerr, rerr, steps = root_finder(lambda x: equilibrium2(x, Z, Tc, psi, nH, A, xi, aB), bracket, return_num_step=True)
        if nH == 1e-4:
            with open("Calculations/equilibrium_low_density.txt", "w") as f:
                f.write(f"{root:.12g} & {equilibrium2(root, Z, Tc, psi, nH, A, xi, aB):.3e} & {int(steps)} & {t:.3e}")
        elif nH == 1:
            with open("Calculations/equilibrium_mid_density.txt", "w") as f:
                f.write(f"{root:.12g} & {equilibrium2(root, Z, Tc, psi, nH, A, xi, aB):.3e} & {int(steps)} & {t:.3e}")
        elif nH == 1e4:
            with open("Calculations/equilibrium_high_density.txt", "w") as f:
                f.write(f"{root:.12g} & {equilibrium2(root, Z, Tc, psi, nH, A, xi, aB):.3e} & {int(steps)} & {t:.3e}")


if __name__ == "__main__":
    main()

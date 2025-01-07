import numpy as np

def newton_raphson_system(N, init_guess='uniform', dv_dx_at_firstPoint='FD2', tol=1e-6, max_iter=1000):
    """
    Solve the nonlinear system using Newton-Raphson method.

    Parameters:
        N: int - Number of grid spaces.
        tol: float - Tolerance for convergence.
        max_iter: int - Maximum number of iterations.

    Returns:
        x: array - Discretized x values.
        v: array - Solution v(x).
    """

    # Validate inputs
    valid_init_guesses = ['random', 'uniform', 'zeros']
    valid_dv_dx_methods = ['FD2', 'FD1', 'Direct']

    if init_guess not in valid_init_guesses:
        raise ValueError(f"Invalid initial guess '{init_guess}'. Must be one of {valid_init_guesses}.")
    if dv_dx_at_firstPoint not in valid_dv_dx_methods:
        raise ValueError(f"Invalid dv/dx method '{dv_dx_at_firstPoint}'. Must be one of {valid_dv_dx_methods}.")

    # Discretization
    L = 1.0
    dx = L / N
    x = np.linspace(0, L, N+1)
    EI =  2000       # Flexural rigidity (N·m^2)
    P  = -1000.0     # Point load at the free end (N)
    w  =  0.5        # Relaxation factor

    # Initial guess
    if init_guess == 'random':
        v = np.random.uniform(0, 0.01 * np.sign(P), N+1)       # Worst among all three. Cons: 1. Chances of 'no convergence' within 'max_iter' due to the randomness in the initial value.
        v[0] = 0                                               #                              2. Different Final Results in each run for large N (> 250).
                                                               #                              3. Jacobian becomes singular for N > 350.
    elif init_guess == 'uniform':
        v = 0.0001 * np.sign(P) * x[:]                         # Better than both. Cons: Jacobian becomes singular for N > 372.
        v[0] = 0
    elif init_guess == 'zeros':
        v = np.zeros(N+1)                                      # Good. Achieves Middle Ground. Cons: Jacobian becomes singular for N > 357.
        v[0] = 0
    print()
    print(v)

    # Moment Equation
    M = np.zeros(N+1)
    for i in range(N+1):
        M[i] = -P*(L - x[i])

    # Residual Function
    def residual(v, M):
        """Compute the residual vector R(v)."""
        R = np.zeros(N+1)
        v_full = np.zeros(N+1)
        v_full[:] = v
        v_full[0] = 0           # Boundary condition v(0) = 0

        for i in range(0, N+1):
            if i == 0:
                # Boundary condition dv/dx[x=0] = 0
                if dv_dx_at_firstPoint == 'FD2':
                    R[i] = (-3*v_full[i]+4*v_full[i+1]-v_full[i+2])/(2*dx)
                elif dv_dx_at_firstPoint == 'FD1':
                    R[i] = (v_full[i+1]-v_full[i])/dx
                elif dv_dx_at_firstPoint == 'Direct':
                    R[i] = (2*v_full[i]-5*v_full[i+1]+4*v_full[i+2]-v_full[i+3])/dx**2 + (M[i]/EI)

            elif i == N:
                R[i] = (2*v_full[i]-5*v_full[i-1]+4*v_full[i-2]-v_full[i-3])/dx**2 + (M[i]/EI) * (1 + 1/(4*dx**2) * (3*v_full[i]-4*v_full[i-1]+v_full[i-2])**2)**(3/2)
            else:
                R[i] = (v_full[i+1]-2*v_full[i]+v_full[i-1])/dx**2 + (M[i]/EI) * (1 + 1/(4*dx**2) * (v_full[i+1]-v_full[i-1])**2)**(3/2)
        return R

    # Jacobian Function
    def jacobian(v, M):
        """Compute the Jacobian matrix J(v)."""
        J = np.zeros((N+1, N+1))
        v_full = np.zeros(N+1)
        v_full[:] = v
        v_full[0] = 0           # Boundary condition v(0) = 0

        for i in range(0, N+1):
            if i == 0:
                # Boundary condition dv/dx[x=0] = 0
                if dv_dx_at_firstPoint == 'FD2':
                    J[i, i], J[i, i+1], J[i, i+2] =  0, 4/(2*dx), -1/(2*dx)
                elif dv_dx_at_firstPoint == 'FD1':
                    J[i, i], J[i, i+1] = 0, 1/dx
                elif dv_dx_at_firstPoint == 'Direct':
                    J[i, i], J[i, i+1], J[i, i+2], J[i, i+3] =  0, -5/dx**2, 4/dx**2, -1/dx**2

            elif i == N:
                J[i, i-3] = -1/dx**2
                J[i, i-2] =  4/dx**2 + (M[i]/EI) * (3/2) * (1 + 1/(4*dx**2) * (3*v_full[i]-4*v_full[i-1]+v_full[i-2])**2)**(1/2) * 1/(4*dx**2) * 2 *(3*v_full[i]-4*v_full[i-1]+v_full[i-2]) *  (1)
                J[i, i-1] = -5/dx**2 + (M[i]/EI) * (3/2) * (1 + 1/(4*dx**2) * (3*v_full[i]-4*v_full[i-1]+v_full[i-2])**2)**(1/2) * 1/(4*dx**2) * 2 *(3*v_full[i]-4*v_full[i-1]+v_full[i-2]) * (-4)
                J[i, i]   =  2/dx**2 + (M[i]/EI) * (3/2) * (1 + 1/(4*dx**2) * (3*v_full[i]-4*v_full[i-1]+v_full[i-2])**2)**(1/2) * 1/(4*dx**2) * 2 *(3*v_full[i]-4*v_full[i-1]+v_full[i-2]) *  (3)
            else:
                if i == 1:
                    J[i, i-1] = 0
                else:
                    J[i, i-1] =  1/dx**2 + (M[i]/EI) * (3/2) * (1 + 1/(4*dx**2) * (v_full[i+1]-v_full[i-1])**2)**(1/2) * 1/(4*dx**2) * (2*v_full[i-1]-2*v_full[i+1])
                J[i, i]   = -2/dx**2
                J[i, i+1] =  1/dx**2 + (M[i]/EI) * (3/2) * (1 + 1/(4*dx**2) * (v_full[i+1]-v_full[i-1])**2)**(1/2) * 1/(4*dx**2) * (2*v_full[i+1]-2*v_full[i-1])
        return J

    # Newton-Raphson iteration
    for iteration in range(max_iter):
        R = residual(v, M)
        print(np.linalg.norm(R))
        if np.linalg.norm(R) < tol:
            print(f"\nConverged in {iteration+1} iterations.\n")
            break

        J = jacobian(v, M)
        try:
            # Attempt to solve the system
            delta_v = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            print(f"\nJacobian is singular for iteration {iteration+1}. Using pseudo-inverse.")
            delta_v = np.linalg.lstsq(J, -R, rcond=None)[0]    # Uses pseudo-inverse as the inverse does not exist

        v += w*delta_v

        if np.linalg.norm(delta_v) < tol:
            print(f"\nConverged in {iteration+1} iterations.\n")
            break
    else:
        raise ValueError(f"Newton-Raphson did not converge within the maximum number of iterations = {iteration+1}.")

    # Full solution including boundaries
    v_full = np.zeros(N+1)
    v_full[:] = v
    v_full[0] = 0

    return x, v_full, L, P, EI


# Solve and visualize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cufflinks as cf
import pandas as pd
'''
    Minimum value of 'N' can be 3.
    Don't take N > 350.  >>> Leads to Jacobian Matrix being singular for all three kinds of initial guess (random, uniform, zeros).
'''
N = 250
init_guess = input("Enter initial guess type (random, uniform, zeros): ")
dv_dx_at_firstPoint = input("Enter dv/dx at x = 0 (FD2, FD1, Direct): ")
x, v, L, P, EI = newton_raphson_system(N, init_guess, dv_dx_at_firstPoint)
print(v)
print()

# plt.plot(x, v, label="Newton-Raphson Solution")
# plt.plot(x, P*x**2/(6*EI) * (3*L - x), label="Euler-Bernoulli Solution")
# plt.xlabel("x")
# plt.ylabel("v(x)")
# plt.legend()
# plt.grid()
# plt.show()

# Create Pandas DataFrame
df = pd.DataFrame({'x': x, 'Newton-Raphson Solution': v, 'Euler-Bernoulli Solution': P*x**2/(6*EI) * (3*L - x)})

# Create interactive plot using Plotly
fig = df.iplot(kind='line', x='x', y=['Newton-Raphson Solution', 'Euler-Bernoulli Solution'], asFigure=True)
fig.show()

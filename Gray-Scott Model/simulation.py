import numpy as np

# Simulation parameters
width = 100.0      # Width in nm
height = 100.0     # Height in nm
grid_size = 100     # Number of grid points along each axis
dx = width / grid_size  # Spatial step in nm
dt = 0.5            # Time step in ns

# Parameters for the Gray-Scott model with appropriate units
f = 0.0028  # Feed rate for U (1/ns)
k = 0.05  # Kill rate for V (1/ns)
diff_U = 0.06  # Diffusion coefficient for U (nm^2/ns)
diff_V = 0.03  # Diffusion coefficient for V (nm^2/ns)


initial_U = 1*np.ones((grid_size, grid_size))  # Start with U initialized to 1
initial_V = 0*np.ones((grid_size, grid_size))  # Start with V initialized to 0

# Add a circular region of V=1 in the center
center_x, center_y = grid_size // 2, grid_size // 2
radius = 3
for i in range(grid_size):
    for j in range(grid_size):
        if (i - center_x)**2 + (j - center_y)**2 < radius**2:
            initial_U[i, j] = 0.5
            initial_V[i, j] = 0.25
            

# Laplacian function to compute diffusion
def laplacian(Z, dx):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z) / (dx**2)

# Update function based on the Gray-Scott model
def update(U, V, f, k, diff_U, diff_V, dt, dx):
    laplacian_U = laplacian(U, dx)
    laplacian_V = laplacian(V, dx)

    # Reaction terms for the Gray-Scott model
    reaction_U = -U * V**2 + f * (1 - U)
    reaction_V = U * V**2 - (f + k) * V

    # Apply reaction and diffusion
    U += (reaction_U + diff_U * laplacian_U) * dt
    V += (reaction_V + diff_V * laplacian_V) * dt

    # Ensure non-negative concentrations
    U[U < 0] = 0
    V[V < 0] = 0
    U[U > 1] = 1
    V[V > 1] = 1


    return U, V

# Initialize U and V arrays
def initialize():
    U = initial_U.copy()
    V = initial_V.copy()
    return U, V

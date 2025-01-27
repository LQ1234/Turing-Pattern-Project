import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# Parameters
nx, ny = 10, 10             # Grid size (reduced as per your request)
Q = 5                       # Number of layers
dx = dy = 1.0               # Space steps
dt = 0.01                   # Time step
steps_per_frame = 5         # Simulation steps per animation frame

# Physical parameters (adjustable via sliders)
K_init = 1.0
beta_init = 1.0
gamma_init = 1.0
alpha_init = 3.0            # Now adjustable via slider

# Initialize c_i(x, y, t)
def initialize_layers():
    c_init = np.zeros((Q, nx, ny))
    # c_init = np.random.rand(Q, nx, ny)  # Random initialization
    c_init[0] = 1.0  # Bottom layer is filled

    #c_init[:,0,0] = 1.0
    c_init += np.random.rand(Q, nx, ny) * 0.05
    

    return c_init

c = initialize_layers()

# Corrected N_alpha(c)
def N_alpha(c, alpha):
    return c**2 * (1 + 2 * (1 - c) + alpha * (1 - c)**2)

# Derivative of N_alpha with respect to c
def dN_alpha_dc(c, alpha):
    return c * (6 + 2 * alpha) + c ** 2 * (-6 - 6 * alpha) + c ** 3 * (4 * alpha)

# Function to compute Laplacian using finite differences with periodic boundary conditions
def laplacian(u):
    u_padded = np.pad(u, ((0, 0), (1, 1), (1, 1)), mode='wrap')
    laplacian_u = (
        u_padded[:, 0:-2, 1:-1] + u_padded[:, 2:, 1:-1] +
        u_padded[:, 1:-1, 0:-2] + u_padded[:, 1:-1, 2:] -
        4 * u
    ) / (dx * dy)
    return laplacian_u

# Function to update the layers c_i
def update_layers(c, K, beta, gamma, alpha):
    lap_c = laplacian(c)
    dc_dt = np.zeros_like(c)
    N_c = N_alpha(c, alpha)
    N_1_minus_c = N_alpha(1 - c, alpha)
    dN_dc = dN_alpha_dc(c, alpha)
    dN_1_minus_dc = -dN_alpha_dc(1 - c, alpha)  # Chain rule for derivative of N_alpha(1 - c)

    print("-----")
    for i in range(Q):
        # print(f"Layer {i} = {np.mean(c[i]):0.3f}:", end=" ")
        # # Compute the derivative of the energy functional with respect to c_i
        # term1 = - K * lap_c[i]
        # term2 = 0#beta * ( 1 -  c[i])
        # print(f"Term 1 = {np.mean(term1):0.3f}, Term 2 = {np.mean(term2):0.3f}", end=" ")

        # # Compute the cumulative product up to i-1
        # cumulative_product = np.ones((nx, ny))
        # if i > 0:
        #     for k in range(i):
        #         cumulative_product *= N_c[k]

        # # First part involving dN_1_minus_dc[i] 
        # term3 = cumulative_product * dN_1_minus_dc[i]
        # print(f"Term 3 = {np.mean(term3):0.3f}", end=" ")

        # print(f"Term 4 = ", end=" ")

        # # Second part involving the sum over j > i
        # term4 = np.zeros((nx, ny))
        # for j in range(i + 1, Q):
        #     # Compute the cumulative product up to j-1 excluding c_i
        #     cumulative_product_j = np.ones((nx, ny))
        #     for k in range(j):
        #         if k != i:
        #             cumulative_product_j *= N_c[k]
        #     cumulative_product_j *= N_1_minus_c[j]
        #     # Compute the factor for c_i
        #     factor = cumulative_product_j * dN_dc[i]
        #     term4 += factor
        #     print(f"({j}) = {np.mean(factor):0.3f}", end=" ")
        # term4 = 0
        # print(f"Term 4 = {np.mean(term4):0.3f}", end=" ")

        # # Total derivative
        # dF_dci = term1 + term2 + gamma * (term3 + term4)
        # print(f"dF_dci = {np.mean(dF_dci):0.3f}")

        # # Update c_i
        # dc_dt[i] = - dF_dci

        g = lambda x: 1 / (1 + np.exp(-30 * (x - 0.85)))
        dc_dt[i] = K * lap_c[i]
        if i > 0:
            dc_dt[i] += beta * g(c[i-1]) * (1 - c[i] )
            dc_dt[i] -= gamma *  g(c[i-1]) * c[i] * (1 - c[i]) * (1 - 2 * c[i])


    # Update c with time step
    c_new = c + dc_dt * dt
    # Ensure c remains within [0, 1]
    c_new = np.clip(c_new, 0, 1)
    return c_new

# Visualization setup

# Create subplots: one for N_alpha plot and one for the 3D voxel plot
fig = plt.figure(figsize=(8, 10))  # Decreased figure size to make the plot smaller

# Create a grid specification for the plots
gs = fig.add_gridspec(4, 1, height_ratios=[1, 0.05, 0.5, 2], hspace=0.4)

# Subplot for N_alpha and its derivative
ax_N = fig.add_subplot(gs[0, 0])

# Subplot for the sliders
# We will position the sliders more carefully to ensure they are visible

# Subplot for the 3D voxel plot
ax_voxel = fig.add_subplot(gs[3, 0], projection='3d')

# Time text
time_template = 'Time = %.2f'
time_text = ax_voxel.text2D(0.05, 0.95, '', transform=ax_voxel.transAxes)

# Initialize plot
epsilon = 0.01  # Small value to include in voxels

# Prepare the N_alpha plot data
c_values = np.linspace(0, 1, 200)
N_values = N_alpha(c_values, alpha_init)
dN_values = dN_alpha_dc(c_values, alpha_init)

# Initial plot of N_alpha and its derivative
line_N, = ax_N.plot(c_values, N_values, label='$N_{\\alpha}(c)$')
line_dN, = ax_N.plot(c_values, dN_values, label='$\\frac{dN_{\\alpha}}{dc}$')
ax_N.set_xlabel('$c$')
ax_N.set_ylabel('Value')
ax_N.legend()
ax_N.set_title('Function $N_{\\alpha}(c)$ and its derivative')

# Adjusting the positions of sliders to ensure visibility
slider_height = 0.03
slider_width = 0.7
slider_left = 0.15
slider_bottom_start = 0.55  # Starting position from the bottom

axcolor = 'lightgoldenrodyellow'
ax_K = plt.axes([slider_left, slider_bottom_start + 3 * (slider_height + 0.01), slider_width, slider_height], facecolor=axcolor)
ax_beta = plt.axes([slider_left, slider_bottom_start + 2 * (slider_height + 0.01), slider_width, slider_height], facecolor=axcolor)
ax_gamma = plt.axes([slider_left, slider_bottom_start + 1 * (slider_height + 0.01), slider_width, slider_height], facecolor=axcolor)
ax_alpha = plt.axes([slider_left, slider_bottom_start + 0 * (slider_height + 0.01), slider_width, slider_height], facecolor=axcolor)

slider_K = Slider(ax_K, 'K', 0.0, 10.0, valinit=K_init)
slider_beta = Slider(ax_beta, 'Beta', 0.0, 10.0, valinit=beta_init)
slider_gamma = Slider(ax_gamma, 'Gamma', 0.0, 10.0, valinit=gamma_init)
slider_alpha = Slider(ax_alpha, 'Alpha', 0.0, 10.0, valinit=alpha_init)  # Alpha >= 3

# Reset button
resetax = plt.axes([0.8, slider_bottom_start - 0.05, 0.1, 0.04])
button_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def init_plot():
    # Initialize N_alpha plot (already done above)
    # Initialize voxel plot
    ax_voxel.clear()
    ax_voxel.set_xlim(0, nx)
    ax_voxel.set_ylim(0, ny)
    ax_voxel.set_zlim(0, Q)
    ax_voxel.set_xlabel('X')
    ax_voxel.set_ylabel('Y')
    ax_voxel.set_zlabel('Layer')
    ax_voxel.set_title('Atom Deposition Simulation')
    return []

# Update function for animation
simulation_time = 0.0

def animate(frame):
    global c, simulation_time
    K = slider_K.val
    beta = slider_beta.val
    gamma = slider_gamma.val
    alpha = slider_alpha.val  # Get alpha from the slider

    for _ in range(steps_per_frame):
        c = update_layers(c, K, beta, gamma, alpha)
        c = c.clip(0, 1)  # Avoid division by zero
        simulation_time += dt

    # Update N_alpha plot
    N_values = N_alpha(c_values, alpha)
    dN_values = dN_alpha_dc(c_values, alpha)
    line_N.set_ydata(N_values)
    line_dN.set_ydata(dN_values)
    ax_N.relim()
    ax_N.autoscale_view()

    # Prepare data for 3D voxel plot
    voxels = np.zeros((nx, ny, Q), dtype=bool)
    # Initialize colors array with an extra dimension for RGBA values
    colors = np.zeros(voxels.shape + (4,), dtype=float)

    for i in range(Q):
        # Include voxels where c_i > epsilon
        layer = c[i] > epsilon
        voxels[:, :, i] = layer

        # Get RGB color from colormap based on layer number
        rgb = plt.cm.viridis(i / Q)[:3]  # Extract RGB values

        # Get alpha values from c_i
        alpha_values_voxel = c[i]
        alpha_values_voxel = np.clip(alpha_values_voxel, 0, 1)  # Ensure alpha is within [0, 1]

        # Assign colors with alpha channel
        colors[:, :, i, 0] = rgb[0]  # Red channel
        colors[:, :, i, 1] = rgb[1]  # Green channel
        colors[:, :, i, 2] = rgb[2]  # Blue channel
        colors[:, :, i, 3] = alpha_values_voxel  # Alpha channel

        # Set alpha to zero where layer is False
        colors[:, :, i, 3][~layer] = 0

    ax_voxel.clear()
    ax_voxel.voxels(voxels, facecolors=colors, edgecolor='k')
    ax_voxel.set_xlim(0, nx)
    ax_voxel.set_ylim(0, ny)
    ax_voxel.set_zlim(0, Q)
    ax_voxel.set_xlabel('X')
    ax_voxel.set_ylabel('Y')
    ax_voxel.set_zlabel('Layer')
    ax_voxel.set_title('Atom Deposition Simulation')
    time_text = ax_voxel.text2D(0.05, 0.95, time_template % simulation_time, transform=ax_voxel.transAxes)
    plt.draw()
    return [line_N, line_dN]

# Reset function
def reset(event):
    global c, simulation_time
    c = initialize_layers()
    simulation_time = 0.0

button_reset.on_clicked(reset)

# Adjust layout to prevent overlap
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)

# Create the animation
ani = animation.FuncAnimation(fig, animate, init_func=init_plot, frames=200, interval=50, blit=False)

plt.show()

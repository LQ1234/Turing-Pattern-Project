import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# Parameters (adjustable via sliders)
nx, ny = 200, 200          # Grid size
Q = 10                      # Number of grains (order parameters)
dx = dy = 0.1              # Space steps
dt = 0.001                   # Time step
steps_per_frame = 10        # Simulation steps per animation frame

# Initialize coefficients (will be linked to sliders)
alpha_init = 1.0
beta_init = 1.0
gamma_init = 1.0
kappa_init = 1.0
L_init = 1.0

# Initialize order parameters h_q(r, t) with small random noise
def initialize_order_parameters():
    np.random.seed(None)  # Use a different seed each time
    return np.random.uniform(0.00, 0.001, (Q, nx, ny))

h = initialize_order_parameters()

# Function to compute Laplacian using finite differences with periodic boundary conditions
def laplacian(h_q):
    h_q_padded = np.pad(h_q, 1, mode='wrap')
    laplacian_h_q = (
        h_q_padded[0:-2, 1:-1] + h_q_padded[2:, 1:-1] +
        h_q_padded[1:-1, 0:-2] + h_q_padded[1:-1, 2:] -
        4 * h_q
    ) / (dx * dy)
    return laplacian_h_q

# Function to update the order parameters
def update_order_parameters(h, alpha, beta, gamma, kappa, L):
    h_sum_sq = np.sum(h ** 2, axis=0)
    lap_h = np.array([laplacian(h_q) for h_q in h])
    interaction = 2 * h * (h_sum_sq - h ** 2)
    print(alpha, beta, gamma, kappa, L)
    dhdt = -L[:, None, None] * (-alpha * h + beta * h ** 3 + gamma * interaction - kappa[:, None, None] * lap_h)
    h_new = h + dhdt * dt
    return h_new

# Prepare for animation
fig = plt.figure(figsize=(8, 8))

# Adjust the layout to have more space between image and sliders
# Main plot area
ax = plt.axes([0.1, 0.35, 0.8, 0.6])  # [left, bottom, width, height]
im = ax.imshow(np.argmax(h, axis=0), cmap='jet')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Grain Orientation (q)')

# Time display as text artist
time_text = ax.text(0.5, -0.1, '', transform=ax.transAxes, ha='center')

# Sliders and reset button
slider_alpha_ax = plt.axes([0.15, 0.22, 0.7, 0.03])
slider_beta_ax = plt.axes([0.15, 0.17, 0.7, 0.03])
slider_gamma_ax = plt.axes([0.15, 0.12, 0.7, 0.03])
slider_kappa_ax = plt.axes([0.15, 0.07, 0.7, 0.03])
slider_rate_ax = plt.axes([0.15, 0.02, 0.7, 0.03])

reset_ax = plt.axes([0.87, 0.02, 0.1, 0.08])

# Sliders
slider_alpha = Slider(slider_alpha_ax, 'Alpha', 0.0, 50.0, valinit=alpha_init)
slider_beta = Slider(slider_beta_ax, 'Beta', 0.0, 50.0, valinit=beta_init)
slider_gamma = Slider(slider_gamma_ax, 'Gamma', 0.0, 50.0, valinit=gamma_init)
slider_kappa = Slider(slider_kappa_ax, 'Kappa', 0.0, 50.0, valinit=kappa_init)
slider_rate = Slider(slider_rate_ax, 'Sim Rate', 1, 30, valinit=steps_per_frame, valfmt='%0.0f')

# Reset button
reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')

# Variables to hold time and parameters
simulation_time = 0.0

# Function to reset the simulation
def reset(event):
    global h, simulation_time
    h = initialize_order_parameters()
    simulation_time = 0.0
    im.set_data(np.argmax(h, axis=0))
    time_text.set_text(f"Time: {simulation_time:.2f}")
    plt.draw()

reset_button.on_clicked(reset)

# Animation function
def animate(frame):
    global h, simulation_time
    alpha = slider_alpha.val
    beta = slider_beta.val
    gamma = slider_gamma.val
    kappa = slider_kappa.val
    steps = int(slider_rate.val)

    L_array = np.ones(Q) * L_init
    kappa_array = np.ones(Q) * kappa

    for _ in range(steps):
        h = update_order_parameters(h, alpha, beta, gamma, kappa_array, L_array)
        simulation_time += dt

    im.set_data(np.argmax(h, axis=0))
    time_text.set_text(f"Time: {simulation_time:.2f}")
    return [im, time_text]

# Create the animation
ani = animation.FuncAnimation(fig, animate, blit=False, interval=5)

plt.show()

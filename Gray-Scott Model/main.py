import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from simulation import initialize, update, f, k, diff_U, diff_V, dt, dx, width, height, grid_size
from draggablepoint import DraggablePoint

# Initialize the activator (U) and inhibitor (V) arrays
U, V = initialize()

fig = plt.figure()

# Image axes
ax_img = fig.add_axes([0.2, 0.15, 0.6, 0.45])
# Display the first frame with color scaling
U_min, U_max = np.min(U), np.max(U)
img = ax_img.imshow(U, cmap='plasma', interpolation='bilinear', vmin=U_min, vmax=U_max,
                    extent=[0, width, 0, height], origin='lower')
ax_img.set_title('Gray-Scott Model')
ax_img.set_xlabel('X (nm)')
ax_img.set_ylabel('Y (nm)')

# Add color bar
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.45])
color_bar = plt.colorbar(img, cax=cbar_ax)

# Real-time update function
fps = 10  # frames per second

def real_time_update():
    global U, V
    n_steps = int(time_slider.val / fps / dt)
    if n_steps < 1:
        n_steps = 1

    print("f, k, diff_U, diff_V: ",f, k, diff_U, diff_V)
    for i in range(n_steps):
        U, V = update(U, V, f, k, diff_U, diff_V, dt, dx)

    # Update the image
    U_min, U_max = np.min(U), np.max(U)
    U_min, U_max = 0, 1
    img.set_data(U)
    img.set_clim(vmin=U_min, vmax=U_max)
    color_bar.update_normal(img)
    fig.canvas.draw_idle()

# Function to update the simulation continuously
def run_real_time_updates(event):
    real_time_update()
    fig.canvas.flush_events()

# 2D Slider for feed (f) and kill (k) rates
ax_f_k = fig.add_axes([0.2, 0.7, 0.3, 0.25], facecolor='lightgoldenrodyellow')

def f_k_update(x, y):
    global f, k
    f = x
    k = y

# Initialize with default values and 0.1 second interval for callback
draggable_point_f_k = DraggablePoint(
    ax_f_k, xlim=(0, 0.1), ylim=(0, 0.1), update_callback=f_k_update,
    default_value=(f, k), callback_interval=0.1
)

# Add labels for f (feed) and k (kill) sliders with units
ax_f_k.text(0.5, -0.3, 'f (1/ns)', transform=ax_f_k.transAxes, fontsize=10, horizontalalignment='center')
ax_f_k.text(-0.3, 0.5, 'k (1/ns)', transform=ax_f_k.transAxes, fontsize=10, verticalalignment='center', rotation=90)

# 2D Slider for diff_U and diff_V
ax_diff = fig.add_axes([0.5, 0.7, 0.3, 0.25], facecolor='lightgoldenrodyellow')

def diff_update(x, y):
    global diff_U, diff_V
    diff_U = x
    diff_V = y

# Initialize with default values and 0.1 second interval for callback
draggable_point_diff = DraggablePoint(
    ax_diff, xlim=(0, 0.1), ylim=(0, 0.1), update_callback=diff_update,
    default_value=(diff_U, diff_V), callback_interval=0.1
)

# Add labels for diff_U and diff_V sliders with units
ax_diff.text(0.5, -0.3, 'diff_U (nm$^2$/ns)', transform=ax_diff.transAxes, fontsize=10, horizontalalignment='center')
ax_diff.text(-0.3, 0.5, 'diff_V (nm$^2$/ns)', transform=ax_diff.transAxes, fontsize=10, verticalalignment='center', rotation=90)

# Slider for time evolution speed
ax_time_slider = fig.add_axes([0.25, 0.05, 0.5, 0.05], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax_time_slider, 'Speed (ns/s)', 10, 1000, valinit=100, valstep=10)

# Enable drawing functionality
drawing = False
circle_radius = 2  # Circle radius (corresponding to 10x10 diameter)

def on_mouse_press(event):
    global drawing
    if event.inaxes == ax_img:
        drawing = True
        add_draw(event)

def on_mouse_release(event):
    global drawing
    drawing = False

def on_mouse_move(event):
    if drawing:
        add_draw(event)

def add_draw(event):
    if event.inaxes == ax_img:
        # Convert mouse coordinates to grid coordinates
        x = int(event.xdata / dx)
        y = int(event.ydata / dx)
        if 0 <= x < grid_size and 0 <= y < grid_size:
            # Draw a 10x10 circle around the mouse position
            for i in range(-circle_radius, circle_radius + 1):
                for j in range(-circle_radius, circle_radius + 1):
                    if i**2 + j**2 <= circle_radius**2:  # Ensure it's within a circle
                        new_x = x + i
                        new_y = y + j
                        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                            V[new_y, new_x] = 1
            img.set_data(U)
            fig.canvas.draw_idle()

# Connect mouse events to the drawing functions
fig.canvas.mpl_connect('button_press_event', on_mouse_press)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Real-time simulation loop
timer = fig.canvas.new_timer(interval=100)  # Update every 100ms (10 FPS)
timer.add_callback(run_real_time_updates, None)
timer.start()

plt.show()

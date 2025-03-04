import sys
import numpy as np
from numba import njit, prange
import time
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider
)
from vispy import scene
from vispy.scene.visuals import Mesh, Line
from vispy.geometry import MeshData
from vispy.color import get_colormap


# ================================
# Global simulation parameters
# ================================

nx, ny = 100, 100       # spatial grid dimensions
Q = 50                # number of layers (z-dimension)
dx = dy = 1.0         # spatial steps
dt = 0.01             # time step
steps_per_frame = 20  # simulation steps per animation frame

# Initial slider values
gamma_init   = 25.0
epsilon_init = 0.6
alpha_init   = 0.4
beta_init    = 1000.0
M_init       = 1.0

# For voxel display: show voxel only if value > eps_vox.
eps_vox = 0.5

# ================================
# Cube geometry data (unit cube)
# with one corner at (0,0,0) and the opposite at (1,1,1)
# ================================

cube_vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
], dtype=np.float32)

cube_faces = np.array([
    [0, 1, 2], [0, 2, 3],  # front (z=0)
    [4, 5, 6], [4, 6, 7],  # back  (z=1)
    [0, 1, 5], [0, 5, 4],  # bottom (y=0)
    [1, 2, 6], [1, 6, 5],  # right  (x=1)
    [2, 3, 7], [2, 7, 6],  # top    (y=1)
    [3, 0, 4], [3, 4, 7]   # left   (x=0)
], dtype=np.int32)

cube_edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],   # front face edges
    [4, 5], [5, 6], [6, 7], [7, 4],   # back face edges
    [0, 4], [1, 5], [2, 6], [3, 7]    # side edges
], dtype=np.int32)


# ================================
# PDE simulation functions
# ================================

def initialize_layers():
    """
    Initialize the simulation state.
    Returns an array c of shape (Q, nx, ny) with
    the bottom layer set near 1 and small random fluctuations.
    """
    c_init = np.zeros((Q, nx, ny))
    c_init[0] = 1.0
    c_init += 0.05 * np.random.rand(Q, nx, ny)
    return c_init

@njit
def laplacian(u):
    """
    Compute the 2D Laplacian of u (with shape (Q, nx, ny))
    using periodic boundary conditions.
    """
    Q_val, nx_val, ny_val = u.shape
    lap = np.empty_like(u)
    for q in range(Q_val):
        for i in range(nx_val):
            ip1 = (i + 1) % nx_val
            im1 = (i - 1) % nx_val
            for j in range(ny_val):
                jp1 = (j + 1) % ny_val
                jm1 = (j - 1) % ny_val
                lap[q, i, j] = (u[q, ip1, j] + u[q, im1, j] +
                                u[q, i, jp1] + u[q, i, jm1] -
                                4.0 * u[q, i, j]) / (dx * dy)
    return lap

@njit
def gradient2D(u):
    """
    Compute the gradient (gx, gy) of a single 2D array u (nx, ny)
    with periodic boundary conditions.
    """
    nx_val, ny_val = u.shape
    gx = np.empty_like(u)
    gy = np.empty_like(u)
    for i in range(nx_val):
        ip1 = (i + 1) % nx_val
        im1 = (i - 1) % nx_val
        for j in range(ny_val):
            jp1 = (j + 1) % ny_val
            jm1 = (j - 1) % ny_val
            gx[i, j] = (u[ip1, j] - u[im1, j]) / (2 * dx)
            gy[i, j] = (u[i, jp1] - u[i, jm1]) / (2 * dy)
    return gx, gy

@njit
def divergence2D(gx, gy):
    """
    Compute the divergence of (gx, gy) with periodic boundary conditions.
    """
    nx_val, ny_val = gx.shape
    div = np.empty_like(gx)
    for i in range(nx_val):
        ip1 = (i + 1) % nx_val
        im1 = (i - 1) % nx_val
        for j in range(ny_val):
            jp1 = (j + 1) % ny_val
            jm1 = (j - 1) % ny_val
            div[i, j] = ((gx[ip1, j] - gx[im1, j]) / (2 * dx) +
                         (gy[i, jp1] - gy[i, jm1]) / (2 * dy))
    return div

@njit
def df_dc(c):
    """For f(C)=½·C²(1-C)², f'(C)=C(1-C)(1-2C)."""
    return c * (1 - c) * (1 - 2 * c)

@njit
def g_func(x):
    """Sigmoid function controlling coupling between layers.
       Works for both scalars and arrays.
    """
    return 1.0 / (1.0 + np.exp(-30.0 * (x - 0.85)))

@njit
def compute_mu(c, gamma, epsilon):
    """
    Compute mu_i = -a*lap(c_i) + b*f'(c_i),
    where a = gamma * epsilon and b = gamma / epsilon.
    """
    a = gamma * epsilon
    b = gamma / epsilon
    lap_c = laplacian(c)
    return -a * lap_c + b * df_dc(c)

@njit(parallel=True)
def update_layers(c, alpha, beta, M_val, gamma, epsilon):
    """
    Update the simulation state by one time step.
    The update equation is:
       dc_i/dt = div(M(c_{i-1}) grad(mu_i))
                 + alpha * g(c_{i-1}) * (1 - c_i)
                 + beta * g(1 - c_i) * c_{i+1}
                 - beta * g(1 - c_{i-1}) * c_i
    with boundary conditions:
       c_{-1} = 1.0 (for i=0) and c_{Q} = 0 (for i=Q-1).
    """
    Q_val, nx_val, ny_val = c.shape
    mu = compute_mu(c, gamma, epsilon)
    dc_dt = np.zeros_like(c)
    for i in prange(Q_val):
        if i > 0:
            c_im1 = c[i - 1]
        else:
            c_im1 = np.ones((nx_val, ny_val), c.dtype)
        if i < Q_val - 1:
            c_ip1 = c[i + 1]
        else:
            c_ip1 = np.zeros((nx_val, ny_val), c.dtype)
        M_i = M_val * g_func(c_im1)
        gx, gy = gradient2D(mu[i])
        flux_x = M_i * gx
        flux_y = M_i * gy
        div_flux = divergence2D(flux_x, flux_y)
        growth = alpha * g_func(c_im1) * (1 - c[i])
        step_up = beta * g_func(1.0 - c[i]) * c_ip1
        step_down = -beta * g_func(1.0 - c_im1) * c[i]

        for ix in range(nx_val):
            for iy in range(ny_val):
                dc_dt[i, ix, iy] = (div_flux[ix, iy] +
                                    growth[ix, iy] +
                                    step_up[ix, iy] +
                                    step_down[ix, iy])
    c_new = c + dc_dt * dt
    return np.clip(c_new, 0.0, 1.0)

@njit
def update_layers_several_steps(c, alpha, beta, M_val, gamma, epsilon, num_steps):
    """
    Update the simulation state by num_steps time steps.
    """
    for _ in range(num_steps):
        c = update_layers(c, alpha, beta, M_val, gamma, epsilon)
    return c

@njit
def build_voxel_geometry(c, eps_vox, cube_vertices, cube_faces, cube_edges, base_colors):
    """
    Build voxel geometry for all voxels where c > eps_vox.
    Precompute vertices, colors, faces, and edges arrays.
    """
    Q_val, nx_val, ny_val = c.shape
    max_count = Q_val * nx_val * ny_val
    vertices_arr = np.empty((max_count * 8, 3), dtype=np.float32)
    colors_arr = np.empty((max_count * 8, 4), dtype=np.float32)
    faces_arr = np.empty((max_count * 12, 3), dtype=np.int32)
    edges_arr = np.empty((max_count * 12 * 2, 3), dtype=np.float32)
    
    cube_count = 0
    face_count = 0
    edge_count = 0
    
    for i in range(Q_val):
        for x in range(nx_val):
            for y in range(ny_val):
                val = c[i, x, y]
                if val > eps_vox:
                    # Offset for this voxel.
                    ox = x
                    oy = y
                    oz = i
                    # Get the base color for this layer and override alpha.
                    r = base_colors[i, 0]
                    g_col = base_colors[i, 1]
                    b = base_colors[i, 2]
                    a = val
                    current_vertex = cube_count * 8
                    # Add cube vertices and assign colors.
                    for j in range(8):
                        vertices_arr[current_vertex + j, 0] = cube_vertices[j, 0] + ox
                        vertices_arr[current_vertex + j, 1] = cube_vertices[j, 1] + oy
                        vertices_arr[current_vertex + j, 2] = cube_vertices[j, 2] + oz
                        colors_arr[current_vertex + j, 0] = r
                        colors_arr[current_vertex + j, 1] = g_col
                        colors_arr[current_vertex + j, 2] = b
                        colors_arr[current_vertex + j, 3] = a
                    # Add faces (each face indices shifted by the current vertex offset).
                    for f in range(12):
                        faces_arr[face_count + f, 0] = cube_faces[f, 0] + current_vertex
                        faces_arr[face_count + f, 1] = cube_faces[f, 1] + current_vertex
                        faces_arr[face_count + f, 2] = cube_faces[f, 2] + current_vertex
                    face_count += 12
                    # Add edges: for each cube edge, add two vertices.
                    for e in range(cube_edges.shape[0]):
                        v0 = cube_edges[e, 0]
                        v1 = cube_edges[e, 1]
                        edges_arr[edge_count, 0] = cube_vertices[v0, 0] + ox
                        edges_arr[edge_count, 1] = cube_vertices[v0, 1] + oy
                        edges_arr[edge_count, 2] = cube_vertices[v0, 2] + oz
                        edges_arr[edge_count + 1, 0] = cube_vertices[v1, 0] + ox
                        edges_arr[edge_count + 1, 1] = cube_vertices[v1, 1] + oy
                        edges_arr[edge_count + 1, 2] = cube_vertices[v1, 2] + oz
                        edge_count += 2
                    cube_count += 1
    return cube_count, face_count, edge_count, vertices_arr, colors_arr, faces_arr, edges_arr


# ================================
# Main Application Class using PyQt and Vispy
# ================================

class VoxelSimulationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up simulation state.
        self.c = initialize_layers()
        self.simulation_time = 0.0

        # Create a Vispy SceneCanvas with a white background.
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
                                        size=(800, 600), show=True)
        self.view = self.canvas.central_widget.add_view()
        # Set the camera to a TurntableCamera with its center at the grid's center.
        self.view.camera = scene.cameras.TurntableCamera(
            fov=60, distance=50, center=(nx/2, ny/2, Q/2)
        )

        # Create visuals: one Mesh for voxel faces, one Line for edges.
        self.mesh = Mesh()
        self.view.add(self.mesh)
        self.lines = Line(color='black', width=1, method='gl', connect='segments')
        self.view.add(self.lines)

        # Use the underlying Qt widget for layout.
        self.canvas_native = self.canvas.native

        # Precompute base colors for each layer using Vispy’s viridis colormap.
        cmap = get_colormap('viridis')
        self.base_colors = np.empty((Q, 4), dtype=np.float32)
        for i in range(Q):
            self.base_colors[i] = cmap.map(np.array([i / Q]))[0]

        # Create PyQt sliders for the five parameters.
        self.alpha_slider = QSlider(QtCore.Qt.Horizontal)
        self.beta_slider = QSlider(QtCore.Qt.Horizontal)
        self.M_slider = QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider = QSlider(QtCore.Qt.Horizontal)
        self.epsilon_slider = QSlider(QtCore.Qt.Horizontal)
        for slider in [self.alpha_slider, self.beta_slider, self.M_slider,
                       self.gamma_slider, self.epsilon_slider]:
            slider.setRange(0, 3000)
        self.alpha_slider.setValue(int(alpha_init * 100))
        self.beta_slider.setValue(int(beta_init * 100))
        self.M_slider.setValue(int(M_init * 100))
        self.gamma_slider.setValue(int(gamma_init * 100))
        self.epsilon_slider.setValue(int(epsilon_init * 100))

        # Create labels to display current values.
        self.alpha_label = QLabel(f"alpha: {alpha_init:.2f}")
        self.beta_label = QLabel(f"beta: {beta_init:.2f}")
        self.M_label = QLabel(f"M: {M_init:.2f}")
        self.gamma_label = QLabel(f"gamma: {gamma_init:.2f}")
        self.epsilon_label = QLabel(f"epsilon: {epsilon_init:.2f}")

        # Create pause and reset buttons.
        self.paused = False
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)

        # Set up layout.
        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.canvas_native)

        slider_layout = QGridLayout()
        slider_layout.addWidget(self.alpha_label, 0, 0)
        slider_layout.addWidget(self.alpha_slider, 0, 1)
        slider_layout.addWidget(self.beta_label, 1, 0)
        slider_layout.addWidget(self.beta_slider, 1, 1)
        slider_layout.addWidget(self.M_label, 2, 0)
        slider_layout.addWidget(self.M_slider, 2, 1)
        slider_layout.addWidget(self.gamma_label, 3, 0)
        slider_layout.addWidget(self.gamma_slider, 3, 1)
        slider_layout.addWidget(self.epsilon_label, 4, 0)
        slider_layout.addWidget(self.epsilon_slider, 4, 1)
        main_layout.addLayout(slider_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reset_button)
        main_layout.addLayout(button_layout)

        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setWindowTitle("Voxel Simulation with Vispy")

        # Set up a QTimer for animation.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(100)  # milliseconds

    def animate(self):
        # Read slider values (scaling by 1/100).
        alpha_val = self.alpha_slider.value() / 100.0
        beta_val = self.beta_slider.value() / 100.0
        M_val = self.M_slider.value() / 100.0
        gamma_val = self.gamma_slider.value() / 100.0
        epsilon_val = self.epsilon_slider.value() / 100.0

        self.alpha_label.setText(f"alpha: {alpha_val:.2f}")
        self.beta_label.setText(f"beta: {beta_val:.2f}")
        self.M_label.setText(f"M: {M_val:.2f}")
        self.gamma_label.setText(f"gamma: {gamma_val:.2f}")
        self.epsilon_label.setText(f"epsilon: {epsilon_val:.2f}")
        
        start = time.time()
        # Run several simulation steps.
        #for _ in range(steps_per_frame):
        #    self.c = update_layers(self.c, alpha_val, beta_val, M_val, gamma_val, epsilon_val)
        #    self.simulation_time += dt

        self.c = update_layers_several_steps(self.c, alpha_val, beta_val, M_val, gamma_val, epsilon_val, steps_per_frame)
        self.simulation_time += dt * steps_per_frame   

        print(f"Time for {steps_per_frame} simulation steps: {time.time() - start:.3f} s")
        # Build voxel geometry using the numba-optimized function.

        start = time.time()
        cube_count, face_count, edge_count, vertices_arr, colors_arr, faces_arr, edges_arr = \
            build_voxel_geometry(self.c, eps_vox, cube_vertices, cube_faces, cube_edges, self.base_colors)
        if cube_count > 0:
            vertices = vertices_arr[:cube_count * 8, :]
            colors = colors_arr[:cube_count * 8, :]
            faces = faces_arr[:face_count, :]
            edges = edges_arr[:edge_count, :]
        else:
            vertices = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 4), dtype=np.float32)
            faces = np.zeros((0, 3), dtype=np.int32)
            edges = np.zeros((0, 3), dtype=np.float32)

        print(f"Time to build geometry: {time.time() - start:.3f} s")

        start = time.time()

        # Update the Mesh visual with the new voxel faces.
        self.mesh.set_data(vertices=vertices, faces=faces, vertex_colors=colors)
        # Update the Line visual with the edges (drawn in black).
        self.lines.set_data(pos=edges, connect='segments', color='black')
        self.canvas.update()
        print(f"Time to update visuals: {time.time() - start:.3f} s")

    def toggle_pause(self):
        if not self.paused:
            self.paused = True
            self.pause_button.setText("Resume")
            self.timer.stop()
        else:
            self.paused = False
            self.pause_button.setText("Pause")
            self.timer.start(50)

    def reset_simulation(self):
        self.c = initialize_layers()
        self.simulation_time = 0.0

# ================================
# Main: start the application
# ================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    import vispy
    print(vispy.sys_info())
    win = VoxelSimulationApp()
    win.show()
    sys.exit(app.exec_())

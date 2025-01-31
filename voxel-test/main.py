import sys
import numpy as np

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

nx, ny = 30, 30       # spatial grid dimensions
Q = 20                # number of layers (z-dimension)
dx = dy = 1.0         # spatial steps
dt = 0.01             # time step
steps_per_frame = 20  # simulation steps per animation frame

# Initial slider values
gamma_init   = 25.0
epsilon_init = 0.6
alpha_init   = 0.4
beta_init    = 0.0
M_init       = 1.0

# For voxel display: show voxel only if value > eps_vox.
eps_vox = 0.1

# ================================
# Cube geometry data (unit cube)
# with one corner at (0,0,0) and the opposite at (1,1,1)
# ================================

# 8 vertices of a cube.
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

# Define 6 faces as 2 triangles each (12 triangles total).
# Each face is specified as a quad split into two triangles.
cube_faces = np.array([
    [0, 1, 2], [0, 2, 3],  # front (z=0)
    [4, 5, 6], [4, 6, 7],  # back  (z=1)
    [0, 1, 5], [0, 5, 4],  # bottom (y=0)
    [1, 2, 6], [1, 6, 5],  # right  (x=1)
    [2, 3, 7], [2, 7, 6],  # top    (y=1)
    [3, 0, 4], [3, 4, 7]   # left   (x=0)
], dtype=np.int32)

# Define the 12 edges of a cube as pairs of vertex indices.
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

def laplacian(u):
    """
    Compute the 2D Laplacian of u (with shape (Q, nx, ny))
    using periodic boundary conditions.
    """
    u_padded = np.pad(u, ((0, 0), (1, 1), (1, 1)), mode='wrap')
    lap_u = (
        u_padded[:, :-2, 1:-1] +
        u_padded[:, 2:  , 1:-1] +
        u_padded[:, 1:-1, :-2] +
        u_padded[:, 1:-1, 2:  ] -
        4.0 * u
    ) / (dx * dy)
    return lap_u

def gradient2D(u):
    """
    Compute the gradient (gx, gy) of a single 2D array u (nx, ny)
    with periodic boundary conditions.
    """
    up = np.pad(u, ((1,1),(1,1)), mode='wrap')
    gx = (up[2:, 1:-1] - up[:-2, 1:-1]) / (2*dx)
    gy = (up[1:-1, 2:] - up[1:-1, :-2]) / (2*dy)
    return gx, gy

def divergence2D(gx, gy):
    """
    Compute the divergence of (gx, gy) with periodic boundary conditions.
    """
    gx_p = np.pad(gx, ((1,1),(0,0)), mode='wrap')
    gy_p = np.pad(gy, ((0,0),(1,1)), mode='wrap')
    dgx_dx = (gx_p[2:] - gx_p[:-2]) / (2*dx)
    dgy_dy = (gy_p[:,2:] - gy_p[:,:-2]) / (2*dy)
    return dgx_dx + dgy_dy

def df_dc(c):
    """For f(C)=½·C²(1-C)², f'(C)=C(1-C)(1-2C)."""
    return c * (1 - c) * (1 - 2*c)

def g(x):
    """Sigmoid function controlling coupling between layers."""
    return 1.0 / (1.0 + np.exp(-30.0 * (x - 0.85)))

def compute_mu(c, gamma, epsilon):
    """
    Compute mu_i = -a*lap(c_i) + b*f'(c_i),
    where a = gamma * epsilon and b = gamma / epsilon.
    """
    a = gamma * epsilon
    b = gamma / epsilon
    lap_c = laplacian(c)
    return -a * lap_c + b * df_dc(c)

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
    
    for i in range(Q_val):
        c_im1 = c[i-1] if i > 0 else np.ones((nx_val, ny_val))
        c_ip1 = c[i+1] if i < Q_val-1 else np.zeros((nx_val, ny_val))
        M_i = M_val * g(c_im1)
        gx, gy = gradient2D(mu[i])
        flux_x = M_i * gx
        flux_y = M_i * gy
        div_flux = divergence2D(flux_x, flux_y)
        growth   = alpha * g(c_im1) * (1 - c[i])
        step_up  = beta * g(1.0 - c[i]) * c_ip1
        step_down = -beta * g(1.0 - c_im1) * c[i]
        dc_dt[i] = div_flux + growth + step_up + step_down

    c_new = c + dc_dt * dt
    return np.clip(c_new, 0.0, 1.0)

# ================================
# Main Application Class using PyQt and Vispy
# ================================

class VoxelSimulationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up simulation state
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

        # Create visuals: one Mesh for colored voxel faces, one Line visual for black edges.
        self.mesh = Mesh()
        self.view.add(self.mesh)
        self.lines = Line(color='black', width=1, method='gl', connect='segments')
        self.view.add(self.lines)

        # Use the underlying Qt widget for layout.
        self.canvas_native = self.canvas.native

        # Create PyQt sliders for the five parameters.
        # We set slider ranges from 0 to 3000; the slider value is divided by 100 to get a float.
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

        # Create labels to display the current values.
        self.alpha_label = QLabel(f"alpha: {alpha_init:.2f}")
        self.beta_label = QLabel(f"beta: {beta_init:.2f}")
        self.M_label = QLabel(f"M: {M_init:.2f}")
        self.gamma_label = QLabel(f"gamma: {gamma_init:.2f}")
        self.epsilon_label = QLabel(f"epsilon: {epsilon_init:.2f}")

        # Create a pause button.
        self.paused = False
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)

        # Create a reset button.
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)

        # Set up the layout.
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

        # Create a horizontal layout for the buttons.
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
        # Read slider values (scale by 1/100).
        alpha_val = self.alpha_slider.value() / 100.0
        beta_val = self.beta_slider.value() / 100.0
        M_val = self.M_slider.value() / 100.0
        gamma_val = self.gamma_slider.value() / 100.0
        epsilon_val = self.epsilon_slider.value() / 100.0

        # Update slider labels.
        self.alpha_label.setText(f"alpha: {alpha_val:.2f}")
        self.beta_label.setText(f"beta: {beta_val:.2f}")
        self.M_label.setText(f"M: {M_val:.2f}")
        self.gamma_label.setText(f"gamma: {gamma_val:.2f}")
        self.epsilon_label.setText(f"epsilon: {epsilon_val:.2f}")

        # Run several simulation steps.
        for _ in range(steps_per_frame):
            self.c = update_layers(self.c, alpha_val, beta_val, M_val, gamma_val, epsilon_val)
            self.simulation_time += dt

        # Build new geometry for visible voxels.
        all_vertices = []   # list of (8,3) arrays (one cube = 8 vertices)
        all_colors = []     # list of (8,4) arrays (one color per vertex)
        all_faces = []      # list of (12,3) arrays (each cube: 12 triangles)
        all_edges = []      # list of points; each pair will form one line segment.
        vertex_offset = 0

        # Use Vispy’s viridis colormap.
        cmap = get_colormap('viridis')

        # Our simulation state c has shape (Q, nx, ny).
        # For each layer (index i), assign a base color from the colormap.
        for i in range(Q):
            base_color = cmap.map(np.array([i / Q]))[0]  # RGBA; we will override the alpha.
            for x in range(nx):
                for y in range(ny):
                    val = self.c[i, x, y]
                    if val > eps_vox:
                        voxel_color = [base_color[0], base_color[1], base_color[2], val]
                        offset = np.array([x, y, i], dtype=np.float32)
                        v = cube_vertices + offset  # shift cube vertices by the voxel position
                        all_vertices.append(v)
                        all_colors.append(np.tile(voxel_color, (8, 1)))
                        all_faces.append(cube_faces + vertex_offset)
                        for edge in cube_edges:
                            all_edges.append(v[edge[0]])
                            all_edges.append(v[edge[1]])
                        vertex_offset += 8

        if all_vertices:
            vertices = np.concatenate(all_vertices, axis=0)  # shape (N,3)
            colors = np.concatenate(all_colors, axis=0)        # shape (N,4)
            faces = np.concatenate(all_faces, axis=0)          # shape (num_faces,3)
            edges = np.array(all_edges, dtype=np.float32)      # shape (num_edge_pts, 3)
        else:
            vertices = np.zeros((0,3), dtype=np.float32)
            colors = np.zeros((0,4), dtype=np.float32)
            faces = np.zeros((0,3), dtype=np.int32)
            edges = np.zeros((0,3), dtype=np.float32)

        # Update the Mesh visual with the new voxel faces.
        self.mesh.set_data(vertices=vertices, faces=faces, vertex_colors=colors)

        # Update the Line visual with the edges (drawn in black).
        self.lines.set_data(pos=edges, connect='segments', color='black')

        self.canvas.update()

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
    win = VoxelSimulationApp()
    win.show()
    sys.exit(app.exec_())

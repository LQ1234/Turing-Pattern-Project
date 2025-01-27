import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d  # NOQA
from collections import defaultdict
import types
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

# ----------------------------------------
# 1) Monkeypatched voxels method
# ----------------------------------------
def monkeypatched_voxels(self, *args, **kwargs):
    """
    Custom 'voxels' method that allows internal faces, broadcasting of
    facecolors/edgecolors, etc., now using np.broadcast_to instead of _backports.
    """

    # Decide how to parse args: if len(args) >= 3, they are x,y,z; otherwise just 'filled'.
    if len(args) >= 3:
        def _voxels(__x, __y, __z, filled, **kwargs):
            return (__x, __y, __z), filled, kwargs
    else:
        def _voxels(filled, **kwargs):
            return None, filled, kwargs

    xyz, filled, kwargs = _voxels(*args, **kwargs)

    if filled.ndim != 3:
        raise ValueError("Argument 'filled' must be 3-dimensional")

    size = np.array(filled.shape, dtype=np.intp)
    coord_shape = tuple(size + 1)  # coordinate arrays have shape = size+1

    if xyz is None:
        x, y, z = np.indices(coord_shape)
    else:
        # broadcast each coordinate array to the shape coord_shape
        x, y, z = (np.broadcast_to(c, coord_shape) for c in xyz)

    def _broadcast_color_arg(color, name):
        """
        Let 'color' be single or shape-matching 'filled'.
        If single => broadcast to entire filled shape.
        If 3D or 4D => must match shape of 'filled' exactly.
        """
        dim = np.ndim(color)
        if dim in (0, 1):
            # e.g. 'red' or [0.5,0.5,0.5] => broadcast
            return np.broadcast_to(color, filled.shape + np.shape(color))
        elif dim in (3, 4):
            # Possibly RGBA array
            # shape of color up to the first 3 dims must match filled
            if np.shape(color)[:3] != filled.shape:
                raise ValueError(
                    f"When multidimensional, {name} must match the shape of 'filled'"
                )
            return color
        else:
            raise ValueError(f"Invalid {name} argument: wrong dimensionality")

    # Face colors
    facecolors = kwargs.pop('facecolors', None)
    if facecolors is None:
        # fallback to next color cycle
        facecolors = self._get_patches_for_fill.get_next_color()
    facecolors = _broadcast_color_arg(facecolors, 'facecolors')

    # Edge colors
    edgecolors = kwargs.pop('edgecolors', None)
    edgecolors = _broadcast_color_arg(edgecolors, 'edgecolors')

    # Draw internal faces or not
    internal_faces = kwargs.pop('internal_faces', False)

    # Always auto-scale to entire domain
    self.auto_scale_xyz(x, y, z)

    # Points of a unit square in XY for constructing each face
    square = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0]
    ], dtype=np.intp)

    voxel_faces = defaultdict(list)

    def permutation_matrices(n):
        """Generator of cyclic permutation matrices."""
        mat = np.eye(n, dtype=np.intp)
        for _ in range(n):
            yield mat
            mat = np.roll(mat, 1, axis=0)

    # Enumerate each face in the voxel array
    for permute in permutation_matrices(3):
        pc, qc, rc = permute.T.dot(size)
        pinds = np.arange(pc)
        qinds = np.arange(qc)
        rinds = np.arange(rc)

        square_rot = square.dot(permute.T)

        for p in pinds:
            for q in qinds:
                # "bottom" face
                p0 = permute.dot([p, q, 0])
                i0 = tuple(p0)
                if filled[i0]:
                    voxel_faces[i0].append(p0 + square_rot)

                # Middle slices (internal surfaces)
                for r1, r2 in zip(rinds[:-1], rinds[1:]):
                    p1 = permute.dot([p, q, r1])
                    p2 = permute.dot([p, q, r2])
                    i1 = tuple(p1)
                    i2 = tuple(p2)
                    if filled[i1] and (internal_faces or not filled[i2]):
                        voxel_faces[i1].append(p2 + square_rot)
                    elif (internal_faces or not filled[i1]) and filled[i2]:
                        voxel_faces[i2].append(p2 + square_rot)

                # "top" face
                pk = permute.dot([p, q, rc - 1])
                pk2 = permute.dot([p, q, rc])
                ik = tuple(pk)
                if filled[ik]:
                    voxel_faces[ik].append(pk2 + square_rot)

    # Build a Poly3DCollection for each voxel
    polygons = {}
    for coord, faces_inds in voxel_faces.items():
        if xyz is None:
            # indices as direct coords
            faces = faces_inds
        else:
            # Convert indices -> real coords
            faces = []
            for face_inds in faces_inds:
                ind = face_inds[:, 0], face_inds[:, 1], face_inds[:, 2]
                face = np.empty(face_inds.shape)
                face[:, 0] = x[ind]
                face[:, 1] = y[ind]
                face[:, 2] = z[ind]
                faces.append(face)

        poly = art3d.Poly3DCollection(
            faces,
            facecolors=facecolors[coord],
            edgecolors=edgecolors[coord],
            **kwargs
        )
        self.add_collection3d(poly)
        polygons[coord] = poly

    return polygons


# ----------------------------------------
# 2) New PDE model and simulation
# ----------------------------------------

# Default parameters
nx, ny = 20, 20       # Grid size
Q = 5                 # Number of layers
dx = dy = 1.0         # Space steps
dt = 0.01             # Time step
steps_per_frame = 20  # Simulation steps per animation frame

# Slider initial values
gamma_init   = 25.0
epsilon_init = 0.6
alpha_init   = 0.4
beta_init    = 0.0
M_init       = 1.0

# Sliders will go from 0 to 30
slider_min = 0.0
slider_max = 30.0

def initialize_layers():
    """Initialize c_i(x,y,t) with slight randomness, bottom layer ~1."""
    c_init = np.zeros((Q, nx, ny))
    # Fill bottom layer with 1
    c_init[0] = 1.0
    # Add small random fluctuations
    c_init += 0.05*np.random.rand(Q, nx, ny)
    return c_init

c = initialize_layers()

def laplacian(u):
    """
    Compute 2D laplacian of u (shape = (Q,nx,ny))
    using periodic boundary conditions.
    """
    u_padded = np.pad(u, ((0, 0), (1, 1), (1, 1)), mode='wrap')
    lap_u = (
        u_padded[:, 0:-2, 1:-1] +
        u_padded[:, 2:   , 1:-1] +
        u_padded[:, 1:-1, 0:-2] +
        u_padded[:, 1:-1, 2:   ] -
        4.0 * u
    ) / (dx * dy)
    return lap_u

def gradient2D(u):
    """
    Compute the gradient (gx, gy) of a single 2D array u (nx, ny)
    with periodic BC.
    """
    up = np.pad(u, ((1,1),(1,1)), mode='wrap')
    gx = (up[2:,1:-1] - up[:-2,1:-1])/(2*dx)
    gy = (up[1:-1,2:] - up[1:-1,:-2])/(2*dy)
    return gx, gy

def divergence2D(gx, gy):
    """
    Compute divergence of (gx, gy) with periodic BC.
    """
    gx_p = np.pad(gx, ((1,1),(0,0)), mode='wrap')
    gy_p = np.pad(gy, ((0,0),(1,1)), mode='wrap')
    dgx_dx = (gx_p[2:] - gx_p[:-2]) / (2*dx)
    dgy_dy = (gy_p[:,2:] - gy_p[:,:-2]) / (2*dy)
    return dgx_dx + dgy_dy

def df_dc(c):
    """f(C) = 1/2 * C^2 * (1-C)^2 => f'(C) = C(1-C)(1-2C)."""
    return c*(1 - c)*(1 - 2*c)

def g(x):
    """Sigmoid controlling presence of previous layer."""
    return 1.0 / (1.0 + np.exp(-30.0 * (x - 0.85)))

def compute_mu(c, gamma, epsilon):
    """
    mu_i = -a lap(c_i) + b f'(c_i),
    a = gamma*epsilon, b = gamma/epsilon.
    """
    a = gamma * epsilon
    b = gamma / epsilon
    lap_c = laplacian(c)   # shape (Q,nx,ny)
    return -a * lap_c + b * df_dc(c)

def update_layers(c, alpha, beta, M_val, gamma, epsilon):
    """
    dc_i/dt = div( M(c_{i-1}) grad mu_i )
              + alpha*g(c_{i-1})*(1 - c_i)
              + beta*g(1 - c_i)*c_{i+1}
              - beta*g(1 - c_{i-1})*c_i
    with M(c_{i-1}) = M_val*g(c_{i-1}).
    Boundaries: c_{-1} = 0 if i=0, c_{Q} = 0 if i=Q-1.
    """
    Q, nx, ny = c.shape
    mu = compute_mu(c, gamma, epsilon)  # shape (Q,nx,ny)
    dc_dt = np.zeros_like(c)

    for i in range(Q):
        if i == 0:
            c_im1 = np.ones((nx, ny))
        else:
            c_im1 = c[i-1]
        if i == Q-1:
            c_ip1 = np.zeros((nx, ny))
        else:
            c_ip1 = c[i+1]

        # Mobility
        M_i = M_val*g(c_im1)

        # Divergence of M_i * grad(mu_i)
        grad_mu_x, grad_mu_y = gradient2D(mu[i])
        flux_x = M_i * grad_mu_x
        flux_y = M_i * grad_mu_y
        div_flux = divergence2D(flux_x, flux_y)

        # Growth/step terms
        growth = alpha*g(c_im1)*(1 - c[i])
        step_up = beta*g(1.0 - c[i])*c_ip1
        step_down = -beta*g(1.0 - c_im1)*c[i]

        dc_dt[i] = div_flux + growth + step_up + step_down

    # Forward-Euler step
    c_new = c + dc_dt*dt
    return np.clip(c_new, 0.0, 1.0)

# ----------------------------------------
# Matplotlib figure/animation setup
# ----------------------------------------
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[0.1, 1.0], hspace=0.2)

# Axes for the sliders on top
ax_sliders = fig.add_subplot(gs[0, 0])
ax_sliders.axis('off')

# Axes for the 3D voxel plot
ax_voxel = fig.add_subplot(gs[1, 0], projection='3d')

# Monkeypatch our custom 'voxels' onto ax_voxel
ax_voxel.voxels = types.MethodType(monkeypatched_voxels, ax_voxel)

time_template = 'Time = %.2f'
time_text = ax_voxel.text2D(0.05, 0.95, '', transform=ax_voxel.transAxes)

# Build sliders
slider_left  = 0.1
slider_width = 0.7
slider_height = 0.03
slider_spacing = 0.01
slider_count = 5

axcolor = 'lightgoldenrodyellow'
slider_axes = []
slider_labels = ['alpha','beta','M','gamma','epsilon']
slider_inits = [alpha_init, beta_init, M_init, gamma_init, epsilon_init]
sliders = []

for i in range(slider_count):
    ax_i = plt.axes([
        slider_left,
        0.95 - (i+1)*(slider_height + slider_spacing),
        slider_width,
        slider_height
    ], facecolor=axcolor)
    slider_axes.append(ax_i)

# Create each slider, range [0,30]
for (ax_i, label, valinit) in zip(slider_axes, slider_labels, slider_inits):
    s = Slider(ax_i, label, slider_min, slider_max, valinit=valinit)
    sliders.append(s)

# Reset Button
resetax = plt.axes([0.85, 0.02, 0.1, 0.04])
button_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def init_plot():
    ax_voxel.clear()
    ax_voxel.set_xlim(0, nx)
    ax_voxel.set_ylim(0, ny)
    ax_voxel.set_zlim(0, Q)
    ax_voxel.set_xlabel('X')
    ax_voxel.set_ylabel('Y')
    ax_voxel.set_zlabel('Layer')
    ax_voxel.set_title('Layered Deposition Simulation')
    return []

simulation_time = 0.0

def animate(frame):
    global c, simulation_time

    alpha_val   = sliders[0].val
    beta_val    = sliders[1].val
    M_val       = sliders[2].val
    gamma_val   = sliders[3].val
    epsilon_val = sliders[4].val

    for _ in range(steps_per_frame):
        c = update_layers(c, alpha_val, beta_val, M_val, gamma_val, epsilon_val)
        simulation_time += dt

    # Prepare boolean array for voxels
    eps_vox = 0.1
    voxels_bool = np.zeros((nx, ny, Q), dtype=bool)
    # Facecolors array: shape (nx, ny, Q, 4)
    facecolors = np.zeros((nx, ny, Q, 4), dtype=float)

    for i in range(Q):
        # Where c_i > eps_vox, we show a voxel
        layer_bool = (c[i] > eps_vox)
        voxels_bool[:, :, i] = layer_bool

        # Map layer i => color
        rgb = plt.cm.viridis(i / Q)[:3]
        alpha_layer = np.clip(c[i], 0, 1)  # alpha from [0..1] per cell
        facecolors[:, :, i, 0] = rgb[0]
        facecolors[:, :, i, 1] = rgb[1]
        facecolors[:, :, i, 2] = rgb[2]
        facecolors[:, :, i, 3] = alpha_layer

        # Where c_i < eps_vox => alpha=0
        facecolors[:, :, i, 3][~layer_bool] = 0.0

    ax_voxel.clear()
    # We now call our monkeypatched 'voxels'
    ax_voxel.voxels(
        voxels_bool, facecolors=facecolors,
        edgecolors='k',
        internal_faces=True  # enable seeing internal boundaries
    )
    ax_voxel.set_xlim(0, nx)
    ax_voxel.set_ylim(0, ny)
    ax_voxel.set_zlim(0, Q)
    ax_voxel.set_xlabel('X')
    ax_voxel.set_ylabel('Y')
    ax_voxel.set_zlabel('Layer')
    ax_voxel.set_title('Layered Deposition Simulation')

    ax_voxel.text2D(0.05, 0.95, time_template % simulation_time,
                    transform=ax_voxel.transAxes)
    plt.draw()
    return []

def reset(event):
    global c, simulation_time
    c = initialize_layers()
    simulation_time = 0.0

button_reset.on_clicked(reset)

ani = animation.FuncAnimation(
    fig, animate, init_func=init_plot,
    frames=200, interval=50, blit=False
)

plt.show()

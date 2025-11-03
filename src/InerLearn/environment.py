"""
environment.py
====================
Ultra-realistic 3D simulation environment with diverse, Minecraft-like terrain and complex missions.

Features:
 - Fractal procedural terrain with valleys, hills, lakes, and plains
 - Random mission goals and realistic obstacles (cylinders, rocks, towers)
 - Dynamic temperature, wind, and pollution
 - Long-range trajectory across kilometers
 - Unity-style 3D visualization with temperature-based coloring
"""

import numpy as np
import os
import pickle
import plotly.graph_objects as go

# ==============================================================
# Environment persistence utilities
# ==============================================================
def load_or_create_environment(
    path="environment_cache.pkl",
    area_size=2000,
    n_obstacles=120,
    n_goals=12,
    seed=42,
    force_regen=False
):
    """
    Load a cached environment if it exists locally; otherwise generate a new one and store it.
    This avoids regenerating terrain, goals, and obstacles every run.
    """
    if not force_regen and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                env = pickle.load(f)
            print(f"[INFO] Loaded cached environment from {path}")
            return env
        except Exception as e:
            print(f"[WARN] Failed to load cached environment ({e}), regenerating...")

    # Create new environment
    env = Environment(area_size=area_size, n_obstacles=n_obstacles, n_goals=n_goals, seed=seed)
    with open(path, "wb") as f:
        pickle.dump(env, f)
    print(f"[INFO] Generated and cached new environment at {path}")
    return env

# ==============================================================
# Utility: Fractal terrain noise
# ==============================================================
def fractal_noise(x, y, seed=None, octaves=6, persistence=0.5, lacunarity=2.0, base_scale=0.003, area_scale=1000.0):
    """Generate scale-aware fractal noise terrain with random offsets per seed."""
    rng = np.random.default_rng(seed)
    noise = np.zeros_like(x, dtype=float)

    # Adjust base frequency to map size
    scale = base_scale * (1000.0 / area_scale)

    freq = scale
    amp = 1.0
    for _ in range(octaves):
        randx, randy = rng.uniform(0, 10000, 2)
        noise += amp * np.sin(freq * (x + randx)) * np.cos(freq * (y + randy))
        freq *= lacunarity
        amp *= persistence
    return noise


# ==============================================================
# Environment
# ==============================================================
class Environment:
    """Physical world with terrain, obstacles, wind, temperature, and goals."""
    def __init__(self, area_size=2000, n_obstacles=120, n_goals=12, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.area = area_size
        self.wind_mag = 1.5
        self.temp_amp = 12.0
        self.base_temp = 22.0
        self.pollution_density = 0.4
        self.terrain_amp = 60.0
        self.obstacles = self._generate_obstacles(n_obstacles)
        self.goal_points = self._generate_goals(n_goals)

    # ---------------- Terrain & Environment Fields ----------------
    def terrain_height(self, x, y):
        """Scale-aware procedural terrain (new details for larger maps)."""
        terrain = self.terrain_amp * fractal_noise(
            x, y,
            seed=self.seed + 1337,
            area_scale=self.area
        )
        terrain = np.where(terrain < -10, -10 + 0.2 * terrain, terrain)
        return terrain

    def terrain_slope(self, x, y):
        dx = 2.0
        dzdx = (self.terrain_height(x+dx, y)-self.terrain_height(x-dx, y))/(2*dx)
        dzdy = (self.terrain_height(x, y+dx)-self.terrain_height(x, y-dx))/(2*dx)
        return np.array([dzdx, dzdy])

    def wind(self, t):
        """Slowly varying wind gusts."""
        return np.array([
            self.wind_mag*(0.7*np.sin(0.001*t)+0.3*np.sin(0.004*t+1.3)),
            self.wind_mag*(0.6*np.cos(0.0012*t)+0.3*np.sin(0.0025*t)),
            0.0
        ])

    def pollution_field(self, x, y):
        """Pollution hotspots that influence temperature."""
        return 0.5*(np.sin(0.015*x)*np.cos(0.02*y)+1)*self.pollution_density

    def temperature(self, x, y, t):
        terrain = self.terrain_height(x, y)
        pollution = self.pollution_field(x, y)
        diurnal = np.sin(0.0004*t)
        return self.base_temp + self.temp_amp*diurnal - 0.25*terrain + 2.5*pollution

    # ---------------- Obstacles & Goals ----------------
    def _generate_obstacles(self, n):
        """Generate cylindrical and cubic obstacles of varying sizes."""
        obstacles = []
        for _ in range(n):
            kind = np.random.choice(["cylinder", "cube", "dome"])
            x, y = np.random.uniform(-self.area/2, self.area/2, 2)
            size = np.random.uniform(10, 40)
            height = np.random.uniform(10, 80)
            obstacles.append(dict(kind=kind, x=x, y=y, size=size, height=height))
        return obstacles

    def _generate_goals(self, n):
        """Spread goals randomly throughout the area."""
        gx = np.random.uniform(-self.area/2, self.area/2, n)
        gy = np.random.uniform(-self.area/2, self.area/2, n)
        return np.column_stack((gx, gy))

    def obstacle_force(self, pos):
        """Compute repulsive force field."""
        force = np.zeros(2)
        for obs in self.obstacles:
            diff = pos - np.array([obs["x"], obs["y"]])
            dist = np.linalg.norm(diff)
            if dist < obs["size"] * 1.5:
                force += 150 * diff / (dist**3 + 1e-3)
        return force


# ==============================================================
# Trajectory Generator
# ==============================================================
class TrajectoryGenerator:
    """Generates a mission-scale trajectory that visits all goals."""
    def __init__(self, env: Environment, T=3000.0, dt=0.05, v_mean=6.0):
        self.env, self.T, self.dt, self.v_mean = env, T, dt, v_mean

    def generate(self):
        t = np.arange(0, self.T, self.dt)
        N = len(t)
        pos, vel, acc, gyro = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3))
        heading = np.random.uniform(0, 2*np.pi)
        goal_id = 0

        for i in range(1, N):
            goal = self.env.goal_points[goal_id]
            to_goal = goal - pos[i-1, :2]
            dist_goal = np.linalg.norm(to_goal)
            if dist_goal < 20 and goal_id < len(self.env.goal_points)-1:
                goal_id += 1

            dir_goal = to_goal / (dist_goal + 1e-6)
            repulsion = self.env.obstacle_force(pos[i-1, :2])
            nav_vec = dir_goal - 0.02 * repulsion
            heading = np.arctan2(nav_vec[1], nav_vec[0])

            wind = self.env.wind(t[i])
            slope = self.env.terrain_slope(pos[i-1,0], pos[i-1,1])
            terrain_z = self.env.terrain_height(pos[i-1,0], pos[i-1,1])

            v_mag = self.v_mean + 0.3*np.sin(0.003*t[i]) + 0.4*np.random.randn()
            vel[i,0] = v_mag*np.cos(heading) + 0.4*wind[0]
            vel[i,1] = v_mag*np.sin(heading) + 0.4*wind[1]
            vel[i,2] = 0.05*np.dot(slope, vel[i,:2])

            pos[i] = pos[i-1] + vel[i]*self.dt
            pos[i,2] = terrain_z
            acc[i] = (vel[i]-vel[i-1])/self.dt
            gyro[i] = np.array([0,0,(heading - gyro[i-1,2])/self.dt])

        return t, pos, vel, acc, gyro


# ==============================================================
# Visualization
# ==============================================================
def plot_environment(env: Environment, pos: np.ndarray, t, title="3D Realistic Terrain with Temperature and Obstacles"):
    """Render terrain, trajectory, obstacles, and goals in 3D with temperature coloring."""
    xs = np.linspace(-env.area/2, env.area/2, 300)
    ys = np.linspace(-env.area/2, env.area/2, 300)
    X, Y = np.meshgrid(xs, ys)
    Z_raw = env.terrain_height(X, Y)
    TEMP = env.temperature(X, Y, t[-1])

    z_exaggeration = 3.5
    Z = z_exaggeration * Z_raw

    fig = go.Figure()

    # Terrain
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=TEMP,
        colorscale='thermal',
        opacity=0.98,
        colorbar=dict(title='Temperature [°C]'),
        lighting=dict(ambient=0.4, diffuse=1.0, specular=0.5, roughness=0.5),
        lightposition=dict(x=1000, y=1200, z=2500),
        name='Terrain'
    ))

    # Trajectory
    z_path = z_exaggeration * env.terrain_height(pos[:,0], pos[:,1]) + 4.0
    fig.add_trace(go.Scatter3d(
        x=pos[:,0], y=pos[:,1], z=z_path,
        mode='lines', line=dict(color='cyan', width=5),
        name='Robot Trajectory'
    ))

    # Obstacles (render as vertical pillars)
    for obs in env.obstacles:
        theta = np.linspace(0, 2*np.pi, 20)
        Xc = obs["x"] + obs["size"] * np.cos(theta)
        Yc = obs["y"] + obs["size"] * np.sin(theta)
        Zc_bottom = z_exaggeration * env.terrain_height(obs["x"], obs["y"])
        Zc_top = Zc_bottom + obs["height"]
        for z in np.linspace(Zc_bottom, Zc_top, 3):
            fig.add_trace(go.Scatter3d(
                x=Xc, y=Yc, z=np.full_like(Xc, z),
                mode='lines', line=dict(color='darkred', width=6), name=''
            ))

    # Goals
    gx, gy = env.goal_points[:,0], env.goal_points[:,1]
    gz = z_exaggeration * env.terrain_height(gx, gy) + 10.0
    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz,
        mode='markers+text',
        marker=dict(size=8, color='gold'),
        text=[f"G{i+1}" for i in range(len(gx))],
        textposition='top center',
        name='Goals'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X [m]'),
            yaxis=dict(title='Y [m]'),
            zaxis=dict(title='Elevation [m]', range=[Z.min()*1.2, Z.max()*1.2]),
            aspectratio=dict(x=1, y=1, z=0.45),
            camera=dict(eye=dict(x=1.6, y=1.8, z=1.2))
        ),
        title=title,
        height=950,
        template='plotly_dark',
        showlegend=False
    )

    fig.show()

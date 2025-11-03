import numpy as np


class IMUSimulator:
    """
    Generates synthetic IMU readings (accelerometer + gyroscope)
    with realistic error sources influenced by Environment conditions.
    """
    def __init__(self, env, dt=0.05):
        self.env = env
        self.dt = dt

        # ---- Accelerometer error model ----
        self.acc_bias = np.random.randn(3) * 0.02           # initial bias [m/s²]
        self.acc_phi  = np.exp(-dt / 50.0)                  # bias correlation time (~50 s)
        self.acc_sigma= 0.001                               # process noise for bias evolution
        self.acc_scale= np.diag(1.0 + 0.01*np.random.randn(3))
        self.acc_misal= np.eye(3) + 0.005*np.random.randn(3,3)

        # ---- Gyroscope error model ----
        self.gyro_bias = np.random.randn(3) * 0.002         # initial bias [rad/s]
        self.gyro_phi  = np.exp(-dt / 80.0)
        self.gyro_sigma= 0.0001
        self.gyro_scale= np.diag(1.0 + 0.01*np.random.randn(3))
        self.gyro_misal= np.eye(3) + 0.005*np.random.randn(3,3)

        # noise levels
        self.acc_noise_std  = 0.02
        self.gyro_noise_std = 0.002

    def step(self, a_true, g_true, t, pos):
        """Propagate biases & corrupt the true acceleration / gyro."""
        temp = self.env.temperature(pos[0], pos[1], t)
        temp_factor = 1 + 0.002*(temp - 20.0)

        # Bias random walk (Gauss–Markov)
        self.acc_bias  = (self.acc_phi * self.acc_bias +
                          np.sqrt(1-self.acc_phi**2)*self.acc_sigma*np.random.randn(3))
        self.gyro_bias = (self.gyro_phi * self.gyro_bias +
                          np.sqrt(1-self.gyro_phi**2)*self.gyro_sigma*np.random.randn(3))

        # Environment-dependent coupling (wind increases vibration)
        wind_mag = np.linalg.norm(self.env.wind(t))
        vib_factor = 1 + 0.1*wind_mag

        # Corrupt signals
        acc_meas  = (self.acc_scale @ self.acc_misal @ a_true
                     + self.acc_bias*temp_factor*vib_factor
                     + self.acc_noise_std*np.random.randn(3))
        gyro_meas = (self.gyro_scale @ self.gyro_misal @ g_true
                     + self.gyro_bias*temp_factor*vib_factor
                     + self.gyro_noise_std*np.random.randn(3))
        return acc_meas, gyro_meas, self.acc_bias.copy(), self.gyro_bias.copy()

def generate_imu_dataset(env, t, p_true, a_true, g_true, dt):
    """
    Runs the IMU simulator over the full trajectory and
    returns measured accel/gyro plus ground-truth biases.
    """
    imu = IMUSimulator(env, dt)
    acc_meas, gyro_meas, acc_biases, gyro_biases = [], [], [], []
    for i, ti in enumerate(t):
        a_m, g_m, a_b, g_b = imu.step(a_true[i], g_true[i], ti, p_true)
        acc_meas.append(a_m); gyro_meas.append(g_m)
        acc_biases.append(a_b); gyro_biases.append(g_b)
    return (np.array(acc_meas), np.array(gyro_meas),
            np.array(acc_biases), np.array(gyro_biases))


def build_supervised_dataset(acc_meas, gyro_meas, acc_bias_true, window=50):
    X, y = [], []
    for i in range(window, len(acc_meas)):
        window_data = np.hstack([acc_meas[i-window:i], gyro_meas[i-window:i]]).ravel()
        X.append(window_data)
        y.append(acc_bias_true[i])
    X, y = np.array(X), np.array(y)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -50, 50)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -5, 5)
    return X, y

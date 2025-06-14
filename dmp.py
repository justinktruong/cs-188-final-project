import numpy as np
from scipy import interpolate

class CanonicalSystem:
    """
    Skeleton of the discrete canonical dynamical system.
    """
    def __init__(self, dt: float, ax: float = 1.0):
        """
        Args:
            dt (float): Timestep duration.
            ax (float): Gain on the canonical decay.
        """
        # Initialize time parameters
        self.dt: float = dt
        self.ax: float = ax
        self.run_time: float = 1.0  # TODO: set total runtime
        self.timesteps: int = int(self.run_time / dt)  # TODO: compute from run_time and dt
        self.x: float = 1.0

    def reset(self) -> None:
        """
        Reset the phase variable to its initial value.
        """
        # TODO: implement
        self.x = 1.0

    def __step_once(self, x, tau, ec):
        """
        Euler update of ẋ = −aₓ x · ec  (scaled by tau).
        """
        return x + (-self.ax * x * ec) * tau * self.dt

    def step(self, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        """
        Advance the phase by one timestep.

        Returns:
            float: Updated phase value.
        """
        # TODO: implement update rule
        self.x = self.__step_once(self.x, tau, error_coupling)
        return self.x

    def rollout(self, tau: float = 1.0, ec: float = 1.0) -> np.ndarray:
        """
        Generate the entire phase sequence.

        Returns:
            np.ndarray: Array of phase values over time.
        """
        # TODO: call reset() then repeatedly call step()
        xs = np.zeros(self.timesteps)
        self.reset()
        for i in range(self.timesteps):
            xs[i] = self.step(tau, ec)
        return xs

class DMP:
    """
    Skeleton of the discrete Dynamic Motor Primitive.
    """
    def __init__(
        self,
        n_dmps: int,
        n_bfs: int,
        dt: float = 0.01,
        y0: float = 0.0,
        goal: float = 1.0,
        ay: float = 25.0,
        by: float = None
    ):
        """
        Args:
            n_dmps (int): Number of dimensions.
            n_bfs (int): Number of basis functions per dimension.
            dt (float): Timestep duration.
            y0 (float|array): Initial state.
            goal (float|array): Goal state.
            ay (float|array): Attractor gain.
            by (float|array): Damping gain.
        """
        # TODO: initialize parameters
        self.n_dmps: int = n_dmps
        self.n_bfs: int = n_bfs
        self.dt: float = dt

        self.y0: np.ndarray = np.full(n_dmps, y0, dtype=float)
        self.goal: np.ndarray = np.full(n_dmps, goal, dtype=float)
        self.ay: np.ndarray = np.full(n_dmps, ay, dtype=float)

        if by is None:
            self.by: np.ndarray = self.ay / 4.0
        else:
            self.by: np.ndarray = np.full(n_dmps, by)

        self.w: np.ndarray = np.zeros((n_dmps, n_bfs))
        self.cs: CanonicalSystem = CanonicalSystem(dt=dt)

        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset trajectories and canonical system state.
        """
        # TODO: reset y, dy, ddy and call self.cs.reset()
        self.y: np.ndarray = self.y0.copy()
        self.dy: np.ndarray = np.zeros(self.n_dmps)
        self.ddy: np.ndarray = np.zeros(self.n_dmps)
        self.cs.reset()

    def _psi(self, x):
        c = np.exp(-self.cs.ax * np.linspace(0, self.cs.run_time, self.n_bfs))
        h = np.ones(self.n_bfs) * self.n_bfs**1.5 / c / self.cs.ax
        return np.exp(-h * (x - c)**2)


    def imitate(self, y_des: np.ndarray) -> np.ndarray:
        """
        Learn DMP weights from a demonstration.

        Args:
            y_des (np.ndarray): Desired trajectory, shape (T, D).

        Returns:
            np.ndarray: Interpolated demonstration (T', D).
        """
        # TODO: interpolate, compute forcing term, solve for w
        trajectory = y_des.copy()
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)
        T, D = trajectory.shape

        t_demo = np.linspace(0, self.cs.run_time, T)
        t_canon = np.linspace(0, self.cs.run_time, self.cs.timesteps)
    
        # y_interp = np.zeros((self.cs.timesteps, D))
        # for d in range(min(D, self.n_dmps)):
        #     f = interpolate.interp1d(t_demo, y_des[:, d], kind='cubic', fill_value='extrapolate')
        #     y_interp[:, d] = f(t_canon)
        y_interp = np.zeros((self.cs.timesteps, self.n_dmps))
        for d in range(min(D, self.n_dmps)):
            y_interp[:, d] = np.interp(t_canon, t_demo, trajectory[:, d])


        # dy = np.gradient(y_interp, self.dt, axis=0)
        # ddy = np.gradient(dy, self.dt, axis=0)

        # self.y0 = y_interp[0].copy()
        # self.goal = y_interp[-1].copy()
        # f = ddy - self.ay * (self.by * (self.goal - y_interp) - dy)

        # x_track = self.cs.rollout()
        # psi = np.vstack([self._psi(x) for x in x_track])
        # psi_sum = psi.sum(axis=1)           
        # x = (psi * x_track[:,None]) / psi_sum[:,None]

        # for d in range(self.n_dmps):
        #     w_d, *_ = np.linalg.lstsq(x, f[:, d], rcond=None)
        #     self.w[d, :] = w_d

        # return y_interp.T

        derivatives = [y_interp]
        for i in range(2):
            derivatives.append(np.gradient(derivatives[-1], axis=0) / self.dt)

        y, dy, ddy = derivatives
        cs_phase = self.cs.rollout()

        weight_matrix = np.zeros((self.n_dmps, self.n_bfs))

        for d in range (self.n_dmps):
            force = ddy[:, d] - self.ay[d] * (self.by[d] * (self.goal[d] - y[:, d]) - dy[:, d])

            psi = np.array([self._psi(phase) for phase in cs_phase])
            weighted_basis = psi * (cs_phase[:, None] * (self.goal[d] - self.y0[d]))
            
            weight_matrix[d] = np.linalg.lstsq(weighted_basis, force, rcond=None)[0]

        self.w = weight_matrix
        return y_interp.T 

    def rollout(
        self,
        tau: float = 1.0,
        error: float = 1.0,
        new_goal: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate a new trajectory from the DMP.

        Args:
            tau (float): Temporal scaling.
            error (float): Feedback coupling.
            new_goal (np.ndarray, optional): Override goal.

        Returns:
            np.ndarray: Generated trajectory (T x D).
        """
        # TODO: implement dynamical update loop
        self.reset_state()

        goal = np.zeros(self.n_dmps)
        if new_goal is not None:
            for i in range(self.n_dmps):
                goal[i] = new_goal[i]
        else:
            for i in range(self.n_dmps):
                goal[i] = self.goal[i]

        if np.isscalar(goal):
            goal = np.ones(self.n_dmps) * goal
        
        # T = self.cs.timesteps
        # Y = np.zeros((T, self.n_dmps))

        # for t in range(T):
        #     x = self.cs.step(tau, error)
        #     psi = self._psi(x)
        #     psi_sum = psi.sum()               

        #     f = (psi * self.w).sum(axis=1) * x / psi_sum
        #     self.ddy = self.ay * (self.by * (self.goal - self.y) - self.dy) + f
        #     self.dy += self.ddy * tau * self.dt
        #     self.y += self.dy * tau * self.dt
        #     Y[t, :] = self.y.copy()

        # return Y

        trajectory_data = {
            'y': np.zeros((self.cs.timesteps, self.n_dmps)),
            'dy': np.zeros((self.cs.timesteps, self.n_dmps)),
            'dyy': np.zeros((self.cs.timesteps, self.n_dmps))
        }

        
        for i in range(self.cs.timesteps):
            curr_phase = self.cs.x

            force_list = []
            for d in range(self.n_dmps):
                psi = self._psi(curr_phase)
                val = curr_phase * (goal[d] - self.y0[d]) * np.dot(psi, self.w[d, :])
                force_list.append(val)
            force = np.array(force_list)

            self.ddy = (self.ay * (self.by * (goal - self.y) - self.dy * tau) + force) / (tau * tau)
            self.dy += self.ddy * self.dt
            self.y += self.dy * self.dt

            current_state = {'y': self.y.copy(), 'dy': self.dy.copy(), 'dyy': self.ddy.copy()}
            
            for key, data in zip(['y', 'dy', 'dyy'], [current_state['y'], current_state['dy'], current_state['dyy']]):
                trajectory_data[key][i] = data
            
            self.cs.step(tau, error)
        
        return trajectory_data['y']
       

# ==============================
# DMP Unit test
# ==============================
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Test canonical system
    cs = CanonicalSystem(dt=0.05)
    x_track = cs.rollout()
    plt.figure()
    plt.plot(x_track, label='Canonical x')
    plt.title('Canonical System Rollout')
    plt.xlabel('Timestep')
    plt.ylabel('x')
    plt.legend()

    # Test DMP behavior with a sine-wave trajectory
    dt = 0.01
    T = 1.0
    t = np.arange(0, T, dt)
    y_des = np.sin(2 * np.pi * 2 * t)

    dmp = DMP(n_dmps=1, n_bfs=50, dt=dt)
    y_interp = dmp.imitate(y_des)
    y_run = dmp.rollout()

    plt.figure()
    plt.plot(t, y_des, 'k--', label='Original')
    plt.plot(np.linspace(0, T, y_interp.shape[1]), y_interp.flatten(), 'b-.', label='Interpolated')
    plt.plot(np.linspace(0, T, y_run.shape[0]), y_run.flatten(), 'r-', label='DMP Rollout')
    plt.title('DMP Imitation and Rollout')
    plt.xlabel('Time (s)')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()

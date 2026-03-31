import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation

class CompassWalkerPlant:
    def __init__(self, hip_mass=1.0, mass=1.0, a=0.5, b=0.3, gamma=0.5, gravity=9.81, matrices = [None, None, None]):
        # Physical parameters
        self.m_h = hip_mass
        self.m = mass
        self.a = a 
        self.b = b 
        self.l = a + b
        self.gamma = gamma
        self.g = gravity
        self.M, self.C, self.G = matrices

        self.in_contact = False
        # Generalized coordinates
        self.dof = 4
        self.x = np.zeros(2 * self.dof)  # [q, qd]
        self.t = 0.0  # time

        # Logging
        self.t_values = []
        self.x_values = []
        self.contact_values = []
        self.phase_portrait_values = []

        # Flag for keeping track off leg-swaps
        self.leg_swapped = False

        self.contactpoint = np.array([0,0])


    def euler_integrator(self, t, x, dt):
        dx = self.f(t, x)
        return x + dt * dx

    def runge_integrator(self, t, x, dt):
        k1 = self.f(t, x)
        k2 = self.f(t + 0.5*dt, x + 0.5*dt*k1)
        k3 = self.f(t + 0.5*dt, x + 0.5*dt*k2)
        k4 = self.f(t + dt, x + dt*k3)
        return x + dt * (k1 + 2*(k2 + k3) + k4) / 6.0

    def set_state(self, time, x):
        self.x = x
        self.t = time

    def get_state(self):
        return self.t, self.x
        
    def forward_kinematics(self, q):
        th_st, th_sw = q

        x, y = self.contactpoint
        # Stance foot position
        st_foot_pos = np.array([x, y])

        # Stance leg mass position
        st_m_pos = np.array([
            x + self.a * np.sin(th_st),
            y + self.a * np.cos(th_st)])

        # Hip position based on stance foot position (x, y)
        hip_pos = np.array([
            x + self.l * np.sin(th_st),
            y + self.l * np.cos(th_st)
        ])

        # Swing foot position
        sw_foot_pos = np.array([
            x + self.l * np.sin(th_st) - self.l * np.sin(th_sw),
            y + self.l * np.cos(th_st) - self.l * np.cos(th_sw)
        ])

        # Swing mass position
        sw_m_pos = np.array([
            hip_pos[0] - self.b * np.sin(th_sw),
            hip_pos[1] - self.b * np.cos(th_sw)
        ])

        return st_foot_pos, st_m_pos, hip_pos, sw_foot_pos, sw_m_pos

    def simulate(self, t0, x0, tf, dt, controller=None, integrator="euler"):
        self.set_state(t0, x0)
        self.contactpoint = np.array([0, 0])

        self.t_values = []
        self.x_values = []
        self.phase_portrait_values = []
        self.contact_values = []

        
        while(self.t <= tf):
            self.step(dt, integrator=integrator)

        return self.t_values, self.x_values

    def animate_from_data(self, Tsim, Xsim, skip = 1, fps = 30, save_video=False):
        """
        Animate a compass walker using precomputed states.
        
        Parameters:
            Tsim : array-like
                Time vector
            Xsim : array-like
                State vector at each time step
            save_video : bool
                Whether to save the animation as a video
        """
        self.t_values = Tsim
        self.x_values = np.array(Xsim)[::skip]
        self.phase_portrait_values = np.array(Xsim)[::skip, :2]  # Assuming theta and theta_dot are first two states
    
        fig, (self.animation_ax, self.ps_ax) = plt.subplots(1, 2, figsize=(10, 5))
        self.animation_plots = []
    
        # Hip plot
        hip_plot, = self.animation_ax.plot([], [], "o", markersize=10, color="blue", zorder=5)
        # Ground
        ground_plot, = self.animation_ax.plot([], [], "-", lw=1, color="black")
        # Legs
        stance_plot, = self.animation_ax.plot([], [], "-", lw=2, color="black")
        swing_plot, = self.animation_ax.plot([], [], "-", lw=2, color="black")
        # Leg masses
        stance_mass_plot, = self.animation_ax.plot([], [], "o", markersize=10, color="red", zorder=6)
        swing_mass_plot, = self.animation_ax.plot([], [], "o", markersize=10, color="green", zorder=6)
    
        self.animation_plots.extend([stance_plot, swing_plot, hip_plot, ground_plot, stance_mass_plot, swing_mass_plot])
    
        # Phase space plot
        ps_plot, = self.ps_ax.plot([], [], "-", lw=1.0, color="blue")
        self.ps_ax.set_xlim(-1.0, 1.0)
        self.ps_ax.set_ylim(-1.0, 1.0)
        self.animation_plots.append(ps_plot)
    
        num_steps = len(Tsim)
        frames = range(num_steps)
    
        animation = FuncAnimation(
            fig, self._animation_step, frames=frames,
            init_func=self._animation_init, blit=True,
            repeat=False, interval=(Tsim[1]-Tsim[0])*1000
        )
    
        if save_video:
            writer = mplanimation.writers['ffmpeg'](fps=fps, bitrate=1800)
            animation.save('compass_walker.mp4', writer=writer)
    
        return animation

    def _animation_init(self):
        self.animation_ax.set_xlim(-1, 2)
        self.animation_ax.set_ylim(-1.5, 1.5)
        self.animation_ax.set_xlabel("x position [m]")
        self.animation_ax.set_ylabel("y position [m]")

        for plot in self.animation_plots:
            plot.set_data([], [])

        self._ps_init()
        return self.animation_plots

    def _animation_step(self, i):
        x = np.array(self.x_values)[i, :2]  # first two states for kinematics
        self.contactpoint = self.contact_values[i]
        st_foot_pos, st_m_pos, hip_pos, sw_foot_pos, sw_m_pos = self.forward_kinematics(x)
        # Update leg positions
        self.animation_plots[0].set_data([hip_pos[0], st_foot_pos[0]], [hip_pos[1], st_foot_pos[1]])
        self.animation_plots[1].set_data([hip_pos[0], sw_foot_pos[0]], [hip_pos[1], sw_foot_pos[1]])
        self.animation_plots[2].set_data([hip_pos[0]], [hip_pos[1]])  # Hip position

        # Update mass positions
        self.animation_plots[4].set_data([st_m_pos[0]], [st_m_pos[1]])
        self.animation_plots[5].set_data([sw_m_pos[0]], [sw_m_pos[1]])

        # Update ground line
        self.animation_plots[3].set_data([-1, 10], [np.tan(self.gamma), -10 * np.tan(self.gamma)])

        # Update phase space
        self._ps_update(i)

        return self.animation_plots

    def _ps_init(self):
        """Initialize phase space plot"""
        self.animation_ax.set_xlim(-1, 2)
        self.animation_ax.set_ylim(-1.5, 1.5)
        self.ps_ax.set_xlabel("theta [rad]")
        self.ps_ax.set_ylabel("theta_dot [rad/s]")
        self.animation_plots[-1].set_data([], [])
        return self.animation_plots
    
    def _ps_update(self, i):
        """Update phase space plot up to current frame i"""
        if hasattr(self, 'phase_portrait_values') and len(self.phase_portrait_values) > 0:
            data = np.array(self.phase_portrait_values[:i+1])
            theta, theta_dot = data.T
            self.animation_plots[-1].set_data(theta, theta_dot)
        return self.animation_plots

def plot_timeseries_compasswalker(T, X, U=None, x_cent=None):
    plt.plot(T, np.asarray(X).T[0], label="th_st")
    plt.plot(T, np.asarray(X).T[1], label="th_st_dot")
    plt.plot(T, np.asarray(X).T[2], label="th_sw")
    plt.plot(T, np.asarray(X).T[3], label="th_sw_dot")
    if U is not None:
        plt.plot(T, U, label="u")
    if x_cent is not None:
        plt.plot(T, np.asarray(x_cent).T[0], label="m_h x-pos")
    plt.legend(loc="best")
    plt.show()
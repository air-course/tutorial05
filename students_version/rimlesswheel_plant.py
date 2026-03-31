import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation

class RimlessWheelPlant:
    def __init__(self, mass=1.0, leglength=0.5, gravity = 9.81, gamma = np.pi/12, nlegs = 8):
        self.m = mass
        self.l = leglength
        self.g = gravity
        self.gamma = gamma
        self.nlegs = nlegs
        self.alpha = np.pi/nlegs
        self.torque_limit = np.inf

        self.dof = 1
        self.x = np.zeros(2*self.dof) #position, velocity
        self.contactpoint = [0,0]
        self.t = 0.0 #time

        self.t_values = []
        self.x_values = []
        self.x_cent_values = []
        self.tau_values = []

    def euler_integrator(self, t, x, dt, tau):
        """
        Implement Forward Euler Integration for a time-step dt and state y
        y = [pos, vel]
        """
        integ = self.f(t, x, tau)
        x_new = x + dt*integ
        return x_new

    def runge_integrator(self, t, x, dt, tau):
        """
        Implement a fourth order Runge-Kutta Integration scheme
        """

        k1 = self.f(t, x, tau)
        k2 = self.f(t + 0.5*dt, x + 0.5*dt*k1, tau)
        k3 = self.f(t + 0.5*dt, x + 0.5*dt*k2, tau)
        k4 = self.f(t + dt, x + dt*k3, tau)
        integ = (k1 + 2*(k2 + k3) + k4) / 6.0
    
        x_new = x + dt*integ
    
        return x_new

    def set_state(self, time, x):
        self.x = x
        self.t = time

    def get_state(self):
        return self.t, self.x

    def forward_kinematics(self, pos):
        """
        forward kinematics, origin at fixed point
        """

        pos = (pos+self.alpha-self.gamma)%(2*self.alpha) + self.gamma - self.alpha 

        ee_pos = np.zeros([self.nlegs,2])

        center = np.array([self.contactpoint[0] + self.l*np.sin(pos), self.contactpoint[1] + self.l*np.cos(self.x[0])])

        for leg in range(self.nlegs):
            th_i = pos + leg*2*self.alpha
            ee_pos[leg] = [center[0] - self.l * np.sin(th_i), center[1] - self.l*np.cos(th_i)]

        return ee_pos, center

    def simulate(self, t0, x0, tf, dt, controller=None, integrator="euler"):
        self.set_state(t0, x0)

        self.t_values = []
        self.x_values = []
        self.tau_values = []
        self.x_cent_values = []
        self.contactpoint = [0,0]

        
        while(self.t <= tf):
            if controller is not None:
                tau = controller.get_control_output(self.x)
            else:
                tau = 0
            self.step(tau, dt, integrator=integrator)

        return self.t_values, self.x_values, self.tau_values, self.x_cent_values

    def simulate_and_animate(self, t0, x0, tf, dt, controller=None, integrator="euler", save_video=False):
        """
        simulate and animate the pendulum
        """
        self.set_state(t0, x0)

        self.t_values = []
        self.x_values = []
        self.tau_values = []
        self.x_cent_values = []
        self.contactpoint = [0,0]

        #fig = plt.figure(figsize=(6,6))
        #self.animation_ax = plt.axes()
        fig, (self.animation_ax, self.ps_ax) = plt.subplots(1, 2, figsize=(10, 5))
        self.animation_plots = []
        ee_plot, = self.animation_ax.plot([], [], "o", markersize=5.0, color="blue", zorder = self.nlegs+5)
        ground_plot, = self.animation_ax.plot([], [], "-", lw=1, color="black")

        for leg in range(self.nlegs):
            bar_plot, = self.animation_ax.plot([], [], "-", lw=2, color="black")
            self.animation_plots.append(bar_plot)

        #text_plot = self.animation_ax.text(0.1, 0.1, [], xycoords="figure fraction")
        self.animation_plots.append(ee_plot)
        self.animation_plots.append(ground_plot)

        num_steps = int(tf / dt/10)
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["controller"] = controller
        par_dict["integrator"] = integrator
        frames = num_steps*[par_dict]

        #ps_fig = plt.figure(figsize=(6,6))
        #self.ps_ax = plt.axes()
        #self.ps_plots = []
        ps_plot, = self.ps_ax.plot([], [], "-", lw=1.0, color="blue")
        #self.ps_plots.append(ps_plot)
        self.animation_plots.append(ps_plot)

        animation = FuncAnimation(fig, self._animation_step, frames=frames, init_func=self._animation_init, blit=True, repeat=False, interval=dt*1000*10)
        animation2 = None
        #if phase_plot:
        #    animation2 = FuncAnimation(fig, self._ps_update, init_func=self._ps_init, blit=True, repeat=False, interval=dt*1000)

        if save_video:
            Writer = mplanimation.writers['ffmpeg']
            writer = Writer(fps=60, bitrate=1800)
            animation.save('pendulum_swingup.mp4', writer=writer)
            #if phase_plot:
            #    Writer2 = mplanimation.writers['ffmpeg']
            #    writer2 = Writer2(fps=60, bitrate=1800)
            #    animation2.save('pendulum_swingup_phase.mp4', writer=writer2)
        #plt.show()

        return self.t_values, self.x_values, self.tau_values, animation, self.x_cent_values#, animation2

    def _animation_init(self):
        """
        init of the animation plot
        """

        self.animation_ax.set_xlim(-0.5, 1.5)
        self.animation_ax.set_ylim(-1, 1.32)

        self.animation_ax.set_xlabel("x position [m]")
        self.animation_ax.set_ylabel("y position [m]")
        for ap in self.animation_plots:
            ap.set_data([], [])

        self._ps_init()
        return self.animation_plots

    def _animation_step(self, par_dict):
        """
        simulate 10 steps and update the animation plot
        """

        dt = par_dict["dt"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        for i in range(10):
            if controller is not None:
                tau = controller.get_control_output(self.x)
            else:
                tau = 0
            self.step(tau, dt, integrator=integrator)

        ee_pos, center = self.forward_kinematics(self.x[0])
        for leg in range(self.nlegs):
            self.animation_plots[leg].set_data([center[0], ee_pos[leg,0]], [center[1], ee_pos[leg, 1]])

        self.animation_plots[self.nlegs].set_data((center[0],), (center[1],)) ## Debug
        #self.animation_plots[self.nlegs+1].set_data((-1.5*self.l,1.5*self.l), 
        #(-self.l + 1.5*self.l*np.tan(self.gamma), -self.l - 1.5*self.l*np.tan(self.gamma)))
        #self.animation_plots[self.nlegs+1].set_data((-2.5*self.l, 7.5*self.l), (2.5*self.l*np.tan(self.gamma), -7.5*self.l*np.tan(self.gamma)))
        self.animation_plots[self.nlegs+1].set_data((-1, 2.5), (np.tan(self.gamma), -2.5*np.tan(self.gamma)))
        self._ps_update(0)

        return self.animation_plots

    def _ps_init(self):
        """
        init of the phase space animation plot
        """
        self.ps_ax.set_xlim(-np.pi/2, np.pi/2)
        self.ps_ax.set_ylim(-2.5, 7.5)
        self.ps_ax.set_xlabel("theta [rad]")
        self.ps_ax.set_ylabel("velocity [rad/s]")
        for ap in self.animation_plots:
            ap.set_data([], [])
        return self.animation_plots

    def _ps_update(self, i):
        """
        update of the phase space animation plot
        """
        self.animation_plots[-1].set_data(np.asarray(self.x_values).T[0], np.asarray(self.x_values).T[1])
        return self.animation_plots

def plot_timeseries_RW(T, X, U=None, x_cent = None):
    plt.plot(T, np.asarray(X).T[0], label="theta")
    plt.plot(T, np.asarray(X).T[1], label="theta dot")
    if U is not None:
        plt.plot(T, U, label="u")
    if x_cent is not None:
        plt.plot(T, np.asarray(x_cent).T[0], label="x")
    plt.legend(loc="best")
    plt.show()
import numpy as np
from collections import defaultdict
from dmp import DMP
from pid import PID
from load_data import reconstruct_from_npz
from nearest_neighbor import nearest_neighbor
import random

class DMPPolicyWithPID:
    """
    A policy that follows a demonstrated path with DMPs and PID control.

    The demonstration is split into segments based on grasp toggles.  
    The first segment's endpoint is re-targeted to a new object pose.
    Subsequent segments replay the original DMP rollouts.

    Args:
        square_obs (dict): 'SquareNut_pos' observed
        demo_path (str): path to .npz file with demo data.
        dt (float): control timestep.
        n_bfs (int): number of basis functions per DMP.
    """
    def __init__(self, square_pos, num_demo, demo_path='demos.npz', dt=0.01, n_bfs=50):
        # demo = demos['demo_98']
        self.dt = dt
        new_obj_pos = square_pos

        demos = reconstruct_from_npz(demo_path) 

        demo_keys = list(demos.keys())
        selected_demos = random.sample(demo_keys, num_demo)

        neighbor = nearest_neighbor(demos, selected_demos, new_obj_pos)
        nearest_demo = demos[neighbor]
        demo_obj_pos = nearest_demo['obs_object'][0, :3]
        
        print(f"Nearest demo: {neighbor}")
        print(f"Demo pos: {demo_obj_pos}")
        print(f"Obs pos: {new_obj_pos}")

        ee_pos = nearest_demo['obs_robot0_eef_pos']  # (T,3)
        ee_grasp = nearest_demo['actions'][:, -1:].astype(int)  # (T,1)
        self.segments = self.detect_grasp_segments(ee_grasp)
        print(f"Segments: {self.segments}")

        start, end = self.segments[0]
        offset = ee_pos[end-1] - demo_obj_pos

        self.dmps = []
        self.segment_rollouts = []
        self.segment_grasps = []

        self.drop_off_pos = []
        if(offset[1] > 0):
            self.drop_off_pos = [0.24,  0.12,  0.95]
            offset[0] += 0.002
            offset[1] += 0.0
            offset[2] += -0.02
        else:
            self.drop_off_pos = [0.22,  0.10,  0.95]
            offset[0] += 0.02
            offset[1] += 0.02
            offset[2] += -0.02

        for i, (start, end) in enumerate(self.segments):
            trajectotry = ee_pos[start:end]
            grasp_state = ee_grasp[start, 0]
            self.segment_grasps.append(grasp_state )

            dmp = DMP(n_dmps=3, n_bfs=n_bfs, dt=dt, y0=trajectotry[0], goal=trajectotry[-1])
            dmp.imitate(trajectotry)
            self.dmps.append(dmp)
            
            if i == 0:
                new_goal = new_obj_pos + offset
                rollout = dmp.rollout(new_goal=new_goal)
            elif i == 1:
                rollout = dmp.rollout(new_goal=self.drop_off_pos)
            else: 
                rollout = dmp.rollout()

            self.segment_rollouts.append(rollout)

        initial_target = np.zeros(3) 
        self.pid = PID(kp=[10.0, 10.0, 10.0], ki=[0.1, 0.1, 0.1], kd=[1.0, 1.0, 1.0], target=initial_target)

        self.current_segment = 0
        self.step = 0
        self.shake_step = 0
        self.shake_position = None



    def detect_grasp_segments(self, grasp_flags: np.ndarray) -> list:
        """
        Identify segments based on grasp toggles.

        Args:
            grasp_flags (np.ndarray): (T,1) array of grasp signals.

        Returns:
            List[Tuple[int,int]]: start and end indices per segment.
        """
        # TODO: implement boundary detection
        flags = grasp_flags.flatten()
        segments = []

        if flags.size == 0:
            return segments
        
        start = 0
        current = flags[0]
        for i in range(1, flags.size):
            if flags[i] != current:
                segments.append((start, i-1))
                start = i
                current = flags[i] 
        segments.append((start, flags.size))
        return segments




    def get_action(self, robot_eef_pos: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos (np.ndarray): Current end-effector position [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,0,0,0,grasp].
        """
        # TODO: select current segment and step
        if self.current_segment >= len(self.segments):
            self.shake_position = self.drop_off_pos.copy()
            
            wiggle_x = 0.04 * np.sin(0.15 * np.pi * self.shake_step)
            wiggle_y = 0.02 * np.cos(0.15 * np.pi * self.shake_step)
            z_movement = [-0.05, 0.05]
            wiggle_z = z_movement[self.shake_step % 2]

            target_pos = self.shake_position + np.array([wiggle_x, wiggle_y, 0])
            self.pid.target = target_pos
            delta_pos = self.pid.update(robot_eef_pos, dt=self.dt)
            action = np.array([delta_pos[0], delta_pos[1], -0.04, 0.0, 0.0, 0.0, 0.0])
            self.shake_step += 1
            return action


        
        rollout = self.segment_rollouts[self.current_segment]

        if self.step >= len(rollout):
            self.step = 0
            self.current_segment += 1
            self.pid.reset()

            if self.current_segment >= len(self.segments):
                return np.zeros(7)

            rollout = self.segment_rollouts[self.current_segment]

        desired_pos = rollout[self.step]

        # TODO: compute PID-based delta_pos
        self.pid.target = desired_pos
        delta = self.pid.update(robot_eef_pos, dt=self.dt)

        # TODO: assemble action (zero rotation + grasp)
        action = np.zeros(7)
        action[0:3] = delta
        action[6] = self.segment_grasps[self.current_segment]

        self.step += 1
        return action
'''

DO NOT MODIFY THIS FILE

'''

import numpy as np
import robosuite as suite
from dmp_policy import DMPPolicyWithPID
from robosuite.utils.placement_samplers import UniformRandomSampler

# You can just remove the placement_initializer in CA 3 test script to use the default task environment:
# create environment instance

env = suite.make(
    env_name="NutAssemblySquare", 
    robots="Panda", 
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True,
    camera_names=["agentview"],
)

# placement_initializer = UniformRandomSampler(
#     name="FixedOriSampler",
#     mujoco_objects=None,            
#     x_range=[-0.115, -0.11],       
#     y_range=[0.05, 0.225],
#     rotation=np.pi,
#     rotation_axis="z",
#     ensure_object_boundary_in_range=False,
#     ensure_valid_placement=False,
#     reference_pos=(0,0,0.82),
#     z_offset=0.02,
# )

# # create environment instance
# env = suite.make(
#     env_name="NutAssemblySquare", 
#     robots="Panda", 
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     placement_initializer=placement_initializer,
#     ignore_done=True  
# )

success_rate = 0
# reset the environment
for _ in range(10):
    obs = env.reset()
    policy = DMPPolicyWithPID(obs['SquareNut_pos'], 200) 
    for _ in range(750):
        action = policy.get_action(obs['robot0_eef_pos'])
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        if reward == 1.0:
            success_rate += 1
            break

success_rate /= 10.0
print('success rate:', success_rate)

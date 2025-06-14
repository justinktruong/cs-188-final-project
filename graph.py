import numpy as np
import robosuite as suite
from dmp_policy import DMPPolicyWithPID
from robosuite.utils.placement_samplers import UniformRandomSampler
import matplotlib.pyplot as plt

env = suite.make(
    env_name="NutAssemblySquare", 
    robots="Panda", 
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True,
    camera_names=["agentview"],
)

runs = 15

success_rate_10 = 0.0

for _ in range(runs):
    obs = env.reset()
    policy = DMPPolicyWithPID(obs['SquareNut_pos'], 10) 
    for _ in range(750):
        action = policy.get_action(obs['robot0_eef_pos'])
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        if reward == 1.0:
            success_rate_10 += 1
            break

success_rate_10 /= runs


success_rate_50 = 0.0
for _ in range(runs):
    obs = env.reset()
    policy = DMPPolicyWithPID(obs['SquareNut_pos'], 50) 
    for _ in range(750):
        action = policy.get_action(obs['robot0_eef_pos'])
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        if reward == 1.0:
            success_rate_50 += 1
            break

success_rate_50 /= runs



success_rate_100 = 0.0
for _ in range(runs):
    obs = env.reset()
    policy = DMPPolicyWithPID(obs['SquareNut_pos'], 100) 
    for _ in range(750):
        action = policy.get_action(obs['robot0_eef_pos'])
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        if reward == 1.0:
            success_rate_100 += 1
            break

success_rate_100 /= runs


success_rate_200 = 0.0
for _ in range(runs):
    obs = env.reset()
    policy = DMPPolicyWithPID(obs['SquareNut_pos'], 200) 
    for _ in range(750):
        action = policy.get_action(obs['robot0_eef_pos'])
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        if reward == 1.0:
            success_rate_200 += 1
            break

success_rate_200 /= runs

demo_counts = [10, 50, 100, 200]
success_rates = [success_rate_10, success_rate_50, success_rate_100, success_rate_200]

plt.figure(figsize=(8, 5))
plt.plot(demo_counts, success_rates, marker='o', linestyle='-', color='blue', linewidth=2)
plt.title("Success Rate vs Number of Random Demos Used")
plt.xlabel("Number of Random Demos Sampled")
plt.ylabel("Success Rate")
plt.xticks(demo_counts)
plt.ylim(0, 1.1)
plt.grid(True)
plt.tight_layout()
plt.show()

print(success_rates)
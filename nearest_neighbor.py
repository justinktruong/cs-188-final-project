import numpy as np
from collections import defaultdict
from load_data import reconstruct_from_npz

def nearest_neighbor(demos, selected_demos, env_obj_pos):

    nearest = "demo_0"
    distance = np.inf

    for i in selected_demos:
        demo = demos[i]
        demo_pos = demo['obs_robot0_eef_pos']
        demo_obj_pos = min(demo_pos, key=lambda x: x[0])  
        if demo_obj_pos[1] > env_obj_pos[1]:
            cur_dist = np.linalg.norm(demo_obj_pos - env_obj_pos) + 1 
        else:
            cur_dist = np.linalg.norm(demo_obj_pos - env_obj_pos)

        if cur_dist < distance: 
            distance = cur_dist
            nearest = i
        
    return nearest


# def find_nearest_orientation(demos, env_obj_quat):
#     nearest = ""
#     distance = np.inf

#     for i in demos:
#         demo = demos[i]
#         demo_obj_quat = demo['obs_object'][0, 3:7]
#         cur_dist = quat_distance(demo_obj_quat, env_obj_quat)

#         if cur_dist < distance: 
#             distance = cur_dist
#             nearest = i
        
#     return nearest


# def quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
#     """
#     Geodesic distance between two unit quaternions:
#       d = 2 * arccos(|q1·q2|)
#     This is the smallest angle one must rotate to go from q1→q2.
#     """
#     dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
#     return 2.0 * np.arccos(abs(dot))


if __name__ == '__main__':
    demos = reconstruct_from_npz("demos.npz")
    # raw = np.load("demos.npz")
    # demos = defaultdict(dict)
    # for key in raw.files:
    #     prefix, trial, field = key.split('_', 2)
    #     demos[f"{prefix}_{trial}"][field] = raw[key]

    obj = np.array([-0.29,  0.19,  0.89])
    # quat = np.array([0.0, 0.0, 0.78252826, 0.62261507])

    nearest = nearest_neighbor(demos, obj)
    # nearest_quat = find_nearest_orientation(demos, quat)
    print(nearest)
    # print(nearest_quat)


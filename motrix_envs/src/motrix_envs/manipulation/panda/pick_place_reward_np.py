# motrix_envs/src/motrix_envs/manipulation/panda/pick_place_reward_np.py

from .compute_reach_grasp import _compute_reward_reach_grasp
from .compute_reach_grasp_transport import _compute_reward_reach_grasp_transport


def compute_reward_and_termination(env, data, info):
    """
    根据 env.cfg.task_mode 分发：
      - "reach_grasp"             → _compute_reward_reach_grasp
      - "reach_grasp_transport"   → _compute_reward_reach_grasp_transport
    """
    mode = getattr(env.cfg, "task_mode", "reach_only")
    if mode == "reach_grasp_transport":
        return _compute_reward_reach_grasp_transport(env, data, info)
    return _compute_reward_reach_grasp(env, data, info)
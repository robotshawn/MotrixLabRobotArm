# motrix_envs/src/motrix_envs/manipulation/panda/pick_place_reward_np.py

# reach_only 的文件名你这边可能出现过 compute_reach_only / compte_reach_only 两种写法
# 为了训练前“必跑通”，这里做兼容导入
try:
    from .compute_reach_only import _compute_reward_reach_only
except ImportError:
    from .compte_reach_only import _compute_reward_reach_only

from .compute_reach_grasp import _compute_reward_reach_grasp
from .compute_grasp_lift import _compute_reward_grasp_lift
from .compute_with_phases import _compute_reward_with_phases
from .compute_reach_grasp_transport import _compute_reward_reach_grasp_transport


def compute_reward_and_termination(env, data, info):
    """
    根据 env.cfg.task_mode 分发：
      - "reach_only"              → _compute_reward_reach_only
      - "reach_grasp"             → _compute_reward_reach_grasp
      - "grasp_lift"              → _compute_reward_grasp_lift
      - "reach_grasp_transport"   → _compute_reward_reach_grasp_transport
    """
    mode = getattr(env.cfg, "task_mode", "reach_only")
    if mode == "reach_only":
        return _compute_reward_reach_only(env, data, info)
    if mode == "reach_grasp":
        return _compute_reward_reach_grasp(env, data, info)
    if mode == "grasp_lift":
        return _compute_reward_grasp_lift(env, data, info)
    if mode == "reach_grasp_transport":
        return _compute_reward_reach_grasp_transport(env, data, info)

    # 默认为完整 pick & place（原来的 phase-based 奖励）
    return _compute_reward_with_phases(env, data, info, mode)

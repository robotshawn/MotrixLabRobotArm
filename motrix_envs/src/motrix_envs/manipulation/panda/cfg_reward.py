# motrix_envs/src/motrix_envs/manipulation/panda/cfg_reward.py
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    # ==================== 通用 scale ====================
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "reach": 3.0,
            "grasp": 4.0,
            "lift": 6.0,
            "transport": 3.0,
            "place": 4.0,
            "success": 40.0,
            "action_penalty": -1e-4,
        }
    )

    reach_grasp_curriculum_alpha: float = 0.7

    # ==================== 约束/惩罚通用 ====================
    max_step_reward: float = 200.0
    invalid_penalty: float = -30.0
    time_penalty: float = -0.02
    drop_penalty: float = -10.0
    success_bonus: float = 40.0

    # -------------------- 几何相关 --------------------
    lift_height: float = 0.06
    success_tolerance: float = 0.03

    reach_radius: float = 0.15
    orient_scale: float = 0.3
    grasp_height_delta: float = 0.015
    grasp_dist_tol: float = 0.08
    max_reach_distance: float = 0.35
    contact_force_grasp: float = 5.0

    hover_height: float = 0.06
    approach_sigma: float = 0.25
    grasp_radius: float = 0.05

    approach_scale: float = 0.2
    approach_fadeout_steps: int = 300

    grasp_bonus: float = 20.0
    hold_bonus: float = 4.0

    # ==================== 受力与力平衡（旧字段保留） ====================
    force_balance_reward_scale: float = 0.5
    grasp_force_scale: float = 5.0
    grasp_force_max: float = 4.0

    # ==================== Stage1: reach_grasp ====================
    # 目标：稳定抓取 N 步（严格条件），并在抓稳后鼓励“真 lift”
    stage1_hold_steps: int = 10

    # <=0 则 compute_reach_grasp 用 0.3*max_step_reward 做兜底
    stage1_success_reward: float = -1.0

    # build：抓稳过程中逐步加一点（k步给 k*step_reward）
    stage1_grasp_step_reward: float = 0.10

    # maintain：抓稳后每步给小奖励
    stage1_grasp_maintain_reward: float = 0.02

    # 单指轻触（小）
    stage1_single_contact_reward: float = 0.01

    # 距离引导（小）
    stage1_pregrasp_reward_scale: float = 0.05
    stage1_pregrasp_dist: float = 0.06
    stage1_pregrasp_margin: float = 0.25

    # delta_dist shaping（削弱 + clip）
    stage1_approach_reward_scale: float = 0.10
    stage1_away_penalty_scale: float = 0.10
    stage1_delta_dist_clip: float = 0.01

    # 关爪但没抓住的惩罚（近距离才罚）
    stage1_closed_no_grasp_penalty: float = 1.0

    # 长时间没接触/没抓住（温和）
    stage1_long_no_contact_penalty: float = 0.05
    stage1_long_no_grasp_penalty: float = 0.025
    stage1_no_contact_steps: int = 80
    stage1_no_grasp_steps: int = 150

    # 原点附近偏好（bonus）
    stage1_origin_tol: float = 0.02
    stage1_origin_bonus_scale: float = 0.02

    # 严格抓取条件（不松动）
    stage1_force_threshold: float = 1.5
    stage1_single_force_threshold: float = 1.5
    stage1_height_threshold: float = 0.02
    stage1_single_height_threshold: float = 0.02
    stage1_grasp_dist_threshold: float = 0.03
    stage1_single_dist_threshold: float = 0.05

    # gap（两指间距）落在“夹住 cube”的合理区间
    stage1_grasp_gap_min: float = 0.018
    stage1_grasp_gap_max: float = 0.031

    # 软约束：超力惩罚（很小）
    stage1_force_soft_threshold: float = 30.0
    stage1_force_penalty_scale: float = 1e-4

    # push 惩罚（未抓稳时 cube 位移增长就扣一点点）
    stage1_push_penalty_scale: float = 0.02
    stage1_push_penalty_scale_lifted: float = 0.005

    # ========== A：pinch 几何（解决“撬/托局部最优”的关键） ==========
    # 是否把 pinch_ok 作为 stable_grasp 的必要条件（强烈建议 True）
    stage1_pinch_require_for_grasp: bool = True

    # pinch_reward：鼓励 cube 中心落在两指连线段中部、且到连线的正交距离小
    stage1_pinch_reward_scale: float = 0.30
    stage1_pinch_proj_min: float = 0.15     # cube center 投影必须落在连线段内（0~1 的范围）
    stage1_pinch_proj_max: float = 0.85
    stage1_pinch_center_tol: float = 0.15   # |proj-0.5| <= tol 给高分
    stage1_pinch_center_margin: float = 0.35
    stage1_pinch_orth_thresh: float = 0.018 # cube center 到两指连线的正交距离阈值
    stage1_pinch_orth_margin: float = 0.05

    # pry_penalty：未 pinch 却把 box 抬高/抬升增量明显 -> 扣分（直接杀掉“撬起来”）
    stage1_pry_penalty_scale: float = 2.0
    stage1_pry_height_start: float = 0.015
    stage1_pry_delta_height_scale: float = 40.0
    stage1_pry_force_gate: float = 2.0  # max_force 超过这个值且没 pinch 时才开始罚（避免误伤）

    # 可选：抓稳后受力对称奖励（默认关；如你要更稳可以开一点）
    stage1_force_balance_scale: float = 0.0

    # ========== stage1 内的 lift 子目标（为 stage2 铺垫） ==========
    stage1_lift_start_height: float = 0.01
    stage1_lift_goal_height: float = 0.05
    stage1_lift_hold_steps: int = 5
    stage1_lift_height_reward_scale: float = 3.0
    stage1_lift_delta_reward_scale: float = 30.0
    stage1_lift_success_reward: float = 120.0  # 默认 0.6*200

    stage1_terminate_on_grasp_success: bool = False
    stage1_terminate_on_lift_success: bool = True

    # ========== B：力度目标 reward（仅 lift 生效） ==========
    # stage1：只在 grasp_ready 且 lifted 后生效（scale 默认很小或 0）
    stage1_force_target_reward_scale: float = 0.10
    stage1_force_target: float = 15.0
    stage1_force_target_margin: float = 20.0

    # ==================== Stage2: grasp_lift ====================
    stage2_mix_steps: int = 300
    stage2_stage1_weight_start: float = 1.0
    stage2_stage1_weight_end: float = 0.0
    stage2_lift_weight_start: float = 0.0
    stage2_lift_weight_end: float = 1.0
    stage2_min_stage1_weight_before_grasp: float = 1.0

    stage2_drop_penalty_scale: float = 1.0
    stage2_time_penalty: float = 0.0

    stage2_lost_steps: int = 3
    stage2_release_dist: float = 0.06
    stage2_terminate_on_success: bool = True
    stage2_terminate_on_drop: bool = True

    # stage2 也建议要求 pinch_ok 才算 has_cube（防止靠结构托住）
    stage2_pinch_require_for_grasp: bool = True

    # stage2 lift shaping
    stage2_lift_height_reward_scale: float = 6.0
    stage2_lift_delta_reward_scale: float = 3.0
    stage2_success_reward: float = 200.0  # lift 成功一次性奖励

    # ========== B：力度目标 reward（仅 lift 生效，stage2 主力） ==========
    stage2_force_target_reward_scale: float = 0.60
    stage2_force_target: float = 18.0
    stage2_force_target_margin: float = 25.0
    stage2_force_target_start_height: float = 0.02  # box_height>=此值才启用

    stage2_force_balance_reward_scale: float = 0.30  # lift 时希望更对称

    # -------------------- 通用 release / drop 高度阈值 --------------------
    drop_height_eps: float = 0.02
    release_dist_tol: float = 0.06

    # ==================== 连续夹爪控制（新增：控制器会用） ====================
    gripper_gap_min: float = 0.0      # 两指总开度（或 gap）下限
    gripper_gap_max: float = 0.08     # 两指总开度上限
    gripper_max_speed: float = 0.12   # gap 每秒最大变化（单位：m/s 或你的 gap 单位/秒）
    gripper_action_deadzone: float = 0.02
    gripper_smoothing: float = 0.20   # 0~1，越大越跟手（但更抖）
    gripper_safe_clip_margin: float = 0.0

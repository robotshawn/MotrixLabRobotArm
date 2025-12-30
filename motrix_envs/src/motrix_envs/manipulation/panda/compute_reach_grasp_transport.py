# motrix_envs/src/motrix_envs/manipulation/panda/compute_reach_grasp_transport.py
import numpy as np

from .compute_common import _compute_common_terms
from .drop_functions import (
    compute_home_progress_after_drop,
    compute_target_progress_before_drop,
)
from .reward_functions import (
    floor_underbox_penalties,
    pinch_quality_geommean,
    stable_grasp_with_pinch,
    update_hold_counter,
    grasp_hold_rewards,
)


def compute_reward_reach_grasp_transport(env, data, info):
    # 兼容旧入口名
    return _compute_reward_reach_grasp_transport(env, data, info)


def _get_float_config_value(reward_configuration, key: str, default: float, old_key: str | None = None) -> float:
    if old_key is None:
        return float(getattr(reward_configuration, key, default))
    return float(getattr(reward_configuration, key, getattr(reward_configuration, old_key, default)))


def _get_int_config_value(reward_configuration, key: str, default: int, old_key: str | None = None) -> int:
    if old_key is None:
        return int(getattr(reward_configuration, key, default))
    return int(getattr(reward_configuration, key, getattr(reward_configuration, old_key, default)))


def _get_bool_config_value(reward_configuration, key: str, default: bool, old_key: str | None = None) -> bool:
    if old_key is None:
        return bool(getattr(reward_configuration, key, default))
    return bool(getattr(reward_configuration, key, getattr(reward_configuration, old_key, default)))


def _clip_to_unit_interval(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def _score_near_distance(distance_value: np.ndarray, distance_scale, eps: float = 1e-6) -> np.ndarray:
    """score_near(distance; scale) = clip01(1 - distance / scale) ; scale can be float or (N,) array"""
    distance_value = np.asarray(distance_value, dtype=np.float32)
    distance_scale = np.asarray(distance_scale, dtype=np.float32)
    distance_scale = np.clip(distance_scale, eps, None)
    return _clip_to_unit_interval(1.0 - distance_value / distance_scale)


def _score_lift_height(height_value: np.ndarray, height_start: float, height_end: float, eps: float = 1e-6) -> np.ndarray:
    height_value = np.asarray(height_value, dtype=np.float32)
    denominator = float(max(height_end - height_start, eps))
    return _clip_to_unit_interval((height_value - float(height_start)) / denominator)


def _compute_reward_reach_grasp_transport(env, data, info):
    number_of_environments = data.shape[0]
    reward_configuration = env.cfg.reward_config
    common_terms = _compute_common_terms(env, data, info)

    box_position_world = common_terms["box_pos"].astype(np.float32)
    end_effector_position_world = common_terms["ee_pos"].astype(np.float32)
    target_position_world = common_terms["target_pos"].astype(np.float32)

    absolute_box_height_world_z = box_position_world[:, 2].astype(np.float32)

    fingertip_center_to_box_distance = common_terms["dist_fingertip_center_box"].astype(np.float32)

    fingertip_center_position_world = common_terms["fingertip_center_pos"].astype(np.float32)
    fingertip_center_to_target_distance_3d = np.linalg.norm(
        fingertip_center_position_world - target_position_world, axis=1
    ).astype(np.float32)

    end_effector_home_position_world = np.asarray(
        info.get("ee_home_pos", end_effector_position_world.copy()), dtype=np.float32
    )
    if end_effector_home_position_world.shape[0] != number_of_environments:
        end_effector_home_position_world = end_effector_home_position_world.reshape(number_of_environments, 3)
    end_effector_to_home_distance = np.linalg.norm(
        end_effector_position_world - end_effector_home_position_world, axis=1
    ).astype(np.float32)

    step_counters_for_reward = info.get("steps_for_reward", info.get("steps", np.zeros(number_of_environments, dtype=np.int32)))
    step_counters_for_reward = np.asarray(step_counters_for_reward, dtype=np.int32)
    reset_episode = (step_counters_for_reward <= 1)

    maximum_absolute_reward = float(getattr(reward_configuration, "max_step_reward", 200.0))

    lift_success_achieved_previous = np.asarray(
        info.get("reach_grasp_transport_lift_success_achieved", np.zeros(number_of_environments, dtype=bool)),
        dtype=bool,
    )
    lift_success_achieved_previous = np.where(reset_episode, False, lift_success_achieved_previous).astype(bool)

    previous_attempt_failed = np.asarray(
        info.get("reach_grasp_transport_previous_attempt_failed", np.zeros(number_of_environments, dtype=bool)),
        dtype=bool,
    )
    previous_attempt_failed = np.where(reset_episode | lift_success_achieved_previous, False, previous_attempt_failed).astype(bool)
    attempt_reset = (reset_episode | previous_attempt_failed).astype(bool)

    force_threshold = _get_float_config_value(reward_configuration, "stage1_force_threshold", 1.5)
    height_threshold = _get_float_config_value(reward_configuration, "stage1_height_threshold", 0.02)
    grasp_distance_threshold = _get_float_config_value(reward_configuration, "stage1_grasp_dist_threshold", 0.03)

    grasp_gap_minimum = _get_float_config_value(reward_configuration, "stage1_grasp_gap_min", 0.018)
    grasp_gap_maximum = _get_float_config_value(reward_configuration, "stage1_grasp_gap_max", 0.031)

    stay_distance_multiplier = _get_float_config_value(reward_configuration, "stage1_stay_dist_multiplier", 1.25)
    stay_force_multiplier = _get_float_config_value(reward_configuration, "stage1_stay_force_multiplier", 0.85)
    stay_gap_margin = _get_float_config_value(reward_configuration, "stage1_stay_gap_margin", 0.003)
    stay_height_margin = _get_float_config_value(reward_configuration, "stage1_stay_height_margin", 0.006)

    left_finger_force = common_terms["left_force"].astype(np.float32)
    right_finger_force = common_terms["right_force"].astype(np.float32)
    finger_gap_value = common_terms["finger_gap_val"].astype(np.float32)

    has_cube_previous = np.asarray(common_terms["has_cube_prev"], dtype=bool)

    left_fingertip_to_box_distance = common_terms.get("dist_left_fingertip_box", None)
    right_fingertip_to_box_distance = common_terms.get("dist_right_fingertip_box", None)
    if left_fingertip_to_box_distance is not None and right_fingertip_to_box_distance is not None:
        maximum_fingertip_to_box_distance = np.maximum(
            np.asarray(left_fingertip_to_box_distance, dtype=np.float32),
            np.asarray(right_fingertip_to_box_distance, dtype=np.float32),
        ).astype(np.float32)
    else:
        maximum_fingertip_to_box_distance = None

    pinch_quality_score, pinch_active_mask, _force_balance_value, _tip_distance_gate_value = pinch_quality_geommean(
        cfg_r=reward_configuration,
        finger_gap_val=finger_gap_value,
        dist_center_box=fingertip_center_to_box_distance,
        left_force=left_finger_force,
        right_force=right_finger_force,
        tip_dist_max=maximum_fingertip_to_box_distance,
        orient_gate=np.ones_like(fingertip_center_to_box_distance, dtype=np.float32),
        gap_min=grasp_gap_minimum,
        gap_max=grasp_gap_maximum,
        grasp_dist_threshold=grasp_distance_threshold,
        stay_gap_margin=stay_gap_margin,
        force_threshold=force_threshold,
        eps=1e-6,
    )
    pinch_quality_score = np.asarray(pinch_quality_score, dtype=np.float32)
    pinch_active_mask = np.asarray(pinch_active_mask, dtype=bool)

    _stable_grasp_candidate_mask, stable_grasp_now = stable_grasp_with_pinch(
        cfg_r=reward_configuration,
        has_cube_prev=has_cube_previous,
        finger_gap_val=finger_gap_value,
        left_force=left_finger_force,
        right_force=right_finger_force,
        dist_center_box=fingertip_center_to_box_distance,
        center_z=common_terms["fingertip_center_pos"][:, 2].astype(np.float32),
        tip_dist_max=maximum_fingertip_to_box_distance,
        orient_approach=common_terms.get("orient_approach", None),
        pinch_quality=pinch_quality_score,
        gap_min=grasp_gap_minimum,
        gap_max=grasp_gap_maximum,
        force_threshold=force_threshold,
        height_threshold=height_threshold,
        grasp_dist_threshold=grasp_distance_threshold,
        stay_dist_mult=stay_distance_multiplier,
        stay_force_mult=stay_force_multiplier,
        stay_gap_margin=stay_gap_margin,
        stay_h_margin=stay_height_margin,
    )
    stable_grasp_now = np.asarray(stable_grasp_now, dtype=bool)
    has_cube_now = stable_grasp_now.astype(bool)

    hold_counter_previous = np.asarray(info.get("hold_counter", np.zeros(number_of_environments, dtype=np.int32)), dtype=np.int32)
    hold_counter_previous = np.where(attempt_reset, 0, hold_counter_previous).astype(np.int32)
    hold_counter_current = update_hold_counter(
        cfg_r=reward_configuration, stable_grasp_now=stable_grasp_now, hold_counter_prev=hold_counter_previous
    )

    _grasp_pre_ready_mask, grasp_ready_mask, _hold_build_reward, _hold_maintain_reward = grasp_hold_rewards(
        cfg_r=reward_configuration, stable_grasp_now=stable_grasp_now, hold_counter=hold_counter_current
    )
    grasp_ready_mask = np.asarray(grasp_ready_mask, dtype=bool)
    grasp_ready_and_has_cube_mask = (grasp_ready_mask & has_cube_now)

    initial_box_height_world_z = float(getattr(env, "_box_z0", 0.03))
    lift_height_start_abs = _get_float_config_value(reward_configuration, "rg3_h_a_abs_rwdfunc", initial_box_height_world_z + 0.005)
    lift_height_transition_abs = _get_float_config_value(
        reward_configuration, "rg3_h_tr_abs_rwdfunc", initial_box_height_world_z + 0.020
    )
    lift_height_end_abs = _get_float_config_value(reward_configuration, "rg3_h_b_abs_rwdfunc", 0.10)

    lift_height_transition_abs = float(max(lift_height_transition_abs, (initial_box_height_world_z + 1e-6)))
    lift_height_start_abs = float(min(lift_height_start_abs, lift_height_transition_abs - 1e-6))
    lift_height_end_abs = float(max(lift_height_end_abs, lift_height_transition_abs + 1e-6))

    grasp_distance_scale_default = float(max(0.12, 4.0 * grasp_distance_threshold))
    grasp_distance_scale = _get_float_config_value(reward_configuration, "rg3_m_gr_rwdfunc", grasp_distance_scale_default)

    pinch_quality_mixing_ratio = _get_float_config_value(reward_configuration, "rg3_gp_beta_rwdfunc", 0.20)
    pinch_quality_mixing_ratio = float(np.clip(pinch_quality_mixing_ratio, 0.0, 1.0))

    score_near_grasp_distance = _score_near_distance(fingertip_center_to_box_distance, grasp_distance_scale)
    grasp_approach_score = (
        score_near_grasp_distance
        * (pinch_quality_mixing_ratio + (1.0 - pinch_quality_mixing_ratio) * pinch_quality_score)
    ).astype(np.float32)

    lift_score = _score_lift_height(
        absolute_box_height_world_z, height_start=lift_height_start_abs, height_end=lift_height_end_abs
    ).astype(np.float32)

    lift_hold_steps_required = max(
        _get_int_config_value(reward_configuration, "lift_hold_steps_rwdfunc", 5, old_key="stage1_lift_hold_steps"),
        1,
    )

    prev_abs_box_height_z_previous = np.asarray(
        info.get("reach_grasp_transport_prev_abs_box_height_z", absolute_box_height_world_z.copy()),
        dtype=np.float32,
    )
    if prev_abs_box_height_z_previous.shape[0] != number_of_environments:
        prev_abs_box_height_z_previous = absolute_box_height_world_z.copy().astype(np.float32)
    prev_abs_box_height_z_previous = np.where(reset_episode, absolute_box_height_world_z, prev_abs_box_height_z_previous).astype(np.float32)

    lift_success_now = (grasp_ready_and_has_cube_mask & (absolute_box_height_world_z >= float(lift_height_end_abs))).astype(bool)
    lift_success_achieved_current = (lift_success_achieved_previous | lift_success_now).astype(bool)
    lift_success_just_achieved = (lift_success_achieved_current & (~lift_success_achieved_previous)).astype(bool)

    # ✅ NEW: lift 成功一次性大额奖励（best 奖励）
    lift_success_bonus_value = _get_float_config_value(reward_configuration, "rg3_R_lift_success_bonus_rwdfunc", 50.0)
    lift_success_bonus_reward = (lift_success_bonus_value * lift_success_just_achieved.astype(np.float32)).astype(np.float32)

    lift_rise_eps = _get_float_config_value(reward_configuration, "rg3_lift_rise_eps_rwdfunc", 1e-5)
    lift_slow_reward_weight = _get_float_config_value(reward_configuration, "rg3_R_lift_slow_rwdfunc", 0.5)

    in_lift_band_now = (
        (absolute_box_height_world_z >= float(lift_height_start_abs))
        & (absolute_box_height_world_z <= float(lift_height_end_abs))
    ).astype(bool)
    is_rising_now = (absolute_box_height_world_z > (prev_abs_box_height_z_previous + float(lift_rise_eps))).astype(bool)

    band_entry_now = (
        grasp_ready_and_has_cube_mask
        & (absolute_box_height_world_z >= float(lift_height_start_abs))
        & (prev_abs_box_height_z_previous < float(lift_height_start_abs))
        & (~lift_success_achieved_current)
    ).astype(bool)

    lift_slow_active_previous = np.asarray(
        info.get("reach_grasp_transport_lift_slow_active", np.zeros(number_of_environments, dtype=bool)),
        dtype=bool,
    )
    if lift_slow_active_previous.shape[0] != number_of_environments:
        lift_slow_active_previous = np.zeros((number_of_environments,), dtype=bool)
    lift_slow_active_previous = np.where(attempt_reset | lift_success_achieved_current, False, lift_slow_active_previous).astype(bool)

    lift_slow_counter_previous = np.asarray(
        info.get("reach_grasp_transport_lift_slow_counter", np.zeros(number_of_environments, dtype=np.int32)),
        dtype=np.int32,
    )
    if lift_slow_counter_previous.shape[0] != number_of_environments:
        lift_slow_counter_previous = np.zeros((number_of_environments,), dtype=np.int32)
    lift_slow_counter_previous = np.where(attempt_reset | lift_success_achieved_current, 0, lift_slow_counter_previous).astype(np.int32)

    slow_condition_now = (grasp_ready_and_has_cube_mask & in_lift_band_now & is_rising_now & (~lift_success_achieved_current)).astype(bool)

    would_finish_by_count = (lift_slow_counter_previous >= int(lift_hold_steps_required)).astype(bool)
    should_terminate_slow = ((~slow_condition_now) | would_finish_by_count | lift_success_now).astype(bool)

    lift_slow_active_current = np.where(
        lift_slow_active_previous,
        np.where(should_terminate_slow, False, True),
        np.where(band_entry_now, True, False),
    ).astype(bool)

    lift_slow_counter_current = np.where(lift_slow_active_current & slow_condition_now, lift_slow_counter_previous + 1, 0).astype(np.int32)

    lift_slow_reward = (
        lift_slow_reward_weight
        * (lift_slow_active_current & slow_condition_now & (lift_slow_counter_current <= int(lift_hold_steps_required))).astype(np.float32)
    ).astype(np.float32)

    # ======================================================================
    # ✅ DROP DESIGN (ONLY):
    # lift 成功之后，drop 判定：gap > 0.037 -> True，否则 False
    # drop 一旦为 True，则一直保持 True，直到 reset episode
    # ======================================================================
    DROP_GAP_THRESHOLD = 0.036

    has_dropped_after_lift_previous = np.asarray(
        info.get("reach_grasp_transport_has_dropped_after_lift", np.zeros(number_of_environments, dtype=bool)),
        dtype=bool,
    )
    if has_dropped_after_lift_previous.shape[0] != number_of_environments:
        has_dropped_after_lift_previous = np.zeros((number_of_environments,), dtype=bool)

    # reset / lift_success 刚达成时，上一帧 drop 视为 False（避免跨段污染）
    has_dropped_after_lift_previous = np.where(reset_episode | lift_success_just_achieved, False, has_dropped_after_lift_previous).astype(bool)

    # ✅ LATCH: drop 一旦触发就保持 True，直到 reset episode
    drop_now = (lift_success_achieved_current & (finger_gap_value > float(DROP_GAP_THRESHOLD))).astype(bool)
    has_dropped_after_lift_current = (has_dropped_after_lift_previous | drop_now).astype(bool)
    has_dropped_after_lift_just_now = (has_dropped_after_lift_current & (~has_dropped_after_lift_previous)).astype(bool)

    # ======================================================================
    # ✅ NEW (REQUESTED):
    # reach success 之后直到 (drop 完成 && dist(ft_center, cube) > 0.1)：
    #   - 夹爪轴需保持与世界系向下方向夹角 <= 45°（cos_down >= cos(45°)）
    #   - 否则视为无效探索，reset episode
    # ======================================================================
    REACH_SUCCESS_DIST = _get_float_config_value(reward_configuration, "rg3_reach_success_dist_rwdfunc", 0.10)
    POST_DROP_RELEASE_DIST = _get_float_config_value(reward_configuration, "rg3_post_drop_release_dist_rwdfunc", 0.10)
    GRIPPER_DOWN_MAX_ANGLE_DEG = _get_float_config_value(reward_configuration, "rg3_gripper_down_max_angle_deg_rwdfunc", 45.0)
    _angle_deg = float(np.clip(GRIPPER_DOWN_MAX_ANGLE_DEG, 0.0, 180.0))
    gripper_down_cos_min = float(np.cos(np.deg2rad(_angle_deg)))

    gripper_cos_down = np.asarray(
        common_terms.get("cos_gripper_down", np.ones(number_of_environments, dtype=np.float32)),
        dtype=np.float32,
    )
    if gripper_cos_down.shape[0] != number_of_environments:
        gripper_cos_down = np.ones((number_of_environments,), dtype=np.float32)
    gripper_within_down_45deg = (gripper_cos_down >= gripper_down_cos_min).astype(bool)

    reach_success_achieved_previous = np.asarray(
        info.get("reach_grasp_transport_reach_success_achieved", np.zeros(number_of_environments, dtype=bool)),
        dtype=bool,
    )
    if reach_success_achieved_previous.shape[0] != number_of_environments:
        reach_success_achieved_previous = np.zeros((number_of_environments,), dtype=bool)
    # attempt_reset 代表“新一轮尝试”，避免跨 attempt 污染；reset_episode 则是 episode 级复位
    reach_success_achieved_previous = np.where(attempt_reset, False, reach_success_achieved_previous).astype(bool)

    reach_success_now = (fingertip_center_to_box_distance <= float(REACH_SUCCESS_DIST)).astype(bool)
    reach_success_achieved_current = (reach_success_achieved_previous | reach_success_now).astype(bool)

    # 只有当 (drop 已完成 && dist > 0.1) 才解除该约束
    orientation_constraint_active = (
        reach_success_achieved_current
        & (~(has_dropped_after_lift_current & (fingertip_center_to_box_distance > float(POST_DROP_RELEASE_DIST))))
    ).astype(bool)

    reset_due_to_bad_gripper_orientation = (
        orientation_constraint_active & (~gripper_within_down_45deg)
    ).astype(bool)

    # ======================================================================
    # ✅ NEW DESIGN (REQUESTED):
    # drop 后 target-region gating：
    #   - 只有当 drop 发生瞬间，夹爪 (ee) 在 target 的 |dx|,|dy|,|dz| <= gate(默认0.15) 区域内，
    #     才“启动/armed”回 home 的 reward（latched）
    #   - 若 drop 发生瞬间不在区域内 => reset episode
    # ======================================================================
    DROP_TO_HOME_TARGET_GATE = _get_float_config_value(reward_configuration, "rg3_drop_to_home_target_gate_rwdfunc", 0.15)
    delta_ee_to_target = np.abs(end_effector_position_world - target_position_world).astype(np.float32)
    in_drop_target_gate = (delta_ee_to_target <= float(DROP_TO_HOME_TARGET_GATE)).all(axis=1).astype(bool)

    home_reward_armed_previous = np.asarray(
        info.get("reach_grasp_transport_home_reward_armed", np.zeros(number_of_environments, dtype=bool)),
        dtype=bool,
    )
    if home_reward_armed_previous.shape[0] != number_of_environments:
        home_reward_armed_previous = np.zeros((number_of_environments,), dtype=bool)

    home_reward_armed_previous = np.where(reset_episode | lift_success_just_achieved, False, home_reward_armed_previous).astype(bool)

    # drop 刚发生：若在 gate 内 => armed=True；否则 armed=False（并触发 reset）
    home_reward_armed_current = np.where(
        home_reward_armed_previous,
        True,
        np.where(has_dropped_after_lift_just_now, in_drop_target_gate, False),
    ).astype(bool)

    reset_due_to_drop_not_in_target_gate = (has_dropped_after_lift_just_now & (~in_drop_target_gate)).astype(bool)

    # ======================================================================
    # ✅ NEW DESIGN #1 (REQUESTED BY USER):
    # drop 后 cube 必须始终处于 target 的 |dx|,|dy|,|dz| <= gate(默认0.15) 区域内：
    #   - 在区域内：按 “cube 每步移动 delta(3D)” 给稳定奖励：移动越少奖励越大
    #   - 一旦 drop 后 cube 离开该区域：reset episode
    # ======================================================================
    DROP_BOX_TARGET_GATE = _get_float_config_value(reward_configuration, "rg3_drop_box_target_gate_rwdfunc", 0.15)
    delta_box_to_target = np.abs(box_position_world - target_position_world).astype(np.float32)
    in_drop_box_target_gate = (delta_box_to_target <= float(DROP_BOX_TARGET_GATE)).all(axis=1).astype(bool)

    reset_due_to_drop_box_outside_target_gate = (has_dropped_after_lift_current & (~in_drop_box_target_gate)).astype(bool)

    # box movement delta (3D) from previous step
    prev_box_xy = np.asarray(common_terms.get("prev_box_xy", box_position_world[:, :2].copy()), dtype=np.float32)
    if prev_box_xy.shape[0] != number_of_environments:
        prev_box_xy = box_position_world[:, :2].copy().astype(np.float32)

    prev_box_height_rel = np.asarray(common_terms.get("prev_box_height", (box_position_world[:, 2] - float(env._box_z0))), dtype=np.float32)
    if prev_box_height_rel.shape[0] != number_of_environments:
        prev_box_height_rel = (box_position_world[:, 2] - float(env._box_z0)).astype(np.float32)

    prev_box_z_world = (prev_box_height_rel + float(env._box_z0)).astype(np.float32)
    prev_box_pos_world = np.concatenate([prev_box_xy, prev_box_z_world.reshape(-1, 1)], axis=1).astype(np.float32)

    drop_box_move_delta_3d = np.linalg.norm(box_position_world - prev_box_pos_world, axis=1).astype(np.float32)

    drop_box_move_scale = _get_float_config_value(reward_configuration, "rg3_drop_box_move_scale_rwdfunc", 0.01)
    drop_box_stability_weight = _get_float_config_value(reward_configuration, "rg3_R_drop_box_stability_rwdfunc", 0.5)

    drop_box_stability_score = _score_near_distance(drop_box_move_delta_3d, drop_box_move_scale)
    drop_box_stability_reward = (
        drop_box_stability_weight
        * drop_box_stability_score
        * (has_dropped_after_lift_current & in_drop_box_target_gate).astype(np.float32)
    ).astype(np.float32)

    # ======================================================================
    # ✅ NEW RESET DESIGNS (existing):
    # 1) drop 后经过 K 步，如果 cube 与爪尖中心距离 <= 0.05 -> reset episode
    # 2) 没有 drop（但 lift_success 已经发生）时，如果 cube 与爪尖中心距离 > 0.05 -> reset episode
    # ======================================================================
    POST_DROP_CLOSE_RESET_DIST = 0.05
    POST_DROP_CLOSE_RESET_STEPS = max(
        _get_int_config_value(reward_configuration, "rg3_post_drop_close_reset_steps_rwdfunc", 5),
        0,
    )
    POST_DROP_FAR_REWARD_VALUE = _get_float_config_value(reward_configuration, "rg3_R_post_drop_far_rwdfunc", 5.0)

    steps_since_drop_previous = np.asarray(
        info.get("reach_grasp_transport_steps_since_drop", np.zeros(number_of_environments, dtype=np.int32)),
        dtype=np.int32,
    )
    if steps_since_drop_previous.shape[0] != number_of_environments:
        steps_since_drop_previous = np.zeros((number_of_environments,), dtype=np.int32)

    # reset / lift_success 切段：计数清零（drop 计数只在 lift_success 后才有意义）
    steps_since_drop_previous = np.where(reset_episode | lift_success_just_achieved, 0, steps_since_drop_previous).astype(np.int32)

    steps_since_drop_current = np.where(
        has_dropped_after_lift_current,
        np.where(has_dropped_after_lift_just_now, 0, steps_since_drop_previous + 1),
        0,
    ).astype(np.int32)

    # ======================================================================
    # ✅ NEW (REQUESTED):
    # drop 后 5 步内远离 cube 的短时奖励：
    # ======================================================================
    prev_fingertip_center_to_box_distance = np.asarray(
        info.get("prev_dist_center_box", fingertip_center_to_box_distance.copy()),
        dtype=np.float32,
    )
    if prev_fingertip_center_to_box_distance.shape[0] != number_of_environments:
        prev_fingertip_center_to_box_distance = fingertip_center_to_box_distance.copy().astype(np.float32)
    prev_fingertip_center_to_box_distance = np.where(
        reset_episode,
        fingertip_center_to_box_distance,
        prev_fingertip_center_to_box_distance,
    ).astype(np.float32)

    post_drop_away_delta_dist_3d = (fingertip_center_to_box_distance - prev_fingertip_center_to_box_distance).astype(np.float32)

    post_drop_away_delta_lo = _get_float_config_value(reward_configuration, "rg3_post_drop_away_delta_lo_rwdfunc", 0.04)
    post_drop_away_delta_hi = _get_float_config_value(reward_configuration, "rg3_post_drop_away_delta_hi_rwdfunc", 0.08)
    post_drop_away_far_dist = _get_float_config_value(reward_configuration, "rg3_post_drop_away_far_dist_rwdfunc", 0.15)

    post_drop_away_step13_reward_value = _get_float_config_value(reward_configuration, "rg3_R_post_drop_away_step13_rwdfunc", 0.5)
    post_drop_away_step45_reward_value = _get_float_config_value(reward_configuration, "rg3_R_post_drop_away_step45_rwdfunc", 1.0)

    post_drop_away_step13_mask = (
        has_dropped_after_lift_current
        & (steps_since_drop_current >= 1)
        & (steps_since_drop_current <= 3)
    ).astype(bool)
    post_drop_away_step45_mask = (
        has_dropped_after_lift_current
        & (steps_since_drop_current >= 4)
        & (steps_since_drop_current <= 5)
    ).astype(bool)

    post_drop_away_valid_gate = (
        (~reset_due_to_drop_not_in_target_gate)
        & (~reset_due_to_drop_box_outside_target_gate)
        & (has_dropped_after_lift_current)
    ).astype(bool)

    post_drop_away_delta_in_band = (
        (post_drop_away_delta_dist_3d >= float(post_drop_away_delta_lo))
        & (post_drop_away_delta_dist_3d <= float(post_drop_away_delta_hi))
    ).astype(bool)
    post_drop_away_far_enough = (fingertip_center_to_box_distance >= float(post_drop_away_far_dist)).astype(bool)

    post_drop_away_step13_reward = (
        float(post_drop_away_step13_reward_value)
        * post_drop_away_delta_in_band.astype(np.float32)
        * post_drop_away_step13_mask.astype(np.float32)
        * post_drop_away_valid_gate.astype(np.float32)
    ).astype(np.float32)

    post_drop_away_step45_reward = (
        float(post_drop_away_step45_reward_value)
        * post_drop_away_far_enough.astype(np.float32)
        * post_drop_away_step45_mask.astype(np.float32)
        * post_drop_away_valid_gate.astype(np.float32)
    ).astype(np.float32)

    post_drop_away_reward = (post_drop_away_step13_reward + post_drop_away_step45_reward).astype(np.float32)

    # ======================================================================
    # ✅ NEW (REQUESTED):
    # drop 后前 K 步内累计 cube 3D 移动越少 => 一次性奖励越大
    # ======================================================================
    POST_DROP_MOVE_REWARD_STEPS = max(
        _get_int_config_value(
            reward_configuration,
            "rg3_post_drop_move_reward_steps_rwdfunc",
            int(POST_DROP_CLOSE_RESET_STEPS),
        ),
        0,
    )

    post_drop_move_sum_previous = np.asarray(
        info.get("reach_grasp_transport_post_drop_move_sum_3d", np.zeros(number_of_environments, dtype=np.float32)),
        dtype=np.float32,
    )
    if post_drop_move_sum_previous.shape[0] != number_of_environments:
        post_drop_move_sum_previous = np.zeros((number_of_environments,), dtype=np.float32)

    # reset / 切段 / drop 刚发生：累计清零
    post_drop_move_sum_previous = np.where(
        reset_episode | lift_success_just_achieved | has_dropped_after_lift_just_now,
        0.0,
        post_drop_move_sum_previous,
    ).astype(np.float32)

    in_post_drop_move_window = (
        has_dropped_after_lift_current
        & (steps_since_drop_current <= int(POST_DROP_MOVE_REWARD_STEPS))
    ).astype(bool)

    post_drop_move_sum_current = np.where(
        in_post_drop_move_window,
        post_drop_move_sum_previous + drop_box_move_delta_3d,
        post_drop_move_sum_previous,
    ).astype(np.float32)

    post_drop_move_check_at_k = (
        has_dropped_after_lift_current
        & (steps_since_drop_current == int(POST_DROP_MOVE_REWARD_STEPS))
    ).astype(bool)

    post_drop_move_sum_scale = _get_float_config_value(
        reward_configuration, "rg3_post_drop_move_sum_scale_rwdfunc", 0.03
    )
    post_drop_move_reward_value = _get_float_config_value(
        reward_configuration, "rg3_R_post_drop_move_stability_rwdfunc", 1.0
    )

    post_drop_move_score = _score_near_distance(post_drop_move_sum_current, post_drop_move_sum_scale).astype(np.float32)
    post_drop_move_stability_reward = (
        float(post_drop_move_reward_value)
        * post_drop_move_score
        * post_drop_move_check_at_k.astype(np.float32)
        * in_drop_box_target_gate.astype(np.float32)
        * (~reset_due_to_drop_not_in_target_gate).astype(np.float32)
        * (~reset_due_to_drop_box_outside_target_gate).astype(np.float32)
    ).astype(np.float32)

    # 只在 “第 K 步这一刻” 做一次判定
    post_drop_check_at_k = (
        has_dropped_after_lift_current
        & (steps_since_drop_current == int(POST_DROP_CLOSE_RESET_STEPS))
    ).astype(bool)

    reset_due_to_post_drop_close = (
        post_drop_check_at_k
        & (fingertip_center_to_box_distance <= float(POST_DROP_CLOSE_RESET_DIST))
    ).astype(bool)

    # 第 K 步未触发 close-reset（距离 > 0.05）：给一次性 reward
    post_drop_far_reward = (
        float(POST_DROP_FAR_REWARD_VALUE)
        * (
            post_drop_check_at_k
            & (fingertip_center_to_box_distance > float(POST_DROP_CLOSE_RESET_DIST))
            & (~reset_due_to_drop_not_in_target_gate)
            & (~reset_due_to_drop_box_outside_target_gate)
        ).astype(np.float32)
    ).astype(np.float32)

    reset_due_to_no_drop_far = (
        lift_success_achieved_current
        & (~has_dropped_after_lift_current)
        & (fingertip_center_to_box_distance > float(POST_DROP_CLOSE_RESET_DIST))
    ).astype(bool)

    # ✅ include new reset: cube leaves target gate after drop
    reset_episode_current = (
        reset_due_to_post_drop_close
        | reset_due_to_no_drop_far
        | reset_due_to_drop_not_in_target_gate
        | reset_due_to_drop_box_outside_target_gate
        | reset_due_to_bad_gripper_orientation
    ).astype(bool)

    before_lift_success_mask = (~lift_success_achieved_current).astype(bool)
    gate_for_grasp_approach_progress = (before_lift_success_mask & (~grasp_ready_and_has_cube_mask)).astype(bool)
    gate_for_lift_progress = (before_lift_success_mask & grasp_ready_and_has_cube_mask).astype(bool)

    def _update_best_score(score_key: str, current_score: np.ndarray, gate_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        best_score_previous = np.asarray(info.get(score_key, np.zeros(number_of_environments, dtype=np.float32)), dtype=np.float32)
        best_score_previous = np.where(attempt_reset, 0.0, best_score_previous).astype(np.float32)
        best_score_current = np.where(gate_mask, np.maximum(best_score_previous, current_score), best_score_previous).astype(np.float32)
        return best_score_previous, best_score_current

    best_grasp_approach_score_previous, best_grasp_approach_score_current = _update_best_score(
        "reach_grasp_transport_best_grasp_approach_score",
        grasp_approach_score,
        gate_for_grasp_approach_progress,
    )
    best_lift_score_previous, best_lift_score_current = _update_best_score(
        "reach_grasp_transport_best_lift_score",
        lift_score,
        gate_for_lift_progress,
    )

    minimum_grasp_approach_delta = _get_float_config_value(reward_configuration, "rg3_delta_gp_rwdfunc", 1e-4)
    minimum_lift_delta = _get_float_config_value(reward_configuration, "rg3_delta_lift_rwdfunc", 1e-3)

    def _positive_best_improvement(best_now: np.ndarray, best_prev: np.ndarray, minimum_delta: float) -> np.ndarray:
        return np.maximum(0.0, (best_now - best_prev - float(max(minimum_delta, 0.0)))).astype(np.float32)

    grasp_approach_best_improvement = _positive_best_improvement(
        best_grasp_approach_score_current, best_grasp_approach_score_previous, minimum_grasp_approach_delta
    )
    lift_best_improvement = _positive_best_improvement(best_lift_score_current, best_lift_score_previous, minimum_lift_delta)

    grasp_approach_weight = _get_float_config_value(reward_configuration, "rg3_w_gp_rwdfunc", 2.0)
    lift_weight = _get_float_config_value(reward_configuration, "rg3_w_lift_rwdfunc", 3.0)

    pre_lift_progress_reward = (grasp_approach_weight * grasp_approach_best_improvement + lift_weight * lift_best_improvement).astype(np.float32)

    # lift 成功后 & drop 前：仍然必须拿着物体（has_cube_now）才继续刷 transport progress
    post_lift_before_drop_mask = (lift_success_achieved_current & (~has_dropped_after_lift_current) & has_cube_now).astype(bool)

    target_out = compute_target_progress_before_drop(
        reward_configuration=reward_configuration,
        info=info,
        number_of_environments=number_of_environments,
        reset_episode=reset_episode,
        lift_success_just_achieved=lift_success_just_achieved,
        post_lift_before_drop_mask=post_lift_before_drop_mask,
        fingertip_center_to_target_distance_3d=fingertip_center_to_target_distance_3d,
    )
    target_progress_reward_before_drop = target_out["target_progress_reward_before_drop"].astype(np.float32)
    best_target_progress_score_current = target_out["best_target_progress_score_current"].astype(np.float32)
    target_progress_distance_scale_current = target_out["target_progress_distance_scale_current"].astype(np.float32)
    lift_success_ftcenter_target_snapshot_current = target_out["lift_success_ftcenter_target_snapshot_current"].astype(np.float32)
    target_progress_score = target_out["target_progress_score"].astype(np.float32)
    target_progress_best_improvement = target_out["target_progress_best_improvement"].astype(np.float32)

    best_target_distance_current = target_out.get("best_target_distance_current", None)
    if best_target_distance_current is not None:
        best_target_distance_current = np.asarray(best_target_distance_current, dtype=np.float32)

    # ======================================================================
    # ✅ drop 后：home reward 只在 home_reward_armed_current=True 时启用
    # ======================================================================
    home_progress_active_mask = (has_dropped_after_lift_current & home_reward_armed_current).astype(bool)

    home_out = compute_home_progress_after_drop(
        reward_configuration=reward_configuration,
        info=info,
        number_of_environments=number_of_environments,
        reset_episode=reset_episode,
        lift_success_just_achieved=lift_success_just_achieved,
        has_dropped_after_lift_current=has_dropped_after_lift_current,
        has_dropped_after_lift_just_now=has_dropped_after_lift_just_now,
        home_progress_active_mask=home_progress_active_mask,
        end_effector_to_home_distance=end_effector_to_home_distance,
        box_position_world=box_position_world,
        target_position_world=target_position_world,
        fingertip_center_position_world=fingertip_center_position_world,
    )
    best_home_distance_previous = home_out["best_home_distance_previous"].astype(np.float32)
    best_home_distance_current = home_out["best_home_distance_current"].astype(np.float32)
    home_distance_best_improvement = home_out["home_distance_best_improvement"].astype(np.float32)
    home_progress_reward = home_out["home_progress_reward"].astype(np.float32)

    # ✅ progress reward（用于 idle/progress ledger 判定）
    positive_progress_reward = (
        pre_lift_progress_reward * before_lift_success_mask.astype(np.float32)
        + lift_slow_reward * before_lift_success_mask.astype(np.float32)
        + lift_success_bonus_reward
        + target_progress_reward_before_drop
        + home_progress_reward
        + post_drop_far_reward
    ).astype(np.float32)

    minimum_progress_epsilon = _get_float_config_value(reward_configuration, "rg3_progress_eps_rwdfunc", 0.0)
    idle_steps_threshold = max(_get_int_config_value(reward_configuration, "rg3_T_idle_rwdfunc", 200), 1)

    idle_steps_counter_previous = np.asarray(
        info.get("reach_grasp_transport_idle_steps_counter", np.zeros(number_of_environments, dtype=np.int32)),
        dtype=np.int32,
    )
    idle_steps_counter_previous = np.where(
        attempt_reset | lift_success_just_achieved | has_dropped_after_lift_just_now,
        0,
        idle_steps_counter_previous,
    ).astype(np.int32)

    idle_steps_counter_current = np.where(
        (positive_progress_reward > minimum_progress_epsilon)
        | attempt_reset
        | lift_success_just_achieved
        | has_dropped_after_lift_just_now,
        0,
        idle_steps_counter_previous + 1,
    ).astype(np.int32)
    stalled_mask = (idle_steps_counter_current >= idle_steps_threshold).astype(bool)

    maximum_episode_seconds = float(getattr(env.cfg, "max_episode_seconds", 10.0))
    control_timestep_seconds = float(getattr(env.cfg, "ctrl_dt", 0.01))
    maximum_steps = max(int(maximum_episode_seconds / max(control_timestep_seconds, 1e-6)), 1)
    timeout_mask = (step_counters_for_reward >= maximum_steps).astype(bool)

    _floor_penalty, _under_box_penalty, deep_floor_violation, _floor_violation_value, _under_box_mask_value = floor_underbox_penalties(
        cfg_r=reward_configuration,
        ee_z=end_effector_position_world[:, 2].astype(np.float32),
        center_z=common_terms["fingertip_center_pos"][:, 2].astype(np.float32),
        dist_center_box=fingertip_center_to_box_distance,
        box_pos_z=box_position_world[:, 2].astype(np.float32),
        pregrasp_dist=_get_float_config_value(reward_configuration, "upright_bonus_near_dist_rwdfunc", 0.06),
    )
    safety_failure_mask = np.asarray(deep_floor_violation, dtype=bool)

    safety_penalty_magnitude = _get_float_config_value(reward_configuration, "rg3_p_safety_mag_rwdfunc", 0.0)
    safety_penalty = (-safety_penalty_magnitude * safety_failure_mask.astype(np.float32)).astype(np.float32)

    progress_ledger_previous = np.asarray(
        info.get("reach_grasp_transport_progress_ledger", np.zeros(number_of_environments, dtype=np.float32)),
        dtype=np.float32,
    )
    progress_ledger_previous = np.where(
        attempt_reset | lift_success_just_achieved,
        0.0,
        progress_ledger_previous,
    ).astype(np.float32)
    progress_ledger_current = (progress_ledger_previous + positive_progress_reward).astype(np.float32)

    stall_penalty_magnitude = _get_float_config_value(reward_configuration, "rg3_p_stall_rwdfunc", 2.0)
    stall_penalty = (stall_penalty_magnitude * stalled_mask.astype(np.float32)).astype(np.float32)

    timeout_penalty_magnitude = _get_float_config_value(reward_configuration, "rg3_p_timeout_rwdfunc", 0.0)
    timeout_penalty = (timeout_penalty_magnitude * timeout_mask.astype(np.float32)).astype(np.float32)

    fail_settlement_multiplier = _get_float_config_value(reward_configuration, "rg3_fail_kappa_rwdfunc", 1.0)
    fail_settlement_bias = _get_float_config_value(reward_configuration, "rg3_fail_p0_rwdfunc", 0.5)

    # ✅ NEW：不再因为 “pre-lift drop” 触发 settlement
    settlement_event_mask = safety_failure_mask.astype(bool)
    settlement_penalty = (
        (fail_settlement_multiplier * progress_ledger_current + fail_settlement_bias) * settlement_event_mask.astype(np.float32)
    ).astype(np.float32)

    home_success_distance_threshold = _get_float_config_value(reward_configuration, "rg3_d_home_succ_rwdfunc", 0.06)
    at_home = (end_effector_to_home_distance <= home_success_distance_threshold).astype(bool)

    success_mask = (at_home & has_dropped_after_lift_current & (~has_cube_now)).astype(bool)

    task_success_achieved_previous = np.asarray(
        info.get("reach_grasp_transport_task_success_achieved", np.zeros(number_of_environments, dtype=bool)),
        dtype=bool,
    )
    task_success_achieved_previous = np.where(reset_episode, False, task_success_achieved_previous).astype(bool)
    success_just_achieved = (success_mask & (~task_success_achieved_previous)).astype(bool)

    # ======================================================================
    # ✅ NEW DESIGN #2 (REQUESTED BY USER):
    # 到达 home 区域后，根据 cube 此刻到 target 的 xy 距离给一次性奖励（越小越大）
    # ======================================================================
    box_to_target_xy_distance = np.linalg.norm(
        box_position_world[:, :2].astype(np.float32) - target_position_world[:, :2].astype(np.float32),
        axis=1,
    ).astype(np.float32)

    home_xy_bonus_scale = _get_float_config_value(reward_configuration, "rg3_home_xy_bonus_scale_rwdfunc", 0.15)
    home_xy_bonus_value = _get_float_config_value(reward_configuration, "rg3_R_home_xy_bonus_rwdfunc", 10.0)
    home_xy_score = _score_near_distance(box_to_target_xy_distance, home_xy_bonus_scale).astype(np.float32)
    home_xy_bonus_reward = (home_xy_bonus_value * home_xy_score * success_just_achieved.astype(np.float32)).astype(np.float32)

    # ✅ extra reward（不参与 idle/progress ledger，避免刷分）
    extra_reward = (drop_box_stability_reward + post_drop_move_stability_reward + post_drop_away_reward + home_xy_bonus_reward).astype(np.float32)

    total_reward = (positive_progress_reward + extra_reward - settlement_penalty - stall_penalty - timeout_penalty + safety_penalty).astype(np.float32)

    # ✅ NEW：attempt_failed 不再包含任何 drop 事件
    attempt_failed_event_mask = (stalled_mask | timeout_mask | safety_failure_mask).astype(bool)
    attempt_failed_event_mask = np.where(lift_success_achieved_previous | lift_success_achieved_current, False, attempt_failed_event_mask).astype(bool)

    terminated = np.zeros(number_of_environments, dtype=bool)
    if _get_bool_config_value(reward_configuration, "rg3_terminate_on_success_rwdfunc", True):
        terminated |= success_just_achieved.astype(bool)
    if _get_bool_config_value(reward_configuration, "rg3_terminate_on_stall_rwdfunc", True):
        terminated |= stalled_mask
    if _get_bool_config_value(reward_configuration, "rg3_terminate_on_safety_rwdfunc", True):
        terminated |= safety_failure_mask
    terminated |= reset_episode_current

    degrees_of_freedom_positions = common_terms["dof_pos"]
    degrees_of_freedom_velocities = common_terms["dof_vel"]
    invalid_reward_mask = np.isnan(total_reward) | np.isinf(total_reward)
    invalid_state_mask = np.isnan(degrees_of_freedom_positions).any(axis=1) | np.isnan(degrees_of_freedom_velocities).any(axis=1)
    invalid_mask = invalid_reward_mask | invalid_state_mask
    if np.any(invalid_mask):
        terminated |= invalid_mask
        invalid_penalty_value = float(getattr(reward_configuration, "invalid_penalty", -30.0))
        total_reward[invalid_mask] = invalid_penalty_value

    total_reward = np.clip(total_reward, -maximum_absolute_reward, maximum_absolute_reward).astype(np.float32)

    info["has_cube"] = has_cube_now.astype(bool)

    info["prev_dist_ee_box"] = common_terms["dist_ee_box"].astype(np.float32)
    info["prev_dist_center_box"] = fingertip_center_to_box_distance.astype(np.float32)
    info["prev_box_height"] = common_terms["box_height"].astype(np.float32)
    info["prev_box_xy"] = box_position_world[:, :2].astype(np.float32)

    info["prev_dist_box_target_3d"] = common_terms["dist_box_target_3d"].astype(np.float32)
    info["prev_dist_ftcenter_target_3d"] = fingertip_center_to_target_distance_3d.astype(np.float32)

    info["hold_counter"] = hold_counter_current.astype(np.int32)

    info["reach_grasp_transport_prev_abs_box_height_z"] = absolute_box_height_world_z.astype(np.float32)

    info["reach_grasp_transport_lift_slow_active"] = lift_slow_active_current.astype(bool)
    info["reach_grasp_transport_lift_slow_counter"] = lift_slow_counter_current.astype(np.int32)

    info["reach_grasp_transport_lift_success_achieved"] = lift_success_achieved_current.astype(bool)
    info["reach_grasp_transport_task_success_achieved"] = (task_success_achieved_previous | success_mask).astype(bool)

    info["reach_grasp_transport_best_grasp_approach_score"] = best_grasp_approach_score_current.astype(np.float32)
    info["reach_grasp_transport_best_lift_score"] = best_lift_score_current.astype(np.float32)

    info["reach_grasp_transport_has_dropped_after_lift"] = has_dropped_after_lift_current.astype(bool)

    # ✅ NEW: reach success latch（用于姿态约束区间）
    info["reach_grasp_transport_reach_success_achieved"] = reach_success_achieved_current.astype(bool)

    # ✅ NEW: drop 后 step 计数器
    info["reach_grasp_transport_steps_since_drop"] = steps_since_drop_current.astype(np.int32)

    # ✅ NEW: drop 后前 K 步累计 3D 位移（用于一次性稳定奖励）
    info["reach_grasp_transport_post_drop_move_sum_3d"] = post_drop_move_sum_current.astype(np.float32)

    # ✅ NEW: home reward gating latch
    info["reach_grasp_transport_home_reward_armed"] = home_reward_armed_current.astype(bool)

    info["reach_grasp_transport_lift_success_ftcenter_to_target_distance_3d_snapshot"] = lift_success_ftcenter_target_snapshot_current.astype(np.float32)
    info["reach_grasp_transport_target_progress_distance_scale"] = target_progress_distance_scale_current.astype(np.float32)
    info["reach_grasp_transport_best_target_progress_score"] = best_target_progress_score_current.astype(np.float32)

    # ✅ NEW: min-distance progress state（用于奖励）
    if best_target_distance_current is not None:
        info["reach_grasp_transport_best_target_distance_3d"] = best_target_distance_current.astype(np.float32)

    info["reach_grasp_transport_best_end_effector_to_home_distance"] = best_home_distance_current.astype(np.float32)

    info["reach_grasp_transport_idle_steps_counter"] = idle_steps_counter_current.astype(np.int32)
    info["reach_grasp_transport_progress_ledger"] = progress_ledger_current.astype(np.float32)

    info["reach_grasp_transport_previous_attempt_failed"] = attempt_failed_event_mask.astype(bool)

    reward_terms = {
        "grasp_approach_best_improvement": grasp_approach_best_improvement.astype(np.float32),
        "lift_best_improvement": lift_best_improvement.astype(np.float32),
        "pre_lift_progress_reward": pre_lift_progress_reward.astype(np.float32),
        "lift_slow_reward": lift_slow_reward.astype(np.float32),

        # ✅ NEW: lift 一次性大额奖励
        "lift_success_bonus_reward": lift_success_bonus_reward.astype(np.float32),

        "post_lift_before_drop_mask": post_lift_before_drop_mask.astype(np.float32),
        "fingertip_center_to_target_distance_3d": fingertip_center_to_target_distance_3d.astype(np.float32),

        # 仍保留 best 相关 debug
        "target_progress_score": target_progress_score.astype(np.float32),
        "target >> best_improvement(debug_only)": target_progress_best_improvement.astype(np.float32),

        # ✅ min-distance progress debug（若未启用则只是记录）
        "target_best_distance_3d(debug)": (best_target_distance_current.astype(np.float32) if best_target_distance_current is not None else np.zeros((number_of_environments,), dtype=np.float32)),
        "target_reward_mode_is_min_dist(debug)": target_out.get(
            "target_progress_reward_mode_is_min_dist",
            np.zeros((number_of_environments,), dtype=np.float32),
        ).astype(np.float32),

        "target_progress_reward_before_drop": target_progress_reward_before_drop.astype(np.float32),

        # ✅ home: best debug
        "home_progress_active_mask": home_progress_active_mask.astype(np.float32),
        "best_home_distance_previous(debug)": best_home_distance_previous.astype(np.float32),
        "best_home_distance_current(debug)": best_home_distance_current.astype(np.float32),
        "home_distance_best_improvement(debug_only)": home_distance_best_improvement.astype(np.float32),
        "home_progress_reward": home_progress_reward.astype(np.float32),

        # ✅ progress（用于 idle/ledger）
        "positive_progress_reward": positive_progress_reward.astype(np.float32),

        # ✅ NEW DESIGN #1 debug/terms
        "drop_box_target_gate": (np.ones((number_of_environments,), dtype=np.float32) * float(DROP_BOX_TARGET_GATE)).astype(np.float32),
        "in_drop_box_target_gate": in_drop_box_target_gate.astype(np.float32),
        "drop_box_move_delta_3d": drop_box_move_delta_3d.astype(np.float32),
        "drop_box_move_scale": (np.ones((number_of_environments,), dtype=np.float32) * float(drop_box_move_scale)).astype(np.float32),
        "drop_box_stability_score": drop_box_stability_score.astype(np.float32),
        "drop_box_stability_reward": drop_box_stability_reward.astype(np.float32),
        "reset_due_to_drop_box_outside_target_gate": reset_due_to_drop_box_outside_target_gate.astype(np.float32),

        # ✅ NEW: post-drop (<=K steps) cumulative movement stability reward
        "post_drop_move_reward_steps_K": (np.ones((number_of_environments,), dtype=np.float32) * float(POST_DROP_MOVE_REWARD_STEPS)).astype(np.float32),
        "post_drop_move_sum_3d": post_drop_move_sum_current.astype(np.float32),
        "post_drop_move_sum_scale": (np.ones((number_of_environments,), dtype=np.float32) * float(post_drop_move_sum_scale)).astype(np.float32),
        "post_drop_move_score": post_drop_move_score.astype(np.float32),
        "post_drop_move_check_at_k": post_drop_move_check_at_k.astype(np.float32),
        "post_drop_move_stability_reward": post_drop_move_stability_reward.astype(np.float32),

        # ✅ NEW: post-drop away-from-cube short-term reward
        "prev_dist_center_box(debug)": prev_fingertip_center_to_box_distance.astype(np.float32),
        "post_drop_away_delta_dist_3d": post_drop_away_delta_dist_3d.astype(np.float32),
        "post_drop_away_delta_lo": (np.ones((number_of_environments,), dtype=np.float32) * float(post_drop_away_delta_lo)).astype(np.float32),
        "post_drop_away_delta_hi": (np.ones((number_of_environments,), dtype=np.float32) * float(post_drop_away_delta_hi)).astype(np.float32),
        "post_drop_away_far_dist": (np.ones((number_of_environments,), dtype=np.float32) * float(post_drop_away_far_dist)).astype(np.float32),
        "post_drop_away_step13_mask": post_drop_away_step13_mask.astype(np.float32),
        "post_drop_away_step45_mask": post_drop_away_step45_mask.astype(np.float32),
        "post_drop_away_step13_reward": post_drop_away_step13_reward.astype(np.float32),
        "post_drop_away_step45_reward": post_drop_away_step45_reward.astype(np.float32),
        "post_drop_away_reward": post_drop_away_reward.astype(np.float32),

        # ✅ NEW DESIGN #2 debug/terms
        "box_to_target_xy_distance": box_to_target_xy_distance.astype(np.float32),
        "home_xy_bonus_scale": (np.ones((number_of_environments,), dtype=np.float32) * float(home_xy_bonus_scale)).astype(np.float32),
        "home_xy_score": home_xy_score.astype(np.float32),
        "home_xy_bonus_reward": home_xy_bonus_reward.astype(np.float32),

        # ✅ extra reward
        "extra_reward": extra_reward.astype(np.float32),

        "at_home": at_home.astype(np.float32),
        "settlement_penalty_negative": (-settlement_penalty).astype(np.float32),
        "stall_penalty_negative": (-stall_penalty).astype(np.float32),
        "timeout_penalty_negative": (-timeout_penalty).astype(np.float32),
        "safety_penalty": safety_penalty.astype(np.float32),

        "finger_gap_value": finger_gap_value.astype(np.float32),
        "drop_gap_threshold": (np.ones((number_of_environments,), dtype=np.float32) * float(DROP_GAP_THRESHOLD)).astype(np.float32),
        "has_dropped_after_lift": has_dropped_after_lift_current.astype(np.float32),
        "has_dropped_after_lift_just_now": has_dropped_after_lift_just_now.astype(np.float32),

        # ✅ NEW: drop->home target gate debug
        "drop_to_home_target_gate": (np.ones((number_of_environments,), dtype=np.float32) * float(DROP_TO_HOME_TARGET_GATE)).astype(np.float32),
        "in_drop_target_gate": in_drop_target_gate.astype(np.float32),
        "home_reward_armed": home_reward_armed_current.astype(np.float32),
        "reset_due_to_drop_not_in_target_gate": reset_due_to_drop_not_in_target_gate.astype(np.float32),

        # ✅ NEW: reset designs debug
        "steps_since_drop": steps_since_drop_current.astype(np.float32),
        "post_drop_close_reset_dist": (np.ones((number_of_environments,), dtype=np.float32) * float(POST_DROP_CLOSE_RESET_DIST)).astype(np.float32),
        "post_drop_close_reset_steps_K": (np.ones((number_of_environments,), dtype=np.float32) * float(POST_DROP_CLOSE_RESET_STEPS)).astype(np.float32),
        "post_drop_check_at_k": post_drop_check_at_k.astype(np.float32),
        "post_drop_far_reward": post_drop_far_reward.astype(np.float32),
        "post_drop_far_reward_value": (np.ones((number_of_environments,), dtype=np.float32) * float(POST_DROP_FAR_REWARD_VALUE)).astype(np.float32),

        "reset_due_to_post_drop_close": reset_due_to_post_drop_close.astype(np.float32),
        "reset_due_to_no_drop_far": reset_due_to_no_drop_far.astype(np.float32),
        "reset_episode_current": reset_episode_current.astype(np.float32),

        # ✅ NEW: orientation constraint debug
        "reach_success_dist_th": (np.ones((number_of_environments,), dtype=np.float32) * float(REACH_SUCCESS_DIST)).astype(np.float32),
        "reach_success_now": reach_success_now.astype(np.float32),
        "reach_success_achieved": reach_success_achieved_current.astype(np.float32),
        "gripper_cos_down": gripper_cos_down.astype(np.float32),
        "gripper_down_cos_min": (np.ones((number_of_environments,), dtype=np.float32) * float(gripper_down_cos_min)).astype(np.float32),
        "orientation_constraint_active": orientation_constraint_active.astype(np.float32),
        "reset_due_to_bad_gripper_orientation": reset_due_to_bad_gripper_orientation.astype(np.float32),
    }

    if np.any(invalid_mask):
        reward_terms["invalid_penalty"] = (
            float(getattr(reward_configuration, "invalid_penalty", -30.0)) * invalid_mask.astype(np.float32)
        ).astype(np.float32)

    metrics = {
        "fingertip_center_to_box_distance": fingertip_center_to_box_distance.astype(np.float32),
        "pinch_quality_score": pinch_quality_score.astype(np.float32),
        "grasp_approach_score": grasp_approach_score.astype(np.float32),
        "lift_score": lift_score.astype(np.float32),
        "lift_success_now": lift_success_now.astype(np.float32),
        "lift_success_achieved": lift_success_achieved_current.astype(np.float32),

        # ✅ NEW
        "lift_success_bonus_reward": lift_success_bonus_reward.astype(np.float32),

        "fingertip_center_to_target_distance_3d": fingertip_center_to_target_distance_3d.astype(np.float32),

        # ✅ NEW: min-distance progress metric（若未启用则只是记录）
        "target_best_distance_3d(debug)": (best_target_distance_current.astype(np.float32) if best_target_distance_current is not None else np.zeros((number_of_environments,), dtype=np.float32)),
        "target_reward_mode_is_min_dist(debug)": target_out.get(
            "target_progress_reward_mode_is_min_dist",
            np.zeros((number_of_environments,), dtype=np.float32),
        ).astype(np.float32),

        "end_effector_to_home_distance": end_effector_to_home_distance.astype(np.float32),
        "best_end_effector_to_home_distance(debug)": best_home_distance_current.astype(np.float32),

        "home_progress_active_mask": home_progress_active_mask.astype(np.float32),

        # ✅ NEW DESIGN #1 metrics
        "in_drop_box_target_gate": in_drop_box_target_gate.astype(np.float32),
        "drop_box_move_delta_3d": drop_box_move_delta_3d.astype(np.float32),
        "drop_box_stability_reward": drop_box_stability_reward.astype(np.float32),
        "reset_due_to_drop_box_outside_target_gate": reset_due_to_drop_box_outside_target_gate.astype(np.float32),

        # ✅ NEW metrics: post-drop cumulative movement stability reward
        "post_drop_move_sum_3d": post_drop_move_sum_current.astype(np.float32),
        "post_drop_move_score": post_drop_move_score.astype(np.float32),
        "post_drop_move_check_at_k": post_drop_move_check_at_k.astype(np.float32),
        "post_drop_move_stability_reward": post_drop_move_stability_reward.astype(np.float32),

        # ✅ NEW metrics: post-drop away-from-cube short-term reward
        "post_drop_away_delta_dist_3d": post_drop_away_delta_dist_3d.astype(np.float32),
        "post_drop_away_step13_reward": post_drop_away_step13_reward.astype(np.float32),
        "post_drop_away_step45_reward": post_drop_away_step45_reward.astype(np.float32),
        "post_drop_away_reward": post_drop_away_reward.astype(np.float32),

        # ✅ NEW DESIGN #2 metrics
        "box_to_target_xy_distance": box_to_target_xy_distance.astype(np.float32),
        "home_xy_bonus_reward": home_xy_bonus_reward.astype(np.float32),

        "extra_reward": extra_reward.astype(np.float32),

        "idle_steps_counter": idle_steps_counter_current.astype(np.float32),
        "stalled": stalled_mask.astype(np.float32),
        "attempt_failed_event": attempt_failed_event_mask.astype(np.float32),
        "settlement_penalty": settlement_penalty.astype(np.float32),
        "stall_penalty": stall_penalty.astype(np.float32),
        "timeout": timeout_mask.astype(np.float32),

        "finger_gap_value": finger_gap_value.astype(np.float32),
        "has_dropped_after_lift": has_dropped_after_lift_current.astype(np.float32),

        # ✅ NEW: drop->home target gate metrics
        "in_drop_target_gate": in_drop_target_gate.astype(np.float32),
        "home_reward_armed": home_reward_armed_current.astype(np.float32),
        "reset_due_to_drop_not_in_target_gate": reset_due_to_drop_not_in_target_gate.astype(np.float32),

        # ✅ NEW: reset designs metrics
        "steps_since_drop": steps_since_drop_current.astype(np.float32),
        "post_drop_check_at_k": post_drop_check_at_k.astype(np.float32),
        "post_drop_far_reward": post_drop_far_reward.astype(np.float32),
        "reset_due_to_post_drop_close": reset_due_to_post_drop_close.astype(np.float32),
        "reset_due_to_no_drop_far": reset_due_to_no_drop_far.astype(np.float32),
        "reset_episode_current": reset_episode_current.astype(np.float32),

        # ✅ NEW: orientation constraint metrics
        "reach_success_now": reach_success_now.astype(np.float32),
        "reach_success_achieved": reach_success_achieved_current.astype(np.float32),
        "orientation_constraint_active": orientation_constraint_active.astype(np.float32),
        "gripper_cos_down": gripper_cos_down.astype(np.float32),
        "gripper_down_cos_min": (np.ones((number_of_environments,), dtype=np.float32) * float(gripper_down_cos_min)).astype(np.float32),
        "reset_due_to_bad_gripper_orientation": reset_due_to_bad_gripper_orientation.astype(np.float32),
    }

    return total_reward, terminated, has_cube_now.astype(bool), reward_terms, metrics

import numpy as np

from motrix_envs.np.reward import tolerance
from .compute_common import _compute_common_terms
from .compute_reach_grasp import _compute_reward_reach_grasp


def _compute_reward_grasp_lift(env, data, info):
    num_envs = data.shape[0]
    cfg_r = env.cfg.reward_config

    c = _compute_common_terms(env, data, info)

    box_pos = c["box_pos"]
    box_height = c["box_height"].astype(np.float32)
    prev_box_height = c["prev_box_height"].astype(np.float32)

    dist_center_box = c["dist_fingertip_center_box"].astype(np.float32)

    dof_pos = c["dof_pos"]
    dof_vel = c["dof_vel"]

    left_force = c["left_force"].astype(np.float32)
    right_force = c["right_force"].astype(np.float32)
    fully_open = c["fully_open"]
    finger_gap_val = c["finger_gap_val"]

    has_cube_prev = c["has_cube_prev"].astype(bool)
    ever_had_cube_prev = c["ever_had_cube_prev"].astype(bool)

    steps = info.get(
        "steps_for_reward",
        info.get("steps", np.zeros(num_envs, dtype=np.int32)),
    ).astype(np.int32)

    stage1_reward, _, has_cube_next_raw, reward_terms_stage1, metrics_stage1 = (
        _compute_reward_reach_grasp(env, data, info)
    )
    has_cube_raw = has_cube_next_raw.astype(bool)

    stage1_grasp_success = reward_terms_stage1.get(
        "grasp_success", np.zeros(num_envs, dtype=np.float32)
    ).astype(np.float32)
    stage1_lift_success = reward_terms_stage1.get(
        "lift_success", np.zeros(num_envs, dtype=np.float32)
    ).astype(np.float32)

    stage1_big_bonus = (stage1_grasp_success + stage1_lift_success).astype(np.float32)
    stage1_shaping = (stage1_reward.astype(np.float32) - stage1_big_bonus).astype(np.float32)

    contact_soft = float(
        getattr(cfg_r, "contact_force_soft", getattr(cfg_r, "contact_force_grasp", 1.0) * 0.1)
    )
    release_dist = float(
        getattr(cfg_r, "stage2_release_dist", max(getattr(cfg_r, "grasp_dist_tol", 0.03) * 1.5, 0.06))
    )
    lost_steps = int(getattr(cfg_r, "stage2_lost_steps", 3))
    lost_steps = max(lost_steps, 1)

    lost_candidate = (left_force < contact_soft) & (right_force < contact_soft) & (
        dist_center_box > release_dist
    )
    u = np.zeros(num_envs, dtype=np.int32)
    lost_counter_prev = info.get("stage2_lost_counter", u).astype(np.int32)
    lost_counter = np.where(
        has_cube_prev & (~has_cube_raw) & lost_candidate,
        lost_counter_prev + 1,
        0,
    ).astype(np.int32)

    release_by_open = fully_open.astype(bool)
    release_by_loss = lost_counter >= lost_steps
    release = release_by_open | release_by_loss

    has_cube_now = has_cube_raw | (has_cube_prev & (~release))
    has_cube_next = has_cube_now.copy()

    ever_had_cube = np.logical_or(ever_had_cube_prev, has_cube_now)
    dropped_now = has_cube_prev & (~has_cube_now)

    lift_goal = float(getattr(cfg_r, "lift_height", 0.06))
    denom = max(lift_goal, 1e-6)

    h_norm = np.clip(box_height / denom, 0.0, 1.0)
    lift_height_scale = float(getattr(cfg_r, "lift_height_reward_scale", 6.0))
    lift_height_r = (
        lift_height_scale * h_norm.astype(np.float32) * has_cube_now.astype(np.float32)
    )

    delta_height = box_height - prev_box_height
    height_delta_pos = np.maximum(delta_height, 0.0)
    lift_delta_scale = float(getattr(cfg_r, "lift_height_delta_scale", 3.0))
    lift_delta_r = (
        lift_delta_scale
        * height_delta_pos.astype(np.float32)
        * has_cube_now.astype(np.float32)
    )

    success = has_cube_now & (box_height >= lift_goal)
    success_bonus = float(getattr(cfg_r, "success_bonus", 8.0)) * success.astype(np.float32)

    drop_penalty_scale = float(getattr(cfg_r, "stage2_drop_penalty_scale", 0.0))
    drop_penalty = drop_penalty_scale * dropped_now.astype(np.float32)

    stage2_time_penalty = float(getattr(cfg_r, "stage2_time_penalty", 0.0))
    time_penalty = stage2_time_penalty * np.ones(num_envs, dtype=np.float32)

    force_target_total = float(getattr(cfg_r, "stage2_force_target_total", 12.0))
    band_ratio = float(getattr(cfg_r, "stage2_force_target_band_ratio", 0.20))
    margin_ratio = float(getattr(cfg_r, "stage2_force_target_margin_ratio", 0.60))
    force_target_scale = float(getattr(cfg_r, "stage2_force_target_scale", 1.0))
    force_target_start_h = float(getattr(cfg_r, "stage2_force_target_start_height", 0.01))

    f_sum = np.clip(left_force, 0.0, None) + np.clip(right_force, 0.0, None)
    low = (1.0 - band_ratio) * force_target_total
    high = (1.0 + band_ratio) * force_target_total
    margin = max(margin_ratio * force_target_total, 1e-6)

    force_active = has_cube_now & (box_height >= force_target_start_h)
    force_mag = tolerance(f_sum, bounds=(low, high), margin=margin).astype(np.float32)
    force_target_r = (force_target_scale * force_mag * force_active.astype(np.float32)).astype(np.float32)

    mix_steps = max(int(getattr(cfg_r, "stage2_mix_steps", 300)), 1)

    grasp_step_prev = info.get("stage2_grasp_step", -np.ones(num_envs, dtype=np.int32)).astype(np.int32)
    new_grasp = has_cube_now & (~has_cube_prev)
    grasp_step = np.where(
        (grasp_step_prev < 0) & new_grasp,
        steps,
        grasp_step_prev,
    ).astype(np.int32)

    since_grasp = np.where(grasp_step >= 0, (steps - grasp_step).astype(np.float32), 0.0).astype(np.float32)
    progress = np.clip(since_grasp / float(mix_steps), 0.0, 1.0).astype(np.float32)

    w_stage1_start = float(getattr(cfg_r, "stage2_stage1_weight_start", 1.0))
    w_stage1_end = float(getattr(cfg_r, "stage2_stage1_weight_end", 0.0))
    w_lift_start = float(getattr(cfg_r, "stage2_lift_weight_start", 0.0))
    w_lift_end = float(getattr(cfg_r, "stage2_lift_weight_end", 1.0))

    w_stage1 = w_stage1_start + (w_stage1_end - w_stage1_start) * progress
    w_lift = w_lift_start + (w_lift_end - w_lift_start) * progress

    min_w_before_grasp = float(getattr(cfg_r, "stage2_min_stage1_weight_before_grasp", 1.0))
    w_stage1 = np.where(grasp_step < 0, np.maximum(w_stage1, min_w_before_grasp), w_stage1)
    w_lift = np.where(grasp_step < 0, 0.0, w_lift)

    lift_shaping = lift_height_r + lift_delta_r + success_bonus + force_target_r

    total_reward = (
        w_stage1 * stage1_shaping
        + w_lift * lift_shaping
        + drop_penalty
        + time_penalty
    ).astype(np.float32)

    terminated = np.zeros(num_envs, dtype=bool)

    terminate_on_success = bool(getattr(cfg_r, "stage2_terminate_on_success", True))
    terminate_on_drop = bool(getattr(cfg_r, "stage2_terminate_on_drop", True))

    if terminate_on_success:
        terminated |= success
    if terminate_on_drop:
        terminated |= dropped_now

    # ✅ 关键修复：默认 clip 不要 10.0
    max_abs = float(getattr(cfg_r, "max_step_reward", 200.0))
    invalid_reward = np.isnan(total_reward) | np.isinf(total_reward)
    total_reward = np.clip(total_reward, -max_abs, max_abs).astype(np.float32)

    invalid_state = np.isnan(dof_pos).any(axis=1) | np.isnan(dof_vel).any(axis=1)
    invalid = invalid_state | invalid_reward
    if np.any(invalid):
        terminated |= invalid
        invalid_penalty = float(getattr(cfg_r, "invalid_penalty", -30.0))
        total_reward[invalid] = invalid_penalty

    info["has_cube"] = has_cube_next.astype(bool)
    info["ever_had_cube"] = ever_had_cube.astype(bool)

    info["prev_box_xy"] = box_pos[:, :2].astype(np.float32)
    info["prev_box_height"] = box_height.astype(np.float32)
    info["prev_dist_ee_box"] = dist_center_box.astype(np.float32)

    info["stage2_grasp_step"] = grasp_step.astype(np.int32)
    info["stage2_lost_counter"] = lost_counter.astype(np.int32)

    reward_terms = {
        "stage1_total": stage1_reward.astype(np.float32),
        "stage1_bonus_removed": (-stage1_big_bonus).astype(np.float32),
        "stage1_shaping": stage1_shaping.astype(np.float32),
        "lift_height": lift_height_r.astype(np.float32),
        "lift_delta": lift_delta_r.astype(np.float32),
        "success_bonus": success_bonus.astype(np.float32),
        "force_target": force_target_r.astype(np.float32),
        "drop_penalty": drop_penalty.astype(np.float32),
        "time_penalty": time_penalty.astype(np.float32),
        "w_stage1": w_stage1.astype(np.float32),
        "w_lift": w_lift.astype(np.float32),
    }

    metrics = {
        "dist_ee_box": dist_center_box.astype(np.float32),
        "box_height": box_height.astype(np.float32),
        "finger_gap": finger_gap_val.astype(np.float32),
        "left_force": left_force.astype(np.float32),
        "right_force": right_force.astype(np.float32),
        "force_sum": f_sum.astype(np.float32),
        "force_target_mag": force_mag.astype(np.float32),
        "force_target_active": force_active.astype(np.float32),
        "has_cube_raw": has_cube_raw.astype(np.float32),
        "has_cube": has_cube_now.astype(np.float32),
        "ever_had_cube": ever_had_cube.astype(np.float32),
        "height_delta": delta_height.astype(np.float32),
        "stage1_reward": stage1_reward.astype(np.float32),
        "stage1_shaping": stage1_shaping.astype(np.float32),
        "lift_height_r": lift_height_r.astype(np.float32),
        "lift_delta_r": lift_delta_r.astype(np.float32),
        "success": success.astype(np.float32),
        "dropped": dropped_now.astype(np.float32),
        "mix_progress": progress.astype(np.float32),
        "w_stage1": w_stage1.astype(np.float32),
        "w_lift": w_lift.astype(np.float32),
        "stage2_lost_counter": lost_counter.astype(np.float32),
        "invalid": invalid.astype(np.float32),
    }

    return total_reward, terminated, has_cube_next, reward_terms, metrics

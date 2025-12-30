import numpy as np

from motrix_envs.np.reward import tolerance
from .compute_common import _compute_common_terms


def _compute_reward_with_phases(env, data, info, mode: str):
    num_envs = data.shape[0]
    cfg_r = env.cfg.reward_config

    c = _compute_common_terms(env, data, info)

    box_pos = c["box_pos"]
    target_pos = c["target_pos"]

    # 用指尖中点距离作为“抓取/推/靠近”的基准距离
    dist_ee_box = c["dist_fingertip_center_box"].astype(np.float32)

    dist_box_target_xy = c["dist_box_target_xy"]
    box_height = c["box_height"]
    dof_pos = c["dof_pos"]
    dof_vel = c["dof_vel"]
    finger_gap_val = c["finger_gap_val"]
    fully_closed = c["fully_closed"]
    fully_open = c["fully_open"]
    is_open = c["is_open"]
    is_closed = c["is_closed"]
    contact_force = c["contact_force"]
    orient_approach = c["orient_approach"]
    orient_carry = c["orient_carry"]
    prev_box_xy = c["prev_box_xy"]
    prev_box_height = c["prev_box_height"]
    has_cube_prev = c["has_cube_prev"].astype(bool)
    ever_had_cube_prev = c["ever_had_cube_prev"].astype(bool)
    left_force = c["left_force"]
    right_force = c["right_force"]

    # 接近 shaping（带衰减）
    sigma = max(cfg_r.approach_sigma, 1e-6)
    approach_r = np.exp(
        -(dist_ee_box**2) / (2.0 * sigma * sigma)
    ).astype(np.float32)
    steps = info.get("steps", np.zeros(num_envs, dtype=np.int32))
    fade_steps = max(int(getattr(cfg_r, "approach_fadeout_steps", 0)), 0)
    if fade_steps > 0:
        coeff = np.clip(
            1.0 - steps.astype(np.float32) / float(fade_steps),
            0.0,
            1.0,
        )
    else:
        coeff = np.ones_like(approach_r, dtype=np.float32)
    approach_r *= coeff

    far_mask = dist_ee_box > cfg_r.reach_radius
    open_pref_r = (
        cfg_r.open_far_reward * is_open
        - cfg_r.close_far_penalty * is_closed
    ).astype(np.float32)
    open_pref_r *= far_mask.astype(np.float32)

    orient_app_r = (
        cfg_r.orient_app_scale * orient_approach
    ).astype(np.float32)
    orient_carry_r = (
        cfg_r.orient_carry_scale * orient_carry
    ).astype(np.float32)

    time_penalty = cfg_r.time_penalty * np.ones(num_envs, dtype=np.float32)

    # --------------------- has_cube：sticky + release --------------------- #
    close_enough = dist_ee_box <= cfg_r.grasp_dist_tol
    lifted_enough = box_height >= cfg_r.grasp_height_delta

    left_strong = left_force >= cfg_r.contact_force_grasp
    right_strong = right_force >= cfg_r.contact_force_grasp
    both_strong = left_strong & right_strong

    finger_in_range = (
        (finger_gap_val >= 0.25 * env._finger_open_val)
        & (finger_gap_val <= 0.75 * env._finger_open_val)
    )

    has_cube_cond = both_strong & lifted_enough & close_enough & finger_in_range

    # release：明显张开，或者软接触全丢 + 远离 + 贴近桌面
    contact_soft = float(
        getattr(cfg_r, "contact_force_soft", cfg_r.contact_force_grasp * 0.1)
    )
    near_table = box_height <= (cfg_r.push_height_eps + float(getattr(cfg_r, "drop_height_eps", 0.02)))
    release_dist = float(getattr(cfg_r, "release_dist_tol", max(cfg_r.grasp_dist_tol * 1.5, 0.06)))
    lost_soft = (left_force < contact_soft) & (right_force < contact_soft)

    release = fully_open | (lost_soft & (dist_ee_box > release_dist) & near_table)

    has_cube_now = has_cube_cond | (has_cube_prev & (~release))
    has_cube_next = has_cube_now.copy()
    ever_had_cube = np.logical_or(ever_had_cube_prev, has_cube_now)

    new_grasp = has_cube_now & (~has_cube_prev)
    grasp_bonus = cfg_r.grasp_bonus * new_grasp.astype(np.float32)
    hold_bonus = cfg_r.hold_bonus * has_cube_now.astype(np.float32)

    # ---------------------- 搬运：抬高 + 向目标移动 ---------------------- #
    lift_goal = float(cfg_r.lift_height)
    lift_r = tolerance(
        box_height,
        bounds=(lift_goal, np.inf),
        margin=max(lift_goal, 1e-6),
        sigmoid="gaussian",
    ).astype(np.float32)
    lift_r *= has_cube_now.astype(np.float32)

    carry_r = tolerance(
        dist_box_target_xy,
        bounds=(0.0, 0.0),
        margin=cfg_r.reach_radius,
        sigmoid="gaussian",
    ).astype(np.float32)
    carry_r *= has_cube_now.astype(np.float32)

    orient_carry_r *= has_cube_now.astype(np.float32)

    near_target_xy = dist_box_target_xy <= cfg_r.success_tolerance

    phase_approach = ~has_cube_now
    phase_carry = has_cube_now & (~near_target_xy)
    phase_place = has_cube_now & near_target_xy

    open_place_mask = near_target_xy & near_table
    open_place_r = (
        cfg_r.open_place_reward
        * is_open
        * open_place_mask.astype(np.float32)
    )

    # ---------------------- 推动惩罚 / 掉落 / 成功 ---------------------- #
    box_xy = box_pos[:, :2]
    box_xy_delta = box_xy - prev_box_xy
    box_xy_speed = np.linalg.norm(box_xy_delta, axis=1)

    push_candidates = (box_height <= cfg_r.push_height_eps) & (
        dist_ee_box <= cfg_r.push_near_xy
    )
    pushing = push_candidates & (~fully_open) & (
        box_xy_speed >= cfg_r.push_speed_threshold
    )
    push_penalty = (
        -cfg_r.push_penalty_scale
        * pushing.astype(np.float32)
        * (box_xy_speed / (cfg_r.push_speed_threshold + 1e-6))
    )

    # 即时 drop：上一帧持有 -> 当前不持有
    dropped_now = has_cube_prev & (~has_cube_now)

    dropped_penalty_mask = dropped_now & (~near_target_xy)
    drop_penalty = cfg_r.drop_penalty * dropped_penalty_mask.astype(np.float32)

    success = np.zeros(num_envs, dtype=bool)
    success_bonus = np.zeros(num_envs, dtype=np.float32)

    if mode == "reach_grasp_transport":
        success = has_cube_now & near_target_xy & (box_height >= lift_goal)
        success_bonus = cfg_r.success_bonus * success.astype(np.float32)
    elif mode == "full_pick_place":
        success = (
            ever_had_cube_prev & (~has_cube_now) & near_table & near_target_xy
        )
        success_bonus = cfg_r.success_bonus * success.astype(np.float32)

    # ---------------------- 双指受力 + 抬高激励（保留） ---------------------- #
    contact_soft2 = getattr(
        cfg_r, "contact_force_soft", cfg_r.contact_force_grasp * 0.1
    )
    left_contact = left_force >= contact_soft2
    right_contact = right_force >= contact_soft2
    both_contact = left_contact & right_contact

    delta_height = box_height - prev_box_height
    height_up = np.maximum(delta_height, 0.0)

    both_abs = (
        cfg_r.both_contact_lift_scale
        * both_contact.astype(np.float32)
        * np.maximum(box_height, 0.0)
    )
    both_delta = (
        cfg_r.both_contact_lift_delta_scale
        * both_contact.astype(np.float32)
        * height_up
    )

    # ---------------------- 距离增长惩罚 ---------------------- #
    prev_dist_box_target_xy = info.get(
        "prev_dist_box_target_xy", dist_box_target_xy.copy()
    )
    dist_delta = dist_box_target_xy - prev_dist_box_target_xy
    dist_inc = np.maximum(dist_delta, 0.0)
    dist_inc_scale = getattr(cfg_r, "dist_increase_penalty_scale", 10.0)
    dist_inc_mask = (has_cube_now).astype(np.float32)
    dist_increase_penalty = (
        -dist_inc_scale * dist_inc * dist_inc_mask
    ).astype(np.float32)

    total_reward = (
        cfg_r.approach_scale * approach_r
        + orient_app_r * phase_approach.astype(np.float32)
        + orient_carry_r
        + open_pref_r
        + time_penalty
        + grasp_bonus
        + hold_bonus
        + cfg_r.carry_scale * (lift_r + carry_r)
        + cfg_r.place_scale * open_place_r
        + both_abs
        + both_delta
        + push_penalty
        + drop_penalty
        + dist_increase_penalty
        + success_bonus
    )

    invalid_reward = np.isnan(total_reward) | np.isinf(total_reward)

    max_abs = getattr(cfg_r, "max_step_reward", 10.0)
    total_reward = np.clip(total_reward, -max_abs, max_abs).astype(np.float32)

    info["has_cube"] = has_cube_next
    info["ever_had_cube"] = ever_had_cube
    info["prev_dist_ee_box"] = dist_ee_box.astype(np.float32)
    info["prev_dist_box_target_xy"] = dist_box_target_xy.astype(np.float32)
    info["prev_box_xy"] = box_xy.astype(np.float32)
    info["prev_box_height"] = box_height.astype(np.float32)

    reward_terms = {
        "approach": cfg_r.approach_scale * approach_r,
        "orient_app": orient_app_r * phase_approach.astype(np.float32),
        "orient_carry": orient_carry_r,
        "open_pref": open_pref_r,
        "time_penalty": time_penalty,
        "grasp_bonus": grasp_bonus,
        "hold_bonus": hold_bonus,
        "lift": cfg_r.carry_scale * lift_r,
        "carry": cfg_r.carry_scale * carry_r,
        "open_place": cfg_r.place_scale * open_place_r,
        "both_contact_abs": both_abs,
        "both_contact_delta": both_delta,
        "push_penalty": push_penalty,
        "drop_penalty": drop_penalty,
        "dist_increase_penalty": dist_increase_penalty,
        "success_bonus": success_bonus,
    }

    metrics = {
        "dist_ee_box": dist_ee_box,
        "dist_box_target_xy": dist_box_target_xy,
        "box_height": box_height,
        "finger_gap": finger_gap_val,
        "is_open": is_open,
        "is_closed": is_closed,
        "contact_force": contact_force,
        "has_cube": has_cube_now.astype(np.float32),
        "ever_had_cube": ever_had_cube.astype(np.float32),
        "phase_approach": phase_approach.astype(np.float32),
        "phase_carry": phase_carry.astype(np.float32),
        "phase_place": phase_place.astype(np.float32),
        "box_xy_speed": box_xy_speed,
        "success": success.astype(np.float32),
        "dropped": dropped_now.astype(np.float32),
    }

    invalid = np.isnan(dof_pos).any(axis=1) | np.isnan(dof_vel).any(axis=1)
    invalid |= invalid_reward
    terminated = np.zeros(num_envs, dtype=bool)
    if np.any(invalid):
        terminated |= invalid
        invalid_penalty = getattr(cfg_r, "invalid_penalty", -30.0)
        total_reward[invalid] = invalid_penalty
        reward_terms["invalid_penalty"] = (
            invalid_penalty * invalid.astype(np.float32)
        )
        metrics["invalid"] = invalid.astype(np.float32)

    return total_reward, terminated, has_cube_next, reward_terms, metrics

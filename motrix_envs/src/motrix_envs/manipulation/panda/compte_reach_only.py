import numpy as np

# -------------------------- reach_only 模式 -------------------------- #


def _compute_reward_reach_only(env, data, info):
    num_envs = data.shape[0]
    cfg_r = env.cfg.reward_config

    box_pos = env.get_box_pos(data)

    # 用“指尖中点”而不是 ee_site，减少几何偏差
    left_tip, right_tip, center_tip = env.get_fingertip_positions(data)
    grip_center_pos = center_tip

    dist_ee_box = np.linalg.norm(grip_center_pos - box_pos, axis=1)

    sigma = max(cfg_r.approach_sigma, 1e-6)
    approach_r = np.exp(-(dist_ee_box ** 2) / (2.0 * sigma * sigma)).astype(
        np.float32
    )

    dof_pos = env.get_dof_pos(data)
    dof_vel = env.get_dof_vel(data)
    finger_gap_val = env.get_finger_gap(data)

    fully_closed = finger_gap_val <= env._finger_closed_threshold
    fully_open = finger_gap_val >= env._finger_open_val * 0.9
    is_open = fully_open.astype(np.float32)
    is_closed = fully_closed.astype(np.float32)

    open_pref_r = (
        1.0 * is_open
        - 1.0 * is_closed
    ).astype(np.float32)

    # 姿态仍用 ee_site（末端自身 z 轴）
    site = env._model.get_site(env.cfg.asset.ee_site_name)
    ee_pose_full = site.get_pose(data)
    ee_quat = ee_pose_full[:, 3:]

    qw = ee_quat[:, 0]
    qx = ee_quat[:, 1]
    qy = ee_quat[:, 2]
    qz = ee_quat[:, 3]

    ee_z_world_x = 2.0 * (qx * qz + qw * qy)
    ee_z_world_y = 2.0 * (qy * qz - qw * qx)
    ee_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    ee_z_world = np.stack([ee_z_world_x, ee_z_world_y, ee_z_world_z], axis=1)

    world_down = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cos_down = np.clip(
        np.sum(ee_z_world * world_down[None, :], axis=1), -1.0, 1.0
    )
    orient_approach = (cos_down + 1.0) * 0.5
    orient_r = (1.0 * orient_approach).astype(np.float32)

    time_penalty = cfg_r.time_penalty * np.ones(num_envs, dtype=np.float32)

    total_reward = (
        cfg_r.approach_scale * approach_r
        + orient_r
        + open_pref_r
        + time_penalty
    )

    invalid_reward = np.isnan(total_reward) | np.isinf(total_reward)

    max_abs = getattr(cfg_r, "max_step_reward", 10.0)
    total_reward = np.clip(total_reward, -max_abs, max_abs).astype(np.float32)

    has_cube_next = np.zeros(num_envs, dtype=bool)
    info["has_cube"] = has_cube_next
    info["ever_had_cube"] = np.zeros(num_envs, dtype=bool)

    terminated = np.zeros(num_envs, dtype=bool)

    # 给后续 stage 统一：更新 prev_dist_ee_box（这里的“ee”即指尖中点距离）
    info["prev_dist_ee_box"] = dist_ee_box.astype(np.float32)

    reward_terms = {
        "approach": cfg_r.approach_scale * approach_r,
        "orient": orient_r,
        "open_pref": open_pref_r,
        "time_penalty": time_penalty,
    }

    metrics = {
        "dist_ee_box": dist_ee_box,
        "finger_gap": finger_gap_val,
        "is_open": is_open,
        "is_closed": is_closed,
    }

    invalid = np.isnan(dof_pos).any(axis=1) | np.isnan(dof_vel).any(axis=1)
    invalid |= invalid_reward
    if np.any(invalid):
        terminated |= invalid
        invalid_penalty = getattr(cfg_r, "invalid_penalty", -30.0)
        total_reward[invalid] = invalid_penalty
        reward_terms["invalid_penalty"] = (
            invalid_penalty * invalid.astype(np.float32)
        )
        metrics["invalid"] = invalid.astype(np.float32)

    return total_reward, terminated, has_cube_next, reward_terms, metrics

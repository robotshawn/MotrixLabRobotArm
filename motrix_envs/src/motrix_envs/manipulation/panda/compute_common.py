import numpy as np

from motrix_envs.np.reward import tolerance  # kept for backward-compat imports (may be unused)


def _safe_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize last-dim vectors safely. v: (..., 3)"""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def compute_gripper_orientation_world(
    model,
    ee_site_name: str,
    data,
    fingertip_center_pos_world: np.ndarray | None,
    eps: float = 1e-9,
):
    """
    统一的“夹爪朝向(世界系)”计算入口：
    **直接使用几何定义**：gripper_axis_world = normalize(tip_center - ee_site_pos)

    返回 dict（np.float32）：
      - ee_pose_full: (N,7) [x,y,z,qw,qx,qy,qz]  # quat 只作为原始数据透出，不参与计算
      - ee_site_pos: (N,3)
      - gripper_axis_world: (N,3)  # normalize(tip_center - ee_site_pos)
      - cos_down / cos_up: (N,)
      - orient_approach: (N,)  = (cos_down+1)/2
      - orient_carry:    (N,)  = (cos_up+1)/2
    """
    site = model.get_site(ee_site_name)
    ee_pose_full = site.get_pose(data)
    ee_pose_full = np.atleast_2d(ee_pose_full).astype(np.float32)  # (N,7)
    ee_site_pos = ee_pose_full[:, :3].astype(np.float32)  # (N,3)

    # tip center
    if fingertip_center_pos_world is None:
        # 极少数情况下拿不到 tip_center：给一个保守 fallback（不参与你主流程时也不会触发）
        gripper_axis_world = np.tile(
            np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (ee_site_pos.shape[0], 1)
        )
    else:
        tip_center = np.asarray(fingertip_center_pos_world, dtype=np.float32)
        tip_center = np.atleast_2d(tip_center)

        # 尽量兜底 N 维度
        if tip_center.shape[0] != ee_site_pos.shape[0]:
            if tip_center.shape[0] == 1 and ee_site_pos.shape[0] > 1:
                tip_center = np.repeat(tip_center, ee_site_pos.shape[0], axis=0)
            else:
                tip_center = tip_center.reshape(ee_site_pos.shape[0], 3)

        gripper_axis_world = _safe_normalize(tip_center - ee_site_pos, eps=eps).astype(np.float32)

    world_down = np.array([0.0, 0.0, -1.0], dtype=np.float32)[None, :]
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)[None, :]

    cos_down = np.clip(np.sum(gripper_axis_world * world_down, axis=1), -1.0, 1.0).astype(
        np.float32
    )
    cos_up = np.clip(np.sum(gripper_axis_world * world_up, axis=1), -1.0, 1.0).astype(np.float32)

    orient_approach = ((cos_down + 1.0) * 0.5).astype(np.float32)  # 越接近“朝下”越大
    orient_carry = ((cos_up + 1.0) * 0.5).astype(np.float32)  # 越接近“朝上”越大

    return {
        "ee_pose_full": ee_pose_full,
        "ee_site_pos": ee_site_pos,
        "gripper_axis_world": gripper_axis_world,
        "cos_down": cos_down,
        "cos_up": cos_up,
        "orient_approach": orient_approach,
        "orient_carry": orient_carry,
    }


def _compute_common_terms(env, data, info):
    """
    在所有 task_mode 下都会用到的一些公共量。
    统一加入“爪尖位置”相关几何量。
    注意：box_height / prev_box_height 一律使用“相对 env._box_z0 的高度增量”。

    ✅ 距离统一：
      - box->target 仅保留 3D 距离（dist_box_target_3d / prev_dist_box_target_3d）
      - 删除 box->target 的 2D/XY 距离项（避免 3D→XY 的敏感性与奖励尺度不一致）
    """
    num_envs = data.shape[0]

    # ---------------------- 位置几何 ---------------------- #
    box_pos = env.get_box_pos(data)  # (N, 3)
    ee_pos = env.get_ee_pos(data)  # (N, 3)
    target_pos = info["target_pos"]  # (N, 3)

    dist_ee_box = np.linalg.norm(ee_pos - box_pos, axis=1).astype(np.float32)
    dist_box_target_3d = np.linalg.norm(box_pos - target_pos, axis=1).astype(np.float32)

    # 相对高度（增量）
    box_height = (box_pos[:, 2] - env._box_z0).astype(np.float32)

    # ---------------------- 爪尖位置 ---------------------- #
    left_tip_pos, right_tip_pos, fingertip_center_pos = env.get_fingertip_positions(data)

    dist_fingertip_center_box = np.linalg.norm(fingertip_center_pos - box_pos, axis=1).astype(
        np.float32
    )
    dist_left_fingertip_box = np.linalg.norm(left_tip_pos - box_pos, axis=1).astype(np.float32)
    dist_right_fingertip_box = np.linalg.norm(right_tip_pos - box_pos, axis=1).astype(np.float32)

    # ---------------------- 关节和手指 ---------------------- #
    dof_pos = env.get_dof_pos(data)
    dof_vel = env.get_dof_vel(data)
    finger_gap_val = env.get_finger_gap(data)

    fully_closed = finger_gap_val <= env._finger_closed_threshold
    fully_open = finger_gap_val >= env._finger_open_val * 0.9
    is_open = fully_open.astype(np.float32)
    is_closed = fully_closed.astype(np.float32)

    # ---------------------- 手指接触力 ---------------------- #
    finger_forces = env.get_finger_forces(data)
    finger_forces = np.asarray(finger_forces, dtype=np.float32)
    finger_forces = np.atleast_2d(finger_forces)

    # 兼容某些后端返回 (2,N)
    if (
        finger_forces.shape[1] != 2
        and finger_forces.ndim == 2
        and finger_forces.shape[0] == 2
        and finger_forces.shape[1] == num_envs
    ):
        finger_forces = finger_forces.T

    if finger_forces.shape[1] != 2:
        finger_forces = finger_forces.reshape(-1, 2)

    left_force = finger_forces[:, 0].astype(np.float32)
    right_force = finger_forces[:, 1].astype(np.float32)
    contact_force = np.linalg.norm(finger_forces, axis=1).astype(np.float32)

    # ---------------------- 姿态（统一调用同一个函数） ---------------------- #
    ori = compute_gripper_orientation_world(
        model=env._model,
        ee_site_name=env.cfg.asset.ee_site_name,
        data=data,
        fingertip_center_pos_world=fingertip_center_pos,
    )
    orient_approach = ori["orient_approach"]
    orient_carry = ori["orient_carry"]

    # ---------------------- 上一帧 ---------------------- #
    prev_dist_ee_box = info.get("prev_dist_ee_box", dist_ee_box.copy())
    prev_dist_box_target_3d = info.get("prev_dist_box_target_3d", dist_box_target_3d.copy())
    prev_box_xy = info.get("prev_box_xy", box_pos[:, :2].copy())
    prev_box_height = info.get("prev_box_height", box_height.copy())

    has_cube_prev = info.get("has_cube", np.zeros(num_envs, dtype=bool))
    ever_had_cube_prev = info.get("ever_had_cube", np.zeros(num_envs, dtype=bool))

    return {
        "box_pos": box_pos,
        "ee_pos": ee_pos,
        "target_pos": target_pos,
        "dist_ee_box": dist_ee_box,
        "dist_box_target_3d": dist_box_target_3d,
        "box_height": box_height,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "finger_gap_val": finger_gap_val,
        "fully_closed": fully_closed,
        "fully_open": fully_open,
        "is_open": is_open,
        "is_closed": is_closed,
        "finger_forces": finger_forces,
        "left_force": left_force,
        "right_force": right_force,
        "contact_force": contact_force,
        # 统一的朝向奖励量
        "orient_approach": orient_approach,
        "orient_carry": orient_carry,
        # 额外返回：便于 debug / reward 扩展（不做四元数推导，仅透出 ee_site_pos 与 axis）
        "ee_pose_full": ori["ee_pose_full"],
        "ee_site_pos": ori["ee_site_pos"],
        "gripper_axis_world": ori["gripper_axis_world"],
        "cos_gripper_down": ori["cos_down"],
        "cos_gripper_up": ori["cos_up"],
        "prev_dist_ee_box": prev_dist_ee_box,
        "prev_dist_box_target_3d": prev_dist_box_target_3d,
        "prev_box_xy": prev_box_xy,
        "prev_box_height": prev_box_height,
        "has_cube_prev": has_cube_prev,
        "ever_had_cube_prev": ever_had_cube_prev,
        # ---- 爪尖相关 ----
        "left_fingertip_pos": left_tip_pos,
        "right_fingertip_pos": right_tip_pos,
        "fingertip_center_pos": fingertip_center_pos,
        "dist_fingertip_center_box": dist_fingertip_center_box,
        "dist_left_fingertip_box": dist_left_fingertip_box,
        "dist_right_fingertip_box": dist_right_fingertip_box,
    }

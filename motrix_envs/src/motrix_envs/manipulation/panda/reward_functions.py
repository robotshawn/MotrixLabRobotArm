import numpy as np
from motrix_envs.np.reward import tolerance

# ✅ 统一调用 compute_common.py 里的关键函数（同一真相源）
# 新版 compute_gripper_orientation_world 使用几何定义：
#   gripper_axis_world = normalize(tip_center - ee_site_pos)
from .compute_common import compute_gripper_orientation_world


# ----------------------------- small helpers -----------------------------
def _getf(cfg_r, key, default, old_key=None):
    """float getattr with optional backward-compatible fallback."""
    if old_key is None:
        return float(getattr(cfg_r, key, default))
    return float(getattr(cfg_r, key, getattr(cfg_r, old_key, default)))


def _geti(cfg_r, key, default, old_key=None):
    if old_key is None:
        return int(getattr(cfg_r, key, default))
    return int(getattr(cfg_r, key, getattr(cfg_r, old_key, default)))


def _getb(cfg_r, key, default, old_key=None):
    if old_key is None:
        return bool(getattr(cfg_r, key, default))
    return bool(getattr(cfg_r, key, getattr(cfg_r, old_key, default)))


def _ones_like(x):
    return np.ones_like(x, dtype=np.float32)


def _zeros_like(x):
    return np.zeros_like(x, dtype=np.float32)


def _safe_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize last-dim vectors safely. v: (..., 3)"""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


# ----------------------------- orientation (unified) -----------------------------
def _resolve_orient_approach(
    orient_approach=None,
    *,
    gripper_axis_world=None,
    dist_center_box=None,
    env=None,
    data=None,
    model=None,
    ee_site_name=None,
    fingertip_center_pos_world=None,
):
    """
    统一解析 “orient_approach(夹爪朝下程度)” 的来源，保持 backward-compatible。
    """
    if dist_center_box is None:
        if orient_approach is not None:
            dummy_like = np.asarray(orient_approach, dtype=np.float32)
        elif gripper_axis_world is not None:
            dummy_like = np.asarray(np.asarray(gripper_axis_world)[..., 0], dtype=np.float32)
        else:
            dummy_like = np.array([0.0], dtype=np.float32)
    else:
        dummy_like = np.asarray(dist_center_box, dtype=np.float32)

    # 1) 最优先：已有 orient_approach
    if orient_approach is not None:
        return np.asarray(orient_approach, dtype=np.float32), dummy_like

    # 2) 次优先：已有 gripper_axis_world
    if gripper_axis_world is not None:
        g = np.asarray(gripper_axis_world, dtype=np.float32)
        g = np.atleast_2d(g)
        g = _safe_normalize(g)
        world_down = np.array([0.0, 0.0, -1.0], dtype=np.float32)[None, :]
        cos_down = np.clip(np.sum(g * world_down, axis=1), -1.0, 1.0).astype(np.float32)
        oa = ((cos_down + 1.0) * 0.5).astype(np.float32)
        return oa, dummy_like

    # 3) 最后：从 env/data 或 model/site/data 计算
    if fingertip_center_pos_world is None and (env is not None) and (data is not None):
        try:
            _, _, ctip = env.get_fingertip_positions(data)
            fingertip_center_pos_world = ctip
        except Exception:
            fingertip_center_pos_world = None

    if env is not None and data is not None:
        try:
            model = env._model if model is None else model
            ee_site_name = env.cfg.asset.ee_site_name if ee_site_name is None else ee_site_name
        except Exception:
            pass

    if fingertip_center_pos_world is None:
        return None, dummy_like

    if (model is not None) and (data is not None) and (ee_site_name is not None):
        ori = compute_gripper_orientation_world(
            model=model,
            ee_site_name=ee_site_name,
            data=data,
            fingertip_center_pos_world=fingertip_center_pos_world,
        )
        return ori["orient_approach"].astype(np.float32), dummy_like

    return None, dummy_like


# ----------------------------- orientation -----------------------------
def orient_upright_bonus_and_gate(
    cfg_r,
    orient_approach,
    dist_center_box=None,
    *,
    gripper_axis_world=None,
    env=None,
    data=None,
    model=None,
    ee_site_name=None,
    fingertip_center_pos_world=None,
):
    """
    返回 (upright_bonus, orient_gate)
    - orient_gate: 姿态门控（全程可用）
    - upright_bonus: 兼容旧行为（按 dist 给弱加分）；新版推荐用 orient_upright_bonus_near_capped()
    """
    orient_approach, dummy_like = _resolve_orient_approach(
        orient_approach,
        gripper_axis_world=gripper_axis_world,
        dist_center_box=dist_center_box,
        env=env,
        data=data,
        model=model,
        ee_site_name=ee_site_name,
        fingertip_center_pos_world=fingertip_center_pos_world,
    )

    if orient_approach is None:
        return _zeros_like(dummy_like), _ones_like(dummy_like)

    orient_approach = orient_approach.astype(np.float32)
    if dist_center_box is None:
        dist_center_box = np.zeros_like(orient_approach, dtype=np.float32)
    else:
        dist_center_box = dist_center_box.astype(np.float32)

    orient_min = _getf(cfg_r, "orient_min_rwdfunc", 0.35, old_key="stage1_orient_min")
    orient_min = float(np.clip(orient_min, 0.0, 1.0))
    orient_gate = tolerance(
        orient_approach,
        bounds=(orient_min, 1.0),
        margin=max(1.0 - orient_min, 1e-6),
    ).astype(np.float32)

    upright_scale = _getf(cfg_r, "upright_bonus_scale_rwdfunc", 0.03)
    near_dist = _getf(cfg_r, "upright_bonus_near_dist_rwdfunc", 0.06)
    near_w = _getf(cfg_r, "upright_bonus_near_weight_rwdfunc", 1.5)
    w = np.where(dist_center_box <= near_dist, near_w, 1.0).astype(np.float32)

    upright_bonus = (upright_scale * orient_gate * w).astype(np.float32)
    return upright_bonus, orient_gate


def orient_upright_bonus_near_capped(
    cfg_r,
    dist_center_box,
    orient_gate,
    pinch_active,
    grasp_pre_ready,
    lift_phase,
    counter_prev=None,
):
    """
    新版 upright bonus：仅近场/阶段性 + 可封顶
    返回 (upright_bonus, counter)
    """
    dist_center_box = dist_center_box.astype(np.float32)
    orient_gate = orient_gate.astype(np.float32)
    pinch_active = pinch_active.astype(bool)
    grasp_pre_ready = grasp_pre_ready.astype(bool)
    lift_phase = lift_phase.astype(bool)

    near = _getf(cfg_r, "upright_bonus_near_dist_rwdfunc", 0.06)
    scale = _getf(cfg_r, "upright_bonus_scale_near_rwdfunc", 0.012)
    # 每 episode 最多给 M 步（0=不封顶）
    cap_steps = _geti(cfg_r, "upright_bonus_cap_steps_rwdfunc", 0)

    if counter_prev is None:
        counter_prev = np.zeros_like(dist_center_box, dtype=np.int32)
    else:
        counter_prev = counter_prev.astype(np.int32)

    allow_stage = (dist_center_box < near) & (~lift_phase) & ((~pinch_active) | (~grasp_pre_ready))
    if cap_steps > 0:
        allow_stage = allow_stage & (counter_prev < cap_steps)

    bonus = (scale * orient_gate * allow_stage.astype(np.float32)).astype(np.float32)
    counter = (counter_prev + allow_stage.astype(np.int32)).astype(np.int32)
    return bonus, counter


def orient_penalty_near(
    cfg_r,
    dist_center_box,
    orient_approach,
    pregrasp_dist,
    *,
    gripper_axis_world=None,
    env=None,
    data=None,
    model=None,
    ee_site_name=None,
    fingertip_center_pos_world=None,
):
    """近距离坏姿态惩罚（备用项）"""
    dist_center_box = dist_center_box.astype(np.float32)

    orient_approach, dummy_like = _resolve_orient_approach(
        orient_approach,
        gripper_axis_world=gripper_axis_world,
        dist_center_box=dist_center_box,
        env=env,
        data=data,
        model=model,
        ee_site_name=ee_site_name,
        fingertip_center_pos_world=fingertip_center_pos_world,
    )
    if orient_approach is None:
        return _zeros_like(dummy_like)

    orient_approach = orient_approach.astype(np.float32)

    orient_penalty_scale = _getf(cfg_r, "orient_penalty_scale_rwdfunc", 0.15, old_key="stage1_orient_penalty_scale")
    orient_penalty_th = _getf(cfg_r, "orient_penalty_th_rwdfunc", 0.20, old_key="stage1_orient_penalty_threshold")
    orient_penalty_dist = _getf(cfg_r, "orient_penalty_dist_rwdfunc", pregrasp_dist, old_key="stage1_orient_penalty_dist")

    bad = (orient_approach < orient_penalty_th) & (dist_center_box <= orient_penalty_dist)
    penalty = (-orient_penalty_scale * (orient_penalty_th - orient_approach) * bad.astype(np.float32)).astype(np.float32)
    return penalty


# ----------------------------- distance shaping -----------------------------
def pregrasp_distance_reward(cfg_r, dist_center_box, orient_gate=None):
    """pregrasp_dist 距离奖励：返回 pregrasp_r"""
    dist_center_box = dist_center_box.astype(np.float32)
    if orient_gate is None:
        orient_gate = _ones_like(dist_center_box)
    else:
        orient_gate = orient_gate.astype(np.float32)

    pregrasp_scale = _getf(cfg_r, "pregrasp_scale_rwdfunc", 0.05, old_key="stage1_pregrasp_reward_scale")
    pregrasp_dist = _getf(cfg_r, "pregrasp_dist_rwdfunc", 0.06, old_key="stage1_pregrasp_dist")
    pregrasp_margin = _getf(cfg_r, "pregrasp_margin_rwdfunc", 0.25, old_key="stage1_pregrasp_margin")

    r = (pregrasp_scale * tolerance(
        dist_center_box,
        bounds=(0.0, pregrasp_dist),
        margin=pregrasp_margin
    ) * orient_gate).astype(np.float32)
    return r


def approach_away_delta_rewards(cfg_r, dist_center_box, prev_dist, orient_gate=None):
    """delta_dist shaping：返回 (approach_r, away_penalty, raw_delta)"""
    dist_center_box = dist_center_box.astype(np.float32)
    prev_dist = prev_dist.astype(np.float32)

    if orient_gate is None:
        orient_gate = _ones_like(dist_center_box)
    else:
        orient_gate = orient_gate.astype(np.float32)

    approach_scale = _getf(cfg_r, "approach_scale_rwdfunc", 0.1, old_key="stage1_approach_reward_scale")
    away_scale = _getf(cfg_r, "away_scale_rwdfunc", 0.1, old_key="stage1_away_penalty_scale")
    delta_clip = _getf(cfg_r, "delta_dist_clip_rwdfunc", 0.01, old_key="stage1_delta_dist_clip")

    valid_prev = prev_dist > 0.0
    raw_delta = np.where(valid_prev, prev_dist - dist_center_box, 0.0).astype(np.float32)
    raw_delta = np.clip(raw_delta, -delta_clip, delta_clip).astype(np.float32)

    approach_pos = np.clip(raw_delta, 0.0, None)
    away_pos = np.clip(-raw_delta, 0.0, None)

    approach_r = (approach_scale * approach_pos * orient_gate).astype(np.float32)
    away_penalty = (-away_scale * away_pos).astype(np.float32)
    return approach_r, away_penalty, raw_delta


# ----------------------------- floor / under-box -----------------------------
def floor_underbox_penalties(cfg_r, ee_z, center_z, dist_center_box, box_pos_z, pregrasp_dist):
    """
    floor / under-box penalty（防钻地）
    返回：floor_penalty, under_box_penalty, deep_floor(bool), floor_violation, under_box_mask
    """
    ee_z = ee_z.astype(np.float32)
    center_z = center_z.astype(np.float32)
    dist_center_box = dist_center_box.astype(np.float32)
    box_pos_z = box_pos_z.astype(np.float32)

    floor_z = _getf(cfg_r, "floor_z_rwdfunc", 0.0, old_key="stage1_floor_z")
    min_ee_z = _getf(cfg_r, "min_ee_z_rwdfunc", floor_z + 0.002, old_key="stage1_min_ee_z")
    min_center_z = _getf(cfg_r, "min_center_z_rwdfunc", floor_z + 0.001, old_key="stage1_min_center_z")

    # ✅ NEW（默认更温和）：避免“持续大惩罚 -> 自杀 reset”
    floor_pen_scale = _getf(cfg_r, "floor_penalty_scale_rwdfunc", 0.2, old_key="stage1_floor_penalty_scale")

    ee_v = np.clip(min_ee_z - ee_z, 0.0, None).astype(np.float32)
    c_v = np.clip(min_center_z - center_z, 0.0, None).astype(np.float32)
    floor_violation = (ee_v + c_v).astype(np.float32)

    floor_penalty = (-floor_pen_scale * (ee_v + 0.5 * c_v)).astype(np.float32)

    # ✅ NEW：每步 penalty 绝对值封顶（0=关闭）
    floor_step_cap = _getf(cfg_r, "floor_penalty_step_cap_rwdfunc", 0.0)
    if floor_step_cap > 0.0:
        floor_penalty = np.maximum(floor_penalty, -float(floor_step_cap)).astype(np.float32)

    under_margin = _getf(cfg_r, "under_box_margin_rwdfunc", 0.015, old_key="stage1_under_box_margin")
    under_dist = _getf(cfg_r, "under_box_dist_rwdfunc", pregrasp_dist, old_key="stage1_under_box_dist")

    # ✅ NEW（默认更温和）
    under_scale = _getf(cfg_r, "under_box_scale_rwdfunc", 0.08, old_key="stage1_under_box_penalty_scale")

    under_box = (dist_center_box <= under_dist) & (center_z < (box_pos_z - under_margin))
    under_box_penalty = (-under_scale * (box_pos_z - under_margin - center_z) * under_box.astype(np.float32)).astype(np.float32)

    # ✅ NEW：每步 under-box penalty 绝对值封顶（0=关闭）
    under_step_cap = _getf(cfg_r, "under_box_step_cap_rwdfunc", 0.0)
    if under_step_cap > 0.0:
        under_box_penalty = np.maximum(under_box_penalty, -float(under_step_cap)).astype(np.float32)

    terminate_on_floor = _getb(cfg_r, "terminate_on_floor_rwdfunc", True, old_key="stage1_terminate_on_floor_violation")
    term_margin = _getf(cfg_r, "floor_terminate_margin_rwdfunc", 0.01, old_key="stage1_floor_terminate_margin")
    deep_floor = ((ee_z < (min_ee_z - term_margin)) | (center_z < (min_center_z - term_margin))).astype(bool)
    deep_floor = deep_floor if terminate_on_floor else np.zeros_like(deep_floor, dtype=bool)

    return floor_penalty, under_box_penalty, deep_floor, floor_violation, under_box.astype(np.float32)


# ----------------------------- pinch / stable grasp -----------------------------
def pinch_quality_geommean(
    cfg_r,
    finger_gap_val,
    dist_center_box,
    left_force,
    right_force,
    tip_dist_max,
    orient_gate,
    gap_min,
    gap_max,
    grasp_dist_threshold,
    stay_gap_margin,
    force_threshold,
    eps=1e-6,
):
    """pinch gating（几何平均版）"""
    finger_gap_val = finger_gap_val.astype(np.float32)
    left_force = left_force.astype(np.float32)
    right_force = right_force.astype(np.float32)
    dist_center_box = dist_center_box.astype(np.float32)
    orient_gate = orient_gate.astype(np.float32)

    pinch_force_th = _getf(cfg_r, "pinch_force_th_rwdfunc", force_threshold * 0.6, old_key="stage1_pinch_force_threshold")
    pinch_balance_min = _getf(cfg_r, "pinch_balance_min_rwdfunc", 0.60, old_key="stage1_pinch_balance_min")
    pinch_on = _getf(cfg_r, "pinch_on_rwdfunc", 0.55, old_key="stage1_pinch_on")

    min_force = np.minimum(left_force, right_force).astype(np.float32)
    force_gate = tolerance(
        min_force,
        bounds=(pinch_force_th, pinch_force_th * 4.0),
        margin=max(pinch_force_th * 4.0, 1e-6),
    ).astype(np.float32)

    gap_gate = tolerance(
        finger_gap_val,
        bounds=(gap_min, gap_max),
        margin=max(stay_gap_margin * 4.0, 1e-6),
    ).astype(np.float32)

    dist_gate = tolerance(
        dist_center_box,
        bounds=(0.0, grasp_dist_threshold),
        margin=max(grasp_dist_threshold * 2.0, 1e-6),
    ).astype(np.float32)

    balance = (1.0 - (np.abs(left_force - right_force) / (left_force + right_force + eps))).astype(np.float32)
    balance_gate = tolerance(
        balance,
        bounds=(pinch_balance_min, 1.0),
        margin=max(1.0 - pinch_balance_min, 1e-6),
    ).astype(np.float32)

    if tip_dist_max is None:
        tip_gate = _ones_like(dist_center_box)
        n_terms = 5
        prod = (force_gate * gap_gate * dist_gate * balance_gate * orient_gate).astype(np.float32)
    else:
        tip_dist_max = tip_dist_max.astype(np.float32)
        tip_dist_th = _getf(cfg_r, "tip_dist_th_rwdfunc", 0.055, old_key="stage1_tip_dist_threshold")
        tip_gate = tolerance(
            tip_dist_max,
            bounds=(0.0, tip_dist_th),
            margin=max(tip_dist_th * 2.0, 1e-6),
        ).astype(np.float32)
        n_terms = 6
        prod = (force_gate * gap_gate * dist_gate * balance_gate * tip_gate * orient_gate).astype(np.float32)

    pinch_quality = np.power(np.clip(prod, 0.0, 1.0) + np.float32(eps), 1.0 / float(n_terms)).astype(np.float32)
    pinch_active = (pinch_quality >= pinch_on)
    return pinch_quality, pinch_active.astype(bool), balance, tip_gate


def pinch_quality_bonus(cfg_r, pinch_quality):
    """pinch_quality 主奖励：返回 pinch_bonus"""
    pinch_quality = pinch_quality.astype(np.float32)
    scale = _getf(cfg_r, "pinch_bonus_scale_rwdfunc", 0.08)
    return (scale * pinch_quality).astype(np.float32)


def stable_grasp_with_pinch(
    cfg_r,
    has_cube_prev,
    finger_gap_val,
    left_force,
    right_force,
    dist_center_box,
    center_z,
    tip_dist_max,
    orient_approach,
    pinch_quality,
    gap_min,
    gap_max,
    force_threshold,
    height_threshold,
    grasp_dist_threshold,
    stay_dist_mult,
    stay_force_mult,
    stay_gap_margin,
    stay_h_margin,
    *,
    gripper_axis_world=None,
    env=None,
    data=None,
    model=None,
    ee_site_name=None,
    fingertip_center_pos_world=None,
):
    """返回：stable_grasp_candidate, stable_grasp_now"""
    finger_gap_val = finger_gap_val.astype(np.float32)
    left_force = left_force.astype(np.float32)
    right_force = right_force.astype(np.float32)
    dist_center_box = dist_center_box.astype(np.float32)
    center_z = center_z.astype(np.float32)
    pinch_quality = pinch_quality.astype(np.float32)

    orient_approach, _ = _resolve_orient_approach(
        orient_approach,
        gripper_axis_world=gripper_axis_world,
        dist_center_box=dist_center_box,
        env=env,
        data=data,
        model=model,
        ee_site_name=ee_site_name,
        fingertip_center_pos_world=fingertip_center_pos_world,
    )

    enter_gap_ok = (finger_gap_val >= gap_min) & (finger_gap_val <= gap_max)
    enter_forces_ok = (left_force >= force_threshold) & (right_force >= force_threshold)
    enter_dist_ok = dist_center_box <= grasp_dist_threshold
    enter_height_ok = center_z >= height_threshold

    stable_orient_min = _getf(cfg_r, "stable_orient_min_rwdfunc", 0.20, old_key="stage1_stable_orient_min")
    if orient_approach is None:
        enter_orient_ok = np.ones_like(dist_center_box, dtype=bool)
    else:
        enter_orient_ok = orient_approach.astype(np.float32) >= stable_orient_min

    tip_dist_th = _getf(cfg_r, "tip_dist_th_rwdfunc", 0.055, old_key="stage1_tip_dist_threshold")
    if tip_dist_max is None:
        enter_tip_ok = np.ones_like(dist_center_box, dtype=bool)
    else:
        enter_tip_ok = tip_dist_max.astype(np.float32) <= (tip_dist_th * 1.2)

    stable_enter = (enter_gap_ok & enter_forces_ok & enter_dist_ok & enter_height_ok & enter_orient_ok & enter_tip_ok)

    stay_gap_ok = (finger_gap_val >= (gap_min - stay_gap_margin)) & (finger_gap_val <= (gap_max + stay_gap_margin))
    stay_forces_ok = (left_force >= (force_threshold * stay_force_mult)) & (right_force >= (force_threshold * stay_force_mult))
    stay_dist_ok = dist_center_box <= (grasp_dist_threshold * stay_dist_mult)
    stay_height_ok = center_z >= (height_threshold - stay_h_margin)

    if orient_approach is None:
        stay_orient_ok = np.ones_like(dist_center_box, dtype=bool)
    else:
        stay_orient_ok = orient_approach.astype(np.float32) >= (stable_orient_min - 0.05)

    if tip_dist_max is None:
        stay_tip_ok = np.ones_like(dist_center_box, dtype=bool)
    else:
        stay_tip_ok = tip_dist_max.astype(np.float32) <= (tip_dist_th * 1.5)

    stable_stay = (stay_gap_ok & stay_forces_ok & stay_dist_ok & stay_height_ok & stay_orient_ok & stay_tip_ok)
    stable_candidate = np.where(has_cube_prev, stable_stay, stable_enter).astype(bool)

    pinch_stable_min = _getf(cfg_r, "pinch_stable_min_rwdfunc", 0.20, old_key="stage1_pinch_stable_min")
    stable_now = (stable_candidate & (pinch_quality >= pinch_stable_min)).astype(bool)
    return stable_candidate, stable_now


# ----------------------------- single-contact hint (capped) -----------------------------
def single_contact_hint_reward(
    cfg_r,
    left_force,
    right_force,
    left_tip_z,
    right_tip_z,
    dist_center_box,
    has_cube_now,
    orient_gate=None,
    counter_prev=None,
    *,
    ever_had_cube=None,
    pinch_active=None,
    finger_gap_val=None,
):
    """单爪接触过渡（带上限）：返回 (reward, counter)

    新版额外门控（若提供）：
      - ~ever_had_cube
      - ~pinch_active
      - finger_gap_val >= single_contact_min_gap_rwdfunc（默认 0 关闭）
    """
    left_force = left_force.astype(np.float32)
    right_force = right_force.astype(np.float32)
    left_tip_z = left_tip_z.astype(np.float32)
    right_tip_z = right_tip_z.astype(np.float32)
    dist_center_box = dist_center_box.astype(np.float32)
    has_cube_now = has_cube_now.astype(bool)

    if orient_gate is None:
        orient_gate = _ones_like(dist_center_box)
    else:
        orient_gate = orient_gate.astype(np.float32)

    single_reward = _getf(cfg_r, "single_contact_reward_rwdfunc", 0.01, old_key="stage1_single_contact_reward")
    single_force_th = _getf(cfg_r, "single_force_th_rwdfunc", 1.5, old_key="stage1_single_force_threshold")
    single_dist_th = _getf(cfg_r, "single_dist_th_rwdfunc", 0.05, old_key="stage1_single_dist_threshold")
    single_h_th = _getf(cfg_r, "single_h_th_rwdfunc", 0.02, old_key="stage1_single_height_threshold")

    cap = _geti(cfg_r, "single_contact_cap_rwdfunc", 10)
    cap = max(cap, 0)

    if counter_prev is None:
        counter_prev = np.zeros_like(dist_center_box, dtype=np.int32)
    else:
        counter_prev = counter_prev.astype(np.int32)

    left_contact = left_force >= single_force_th
    right_contact = right_force >= single_force_th
    left_single = left_contact & (~right_contact) & (left_tip_z >= single_h_th)
    right_single = right_contact & (~left_contact) & (right_tip_z >= single_h_th)

    single_mask = (left_single | right_single) & (dist_center_box <= single_dist_th) & (~has_cube_now)

    if ever_had_cube is not None:
        single_mask = single_mask & (~np.asarray(ever_had_cube).astype(bool))
    if pinch_active is not None:
        single_mask = single_mask & (~np.asarray(pinch_active).astype(bool))

    # 可选：避免“闭死按压”也拿到单爪过渡分
    single_contact_min_gap = _getf(cfg_r, "single_contact_min_gap_rwdfunc", 0.0)
    if single_contact_min_gap > 0.0 and finger_gap_val is not None:
        finger_gap_val = np.asarray(finger_gap_val, dtype=np.float32)
        single_mask = single_mask & (finger_gap_val >= single_contact_min_gap)

    counter = counter_prev + single_mask.astype(np.int32)

    if cap == 0:
        return _zeros_like(dist_center_box), counter

    allow = (counter_prev < cap)
    r = (single_reward * single_mask.astype(np.float32) * allow.astype(np.float32) * orient_gate).astype(np.float32)
    return r, counter


# ----------------------------- balance / force target / force soft -----------------------------
def force_balance_bonus(
    cfg_r,
    balance,
    pinch_quality,
    grasp_pre_ready,
    *,
    force_sum=None,
    grasp_ready=None,
    lift_phase=None,
    delta_h=None,
):
    """
    force balance 奖励：返回 force_balance_r
    - 兼容旧：仅依赖 grasp_pre_ready/pinch_quality
    - 新版（提供 force_sum/grasp_ready/lift_phase/delta_h 时）：强门控 + 防按压
    """
    balance = balance.astype(np.float32)
    pinch_quality = pinch_quality.astype(np.float32)
    grasp_pre_ready = np.asarray(grasp_pre_ready).astype(bool)

    if force_sum is None or grasp_ready is None or lift_phase is None or delta_h is None:
        scale = _getf(cfg_r, "force_balance_scale_rwdfunc", 0.15, old_key="stage1_force_balance_scale")
        w = np.where(grasp_pre_ready, 1.0, pinch_quality).astype(np.float32)
        return (scale * balance * w).astype(np.float32)

    force_sum = np.asarray(force_sum).astype(np.float32)
    grasp_ready = np.asarray(grasp_ready).astype(bool)
    lift_phase = np.asarray(lift_phase).astype(bool)
    delta_h = np.asarray(delta_h).astype(np.float32)

    scale = _getf(cfg_r, "force_balance_scale_rwdfunc", 0.06, old_key="stage1_force_balance_scale")
    pq_th = _getf(cfg_r, "force_balance_pinch_th_rwdfunc", 0.20)
    h_eps = _getf(cfg_r, "force_balance_delta_eps_rwdfunc", 1e-4)

    target = _getf(cfg_r, "lift_force_target_rwdfunc", 60.0, old_key="stage1_lift_force_target")
    tol = _getf(cfg_r, "lift_force_tol_rwdfunc", 0.25, old_key="stage1_lift_force_tol")
    margin = _getf(cfg_r, "lift_force_margin_rwdfunc", 80.0, old_key="stage1_lift_force_margin")

    f_low = float(target * (1.0 - tol))
    f_high = float(target * (1.0 + tol))

    pq_gate = (pinch_quality >= pq_th).astype(np.float32)
    force_gate = tolerance(force_sum, bounds=(f_low, f_high), margin=max(margin, 1e-6)).astype(np.float32)

    stage_gate = ((~grasp_ready) | (grasp_ready & lift_phase)).astype(np.float32)
    trend_gate = ((lift_phase) | (delta_h > h_eps)).astype(np.float32)

    return (scale * balance * pq_gate * force_gate * stage_gate * trend_gate).astype(np.float32)


def force_target_bonus(
    cfg_r,
    force_sum,
    box_height,
    lift_gate=None,
    progress=None,
    *,
    grasp_ready=None,
    has_cube_now=None,
    delta_h=None,
    lift_phase=None,
):
    """
    夹持总力目标带奖励：返回 force_target_r
    - 兼容旧：active=(box_height>=start_h)&(lift_gate>0)
    - 新版（提供 grasp_ready/has_cube_now/delta_h/lift_phase 时）：抓稳+抬升趋势硬门控
    """
    force_sum = force_sum.astype(np.float32)
    box_height = box_height.astype(np.float32)

    if lift_gate is None:
        lift_gate = _ones_like(force_sum)
    else:
        lift_gate = lift_gate.astype(np.float32)

    if progress is None:
        progress = _ones_like(force_sum)
    else:
        progress = progress.astype(np.float32)

    start_h = _getf(cfg_r, "lift_force_start_h_rwdfunc", 0.006, old_key="stage1_lift_force_start_height")
    target = _getf(cfg_r, "lift_force_target_rwdfunc", 60.0, old_key="stage1_lift_force_target")
    tol = _getf(cfg_r, "lift_force_tol_rwdfunc", 0.25, old_key="stage1_lift_force_tol")
    margin = _getf(cfg_r, "lift_force_margin_rwdfunc", 80.0, old_key="stage1_lift_force_margin")
    scale = _getf(cfg_r, "lift_force_reward_scale_rwdfunc", 0.25, old_key="stage1_lift_force_reward_scale")

    f_low = float(target * (1.0 - tol))
    f_high = float(target * (1.0 + tol))

    if grasp_ready is None or has_cube_now is None or delta_h is None or lift_phase is None:
        active = (box_height >= start_h) & (lift_gate > 0.0)
    else:
        grasp_ready = np.asarray(grasp_ready).astype(bool)
        has_cube_now = np.asarray(has_cube_now).astype(bool)
        delta_h = np.asarray(delta_h).astype(np.float32)
        lift_phase = np.asarray(lift_phase).astype(bool)
        h_eps = _getf(cfg_r, "lift_force_delta_eps_rwdfunc", 1e-4)
        active = grasp_ready & has_cube_now & (box_height >= start_h) & ((delta_h > h_eps) | lift_phase)

    r = (scale * tolerance(force_sum, bounds=(f_low, f_high), margin=max(margin, 1e-6))
         * active.astype(np.float32) * progress).astype(np.float32)
    return r


def force_soft_penalty(cfg_r, max_force):
    """软超力惩罚：返回 force_penalty"""
    max_force = max_force.astype(np.float32)
    th = _getf(cfg_r, "force_soft_th_rwdfunc", 80.0, old_key="stage1_force_soft_threshold")
    scale = _getf(cfg_r, "force_soft_scale_rwdfunc", 2e-5, old_key="stage1_force_penalty_scale")
    excess = np.clip(max_force - th, 0.0, None).astype(np.float32)
    return (-scale * excess).astype(np.float32)


# ----------------------------- displacement / closed-grasp -----------------------------
def push_displacement_penalty(cfg_r, delta_cube_disp, lifted, stable_grasp_candidate_or_now):
    """推走/位移惩罚：返回 push_penalty"""
    delta_cube_disp = delta_cube_disp.astype(np.float32)
    lifted = lifted.astype(bool)
    stable = stable_grasp_candidate_or_now.astype(bool)

    scale = _getf(cfg_r, "push_penalty_scale_rwdfunc", 0.02, old_key="stage1_push_penalty_scale")
    scale_lifted = _getf(cfg_r, "push_penalty_scale_lifted_rwdfunc", scale * 0.25, old_key="stage1_push_penalty_scale_lifted")
    s = np.where(lifted, scale_lifted, scale).astype(np.float32)

    return (-s * delta_cube_disp * (~stable).astype(np.float32)).astype(np.float32)


def closed_no_grasp_penalty(cfg_r, finger_gap_val, has_cube_now, dist_center_box):
    """闭合但没抓住的惩罚：返回 penalty"""
    finger_gap_val = finger_gap_val.astype(np.float32)
    has_cube_now = has_cube_now.astype(bool)
    dist_center_box = dist_center_box.astype(np.float32)

    mag = _getf(cfg_r, "closed_no_grasp_mag_rwdfunc", 1.0, old_key="stage1_closed_no_grasp_penalty")
    gap_th = _getf(cfg_r, "closed_gap_th_rwdfunc", 0.020, old_key="stage1_closed_gap_thresh")
    dist_th = _getf(cfg_r, "closed_no_grasp_dist_th_rwdfunc", 0.05, old_key="stage1_single_dist_threshold")

    mask = (finger_gap_val <= gap_th) & (~has_cube_now) & (dist_center_box <= dist_th)
    return (-mag * mask.astype(np.float32)).astype(np.float32)


# ----------------------------- counters: hold / no_contact / no_grasp -----------------------------
def update_hold_counter(cfg_r, stable_grasp_now, hold_counter_prev):
    """hold_counter 带 decay：返回 hold_counter"""
    stable_grasp_now = stable_grasp_now.astype(bool)
    hold_counter_prev = hold_counter_prev.astype(np.int32)

    decay = _geti(cfg_r, "hold_decay_rwdfunc", 1, old_key="stage1_hold_decay")
    decay = max(decay, 1)

    hold_counter = np.where(
        stable_grasp_now,
        hold_counter_prev + 1,
        np.maximum(hold_counter_prev - decay, 0),
    ).astype(np.int32)
    return hold_counter


def grasp_hold_rewards(cfg_r, stable_grasp_now, hold_counter):
    """
    抓稳 build+maintain：返回 (grasp_pre_ready, grasp_ready, hold_build_r, hold_maintain_r)
    新版：build 仅当步稳定才给（避免“衰减余额”刷分）
    """
    stable_grasp_now = stable_grasp_now.astype(bool)
    hold_counter = hold_counter.astype(np.int32)

    hold_steps = _geti(cfg_r, "hold_steps_rwdfunc", 3, old_key="stage1_hold_steps")
    pre_steps = _geti(cfg_r, "pre_ready_steps_rwdfunc", 1, old_key="stage1_pre_ready_steps")
    hold_steps = max(hold_steps, 1)
    pre_steps = max(pre_steps, 1)

    step_r = _getf(cfg_r, "grasp_step_reward_rwdfunc", 0.06, old_key="stage1_grasp_step_reward")
    maintain_r = _getf(cfg_r, "grasp_maintain_reward_rwdfunc", 0.03, old_key="stage1_grasp_maintain_reward")

    grasp_ready = (stable_grasp_now & (hold_counter >= hold_steps)).astype(bool)
    grasp_pre_ready = (stable_grasp_now & (hold_counter >= pre_steps)).astype(bool)

    hold_build = np.minimum(hold_counter.astype(np.float32), float(hold_steps))
    build_r = (step_r * hold_build * stable_grasp_now.astype(np.float32) * (~grasp_ready).astype(np.float32)).astype(np.float32)

    maintain = (maintain_r * grasp_ready.astype(np.float32)).astype(np.float32)
    return grasp_pre_ready, grasp_ready, build_r, maintain


def update_no_contact_no_grasp(
    cfg_r,
    any_contact_with_cube,
    has_cube_now,
    ever_had_cube,
    steps_since_contact_prev,
    steps_since_grasp_prev,
    *,
    triggered_contact_prev=None,
    triggered_grasp_prev=None,
    return_termination: bool = False,
):
    """
    no_contact / no_grasp（新版默认：不做持续性扣分，而是“超阈终止 + 可选一次性轻惩罚”）
    - return_termination=False：保持旧签名，返回 4 个量（兼容其它 stage）
    - return_termination=True ：返回 8 个量（含终止 mask + triggered flag）

    cfg 开关（都可选，不配也能跑）：
      - no_contact_mode_rwdfunc: 0=off, 1=legacy-per-step, 2=terminate+once (默认)
      - no_grasp_mode_rwdfunc:   0=off, 1=legacy-per-step, 2=terminate+once (默认)
      - no_contact_terminate_rwdfunc / no_grasp_terminate_rwdfunc: 是否触发终止
      - no_contact_once_mag_rwdfunc / no_grasp_once_mag_rwdfunc: 终止时一次性惩罚幅度（正数，内部取负）
    """
    any_contact_with_cube = any_contact_with_cube.astype(bool)
    has_cube_now = has_cube_now.astype(bool)
    ever_had_cube = ever_had_cube.astype(bool)

    steps_since_contact_prev = steps_since_contact_prev.astype(np.int32)
    steps_since_grasp_prev = steps_since_grasp_prev.astype(np.int32)

    freeze_after_grasp = _getb(cfg_r, "freeze_nopen_after_grasp_rwdfunc", True)

    steps_since_contact = np.where(any_contact_with_cube, 0, steps_since_contact_prev + 1).astype(np.int32)
    steps_since_grasp = np.where(has_cube_now, 0, steps_since_grasp_prev + 1).astype(np.int32)

    if freeze_after_grasp:
        steps_since_contact = np.where(ever_had_cube, 0, steps_since_contact).astype(np.int32)
        steps_since_grasp = np.where(ever_had_cube, 0, steps_since_grasp).astype(np.int32)

    no_contact_steps = _geti(cfg_r, "no_contact_steps_rwdfunc", 80, old_key="stage1_no_contact_steps")
    no_grasp_steps = _geti(cfg_r, "no_grasp_steps_rwdfunc", 150, old_key="stage1_no_grasp_steps")

    contact_exceeded = (steps_since_contact > no_contact_steps)
    grasp_exceeded = (steps_since_grasp > no_grasp_steps)

    # ---------- defaults ----------
    mode_contact = _geti(cfg_r, "no_contact_mode_rwdfunc", 2)
    mode_grasp = _geti(cfg_r, "no_grasp_mode_rwdfunc", 2)

    # legacy mags（仍保留 old_key 兼容）
    legacy_contact_mag = _getf(cfg_r, "no_contact_mag_rwdfunc", 0.0, old_key="stage1_long_no_contact_penalty")
    legacy_grasp_mag = _getf(cfg_r, "no_grasp_mag_rwdfunc", 0.0, old_key="stage1_long_no_grasp_penalty")

    term_contact_enable = _getb(cfg_r, "no_contact_terminate_rwdfunc", True)
    term_grasp_enable = _getb(cfg_r, "no_grasp_terminate_rwdfunc", True)

    once_contact_mag = _getf(cfg_r, "no_contact_once_mag_rwdfunc", 0.0)
    once_grasp_mag = _getf(cfg_r, "no_grasp_once_mag_rwdfunc", 0.0)

    if triggered_contact_prev is None:
        triggered_contact_prev = np.zeros_like(contact_exceeded, dtype=bool)
    else:
        triggered_contact_prev = np.asarray(triggered_contact_prev, dtype=bool)

    if triggered_grasp_prev is None:
        triggered_grasp_prev = np.zeros_like(grasp_exceeded, dtype=bool)
    else:
        triggered_grasp_prev = np.asarray(triggered_grasp_prev, dtype=bool)

    # ---------- penalties & termination ----------
    no_contact_penalty = np.zeros_like(steps_since_contact, dtype=np.float32)
    no_grasp_penalty = np.zeros_like(steps_since_grasp, dtype=np.float32)

    term_contact = np.zeros_like(contact_exceeded, dtype=bool)
    term_grasp = np.zeros_like(grasp_exceeded, dtype=bool)

    trig_contact = triggered_contact_prev.copy()
    trig_grasp = triggered_grasp_prev.copy()

    # mode 1: legacy per-step after threshold
    if mode_contact == 1:
        no_contact_penalty = (-legacy_contact_mag * contact_exceeded.astype(np.float32)).astype(np.float32)
    elif mode_contact == 2:
        term_contact = (contact_exceeded & term_contact_enable).astype(bool)
        new_trig = contact_exceeded & (~triggered_contact_prev)
        trig_contact = (triggered_contact_prev | new_trig).astype(bool)
        if once_contact_mag > 0.0:
            no_contact_penalty = (-once_contact_mag * new_trig.astype(np.float32)).astype(np.float32)

    if mode_grasp == 1:
        no_grasp_penalty = (-legacy_grasp_mag * grasp_exceeded.astype(np.float32)).astype(np.float32)
    elif mode_grasp == 2:
        term_grasp = (grasp_exceeded & term_grasp_enable).astype(bool)
        new_trig = grasp_exceeded & (~triggered_grasp_prev)
        trig_grasp = (triggered_grasp_prev | new_trig).astype(bool)
        if once_grasp_mag > 0.0:
            no_grasp_penalty = (-once_grasp_mag * new_trig.astype(np.float32)).astype(np.float32)

    if not return_termination:
        return no_contact_penalty, no_grasp_penalty, steps_since_contact, steps_since_grasp

    return (
        no_contact_penalty,
        no_grasp_penalty,
        steps_since_contact,
        steps_since_grasp,
        term_contact,
        term_grasp,
        trig_contact,
        trig_grasp,
    )


# ----------------------------- success-once helpers -----------------------------
def success_once_reward(cfg_r, success_mask, achieved_prev, reward_key, default_reward):
    """通用一次性成功奖励：返回 (success_r, achieved_now, success_new)"""
    success_mask = success_mask.astype(bool)
    achieved_prev = achieved_prev.astype(bool)

    r_value = _getf(cfg_r, reward_key, default_reward)
    success_new = success_mask & (~achieved_prev)
    success_r = (r_value * success_new.astype(np.float32)).astype(np.float32)
    achieved_now = achieved_prev | success_mask
    return success_r, achieved_now.astype(bool), success_new.astype(bool)


# ----------------------------- lift shaping -----------------------------
def lift_height_shaping(cfg_r, box_height, lift_start_h, lift_goal_h, lift_gate=None, progress=None):
    """lift height shaping：返回 lift_height_r, h_norm"""
    box_height = box_height.astype(np.float32)
    if lift_gate is None:
        lift_gate = _ones_like(box_height)
    else:
        lift_gate = lift_gate.astype(np.float32)

    if progress is None:
        progress = _ones_like(box_height)
    else:
        progress = progress.astype(np.float32)

    scale = _getf(cfg_r, "lift_height_scale_rwdfunc", 2.0, old_key="stage1_lift_height_reward_scale")

    denom = np.float32(max(float(lift_goal_h - lift_start_h), 1e-6))
    h_norm = np.clip((box_height - float(lift_start_h)) / denom, 0.0, 1.0).astype(np.float32)
    r = (scale * h_norm * lift_gate * progress).astype(np.float32)
    return r, h_norm


def lift_delta_shaping(cfg_r, box_height, prev_box_height, lift_gate=None, progress=None):
    """lift delta shaping：返回 lift_delta_r, lift_delta_pos, delta_h"""
    box_height = box_height.astype(np.float32)
    prev_box_height = prev_box_height.astype(np.float32)

    if lift_gate is None:
        lift_gate = _ones_like(box_height)
    else:
        lift_gate = lift_gate.astype(np.float32)

    if progress is None:
        progress = _ones_like(box_height)
    else:
        progress = progress.astype(np.float32)

    scale = _getf(cfg_r, "lift_delta_scale_rwdfunc", 30.0, old_key="stage1_lift_delta_reward_scale")
    delta_h = (box_height - prev_box_height).astype(np.float32)
    delta_pos = np.clip(delta_h, 0.0, None).astype(np.float32)
    r = (scale * delta_pos * lift_gate * progress).astype(np.float32)
    return r, delta_pos, delta_h


def lift_band_stepcount_reward(
    cfg_r,
    box_height,
    prev_box_height,
    has_cube_now,
    grasp_ready,
    count_prev,
    paid_prev,
    cooldown_prev=None,
):
    """
    区间上升刷分封顶 + 超阈一次性补齐（不要求先刷满 N）
    返回：band_r, count, paid, in_band, over_high, cooldown
    """
    box_height = box_height.astype(np.float32)
    prev_box_height = prev_box_height.astype(np.float32)
    has_cube_now = has_cube_now.astype(bool)
    grasp_ready = grasp_ready.astype(bool)

    if count_prev is None:
        count_prev = np.zeros_like(box_height, dtype=np.int32)
    else:
        count_prev = count_prev.astype(np.int32)

    if paid_prev is None:
        paid_prev = np.zeros_like(box_height, dtype=bool)
    else:
        paid_prev = paid_prev.astype(bool)

    if cooldown_prev is None:
        cooldown_prev = np.zeros_like(box_height, dtype=np.int32)
    else:
        cooldown_prev = cooldown_prev.astype(np.int32)

    low = _getf(cfg_r, "lift_band_low_rwdfunc", 0.03)
    high = _getf(cfg_r, "lift_band_high_rwdfunc", 0.10)
    n_max = _geti(cfg_r, "lift_band_nmax_rwdfunc", 10)
    n_max = max(n_max, 1)

    step_reward = _getf(cfg_r, "lift_band_step_reward_rwdfunc", 1.0)
    h_eps = _getf(cfg_r, "lift_band_delta_eps_rwdfunc", 1e-4)

    require_ready = _getb(cfg_r, "lift_band_require_grasp_ready_rwdfunc", True)
    cooldown_steps = _geti(cfg_r, "lift_band_cooldown_steps_rwdfunc", 0)
    cooldown_steps = max(cooldown_steps, 0)

    delta_h = (box_height - prev_box_height).astype(np.float32)
    up = (delta_h > h_eps)

    in_band = (box_height >= low) & (box_height <= high)
    over_high = (box_height > high)

    gate = (grasp_ready & has_cube_now) if require_ready else has_cube_now

    reset = ~gate
    count = np.where(reset, 0, count_prev).astype(np.int32)
    paid = np.where(reset, False, paid_prev).astype(bool)
    cooldown = np.where(reset, 0, np.maximum(cooldown_prev - 1, 0)).astype(np.int32)

    can_inc = (cooldown == 0)
    inc_mask = gate & in_band & up & can_inc & (count < n_max)
    count2 = (count + inc_mask.astype(np.int32)).astype(np.int32)

    band_r = (step_reward * inc_mask.astype(np.float32)).astype(np.float32)

    if cooldown_steps > 0:
        cooldown = np.where(inc_mask, cooldown_steps, cooldown).astype(np.int32)

    # 超过 high：一次性补齐到 N（即支付剩余次数的等额奖励）
    pay_mask = gate & over_high & (~paid)
    remaining = np.clip(n_max - count2, 0, n_max).astype(np.int32)
    band_r = (band_r + (remaining.astype(np.float32) * step_reward) * pay_mask.astype(np.float32)).astype(np.float32)
    count2 = np.where(pay_mask, n_max, count2).astype(np.int32)
    paid = (paid | pay_mask).astype(bool)

    return band_r, count2, paid, in_band.astype(np.float32), over_high.astype(np.float32), cooldown


def lift_height_milestone_reward(cfg_r, box_height, achieved_prev):
    """lift height milestone：返回 (milestone_r, achieved_now, triggered)"""
    box_height = box_height.astype(np.float32)
    achieved_prev = achieved_prev.astype(bool)

    h1 = _getf(cfg_r, "lift_milestone_h1_rwdfunc", 0.05)
    h2 = _getf(cfg_r, "lift_milestone_h2_rwdfunc", 0.10)
    r1 = _getf(cfg_r, "lift_milestone_r1_rwdfunc", 5.0)
    r2 = _getf(cfg_r, "lift_milestone_r2_rwdfunc", 10.0)

    trig1 = (box_height >= h1) & (~achieved_prev)
    achieved1 = achieved_prev | trig1
    trig2 = (box_height >= h2) & (~achieved1)
    achieved_now = achieved1 | trig2

    reward = (r1 * trig1.astype(np.float32) + r2 * trig2.astype(np.float32)).astype(np.float32)
    triggered = (trig1 | trig2).astype(bool)
    return reward, achieved_now.astype(bool), triggered


def lift_success_mask(cfg_r, box_height, lift_hold_counter):
    """lift success 判定：返回 success_mask(bool)"""
    box_height = box_height.astype(np.float32)
    lift_hold_counter = lift_hold_counter.astype(np.int32)

    lift_goal_h = _getf(cfg_r, "lift_goal_h_rwdfunc", 0.03, old_key="stage1_lift_goal_height")
    lift_hold_steps = _geti(cfg_r, "lift_hold_steps_rwdfunc", 5, old_key="stage1_lift_hold_steps")
    lift_hold_steps = max(lift_hold_steps, 1)

    return ((lift_hold_counter >= lift_hold_steps) & (box_height >= lift_goal_h)).astype(bool)


# ----------------------------- progress mixing / pry penalties -----------------------------
def progress_mixing_start(cfg_r, steps, grasp_step_prev, grasp_pre_ready, pinch_active):
    """grasp_pre_ready 或 pinch_active 触发 progress 起点：返回 grasp_step, progress"""
    steps = steps.astype(np.int32)
    grasp_step_prev = grasp_step_prev.astype(np.int32)
    grasp_pre_ready = grasp_pre_ready.astype(bool)
    pinch_active = pinch_active.astype(bool)

    mix_steps = _geti(cfg_r, "mix_steps_rwdfunc", 250, old_key="stage1_mix_steps")
    mix_steps = max(mix_steps, 1)

    new_start = (grasp_pre_ready | pinch_active) & (grasp_step_prev < 0)
    grasp_step = np.where(new_start, steps, grasp_step_prev).astype(np.int32)

    since = np.where(grasp_step >= 0, (steps - grasp_step).astype(np.float32), 0.0).astype(np.float32)
    progress = np.clip(since / float(mix_steps), 0.0, 1.0).astype(np.float32)
    return grasp_step, progress


def pry_penalty_delta(cfg_r, has_cube_now, any_contact_with_cube, lift_delta_pos, pinch_quality):
    """pry_penalty_delta"""
    has_cube_now = has_cube_now.astype(bool)
    any_contact_with_cube = any_contact_with_cube.astype(bool)
    lift_delta_pos = lift_delta_pos.astype(np.float32)
    pinch_quality = pinch_quality.astype(np.float32)

    scale = _getf(cfg_r, "pry_delta_scale_rwdfunc", 80.0, old_key="stage1_pry_penalty_scale")
    mask = (~has_cube_now) & any_contact_with_cube
    return (-scale * lift_delta_pos * (1.0 - pinch_quality) * mask.astype(np.float32)).astype(np.float32)


def pry_penalty_abs(cfg_r, has_cube_now, any_contact_with_cube, box_height, lift_start_h, pinch_quality):
    """pry_penalty_abs"""
    has_cube_now = has_cube_now.astype(bool)
    any_contact_with_cube = any_contact_with_cube.astype(bool)
    box_height = box_height.astype(np.float32)
    pinch_quality = pinch_quality.astype(np.float32)

    scale = _getf(cfg_r, "pry_abs_scale_rwdfunc", 25.0, old_key="stage1_pry_penalty_abs_scale")
    mask = (~has_cube_now) & any_contact_with_cube
    pry_h = np.clip(box_height - float(lift_start_h), 0.0, None).astype(np.float32)
    return (-scale * pry_h * (1.0 - pinch_quality) * mask.astype(np.float32)).astype(np.float32)


def pry_penalty_terms(cfg_r, has_cube_now, any_contact_with_cube, lift_delta_pos, box_height, lift_start_h, pinch_quality, delta_cube_disp):
    """pry penalty 总和：返回 pry_penalty"""
    delta_cube_disp = delta_cube_disp.astype(np.float32)
    xy_scale = _getf(cfg_r, "pry_xy_scale_rwdfunc", 0.05, old_key="stage1_pry_penalty_xy_scale")

    p_delta = pry_penalty_delta(cfg_r, has_cube_now, any_contact_with_cube, lift_delta_pos, pinch_quality)
    p_abs = pry_penalty_abs(cfg_r, has_cube_now, any_contact_with_cube, box_height, lift_start_h, pinch_quality)

    mask = (~has_cube_now.astype(bool)) & any_contact_with_cube.astype(bool)
    p_xy = (-xy_scale * delta_cube_disp * (1.0 - pinch_quality.astype(np.float32)) * mask.astype(np.float32)).astype(np.float32)
    return (p_delta + p_abs + p_xy).astype(np.float32)


# ----------------------------- new penalties -----------------------------
def pressing_penalty_by_force(
    cfg_r,
    dist_center_box,
    any_contact_with_cube,
    pinch_active,
    pinch_quality,
    force_sum,
    box_height,
    delta_h,
    counter_prev=None,
):
    """
    新增：按压惩罚（近场接触 + pinch低 + 大力 + 没抬）
    返回 (penalty, counter)
    """
    dist_center_box = dist_center_box.astype(np.float32)
    any_contact_with_cube = any_contact_with_cube.astype(bool)
    pinch_active = pinch_active.astype(bool)
    pinch_quality = pinch_quality.astype(np.float32)
    force_sum = force_sum.astype(np.float32)
    box_height = box_height.astype(np.float32)
    delta_h = delta_h.astype(np.float32)

    near = _getf(cfg_r, "pressing_near_dist_rwdfunc", _getf(cfg_r, "upright_bonus_near_dist_rwdfunc", 0.06))
    force_th = _getf(cfg_r, "pressing_force_th_rwdfunc", 20.0)

    # ✅ NEW（默认更温和）
    scale = _getf(cfg_r, "pressing_penalty_scale_rwdfunc", 0.01)

    # 每步惩罚幅度上限（0 表示不封顶）
    cap = _getf(cfg_r, "pressing_penalty_cap_rwdfunc", 0.05)
    lift_start_h = _getf(cfg_r, "pressing_lift_start_h_rwdfunc", 0.006)
    h_eps = _getf(cfg_r, "pressing_delta_eps_rwdfunc", 1e-4)
    sustain = _geti(cfg_r, "pressing_sustain_steps_rwdfunc", 0)
    sustain = max(sustain, 0)
    pinch_on = _getf(cfg_r, "pinch_on_rwdfunc", 0.55, old_key="stage1_pinch_on")
    pinch_low_mult = _getf(cfg_r, "pressing_pinch_low_mult_rwdfunc", 0.8)
    pinch_low_th = float(pinch_on * pinch_low_mult)

    if counter_prev is None:
        counter_prev = np.zeros_like(dist_center_box, dtype=np.int32)
    else:
        counter_prev = counter_prev.astype(np.int32)

    base_mask = (
        (dist_center_box < near)
        & any_contact_with_cube
        & (~pinch_active)
        & (pinch_quality < pinch_low_th)
        & (force_sum > force_th)
        & ((box_height < lift_start_h) | (delta_h <= h_eps))
    )

    counter = np.where(base_mask, counter_prev + 1, 0).astype(np.int32)
    active = base_mask if sustain == 0 else (base_mask & (counter >= sustain))

    denom = np.maximum(force_th, 1e-6)
    excess_ratio = (np.clip(force_sum - force_th, 0.0, None) / denom).astype(np.float32)

    pen_mag = (scale * excess_ratio).astype(np.float32)
    if cap > 0.0:
        pen_mag = np.minimum(pen_mag, cap).astype(np.float32)

    pen = (-pen_mag * active.astype(np.float32)).astype(np.float32)
    return pen, counter


def post_grasp_time_penalty(
    cfg_r,
    grasp_ready,
    has_cube_now,
    lift_success,
    timer_prev=None,
    *,
    box_height=None,
    delta_h=None,
):
    """
    新增：抓稳后“卡住才扣”的时间惩罚（延迟K步开始，默认更轻）
    - timer 只在 running 且（可选）无净抬升进展时累加
    返回 (penalty, timer)
    """
    grasp_ready = grasp_ready.astype(bool)
    has_cube_now = has_cube_now.astype(bool)
    lift_success = lift_success.astype(bool)

    # ✅ NEW（默认更轻）
    c = _getf(cfg_r, "post_grasp_time_penalty_c_rwdfunc", 0.002)
    delay = _geti(cfg_r, "post_grasp_time_penalty_delay_rwdfunc", 5)
    delay = max(delay, 0)

    # 可选：只在“无进展”时计时
    h_eps = _getf(cfg_r, "post_grasp_delta_eps_rwdfunc", 1e-4)
    require_no_progress = _getb(cfg_r, "post_grasp_require_no_progress_rwdfunc", True)

    # 可选：只在进入 band_low 前启用
    band_low = _getf(cfg_r, "lift_band_low_rwdfunc", 0.03)
    preband_only = _getb(cfg_r, "post_grasp_preband_only_rwdfunc", True)

    # 可选：持续 M 步才开始扣（0=不需要）
    sustain = _geti(cfg_r, "post_grasp_sustain_steps_rwdfunc", 0)
    sustain = max(sustain, 0)

    # 每步封顶（0=不封顶）
    step_cap = _getf(cfg_r, "post_grasp_time_penalty_step_cap_rwdfunc", 0.02)

    if timer_prev is None:
        timer_prev = np.zeros_like(grasp_ready, dtype=np.int32)
    else:
        timer_prev = timer_prev.astype(np.int32)

    running = grasp_ready & has_cube_now & (~lift_success)

    if preband_only and (box_height is not None):
        bh = np.asarray(box_height, dtype=np.float32)
        running = running & (bh < band_low)

    if require_no_progress and (delta_h is not None):
        dh = np.asarray(delta_h, dtype=np.float32)
        running = running & (dh <= h_eps)

    timer = np.where(running, timer_prev + 1, 0).astype(np.int32)

    active = running & (timer > delay)
    if sustain > 0:
        active = active & (timer >= (delay + sustain))

    pen_mag = (c * active.astype(np.float32)).astype(np.float32)
    if step_cap > 0.0:
        pen_mag = np.minimum(pen_mag, step_cap).astype(np.float32)

    pen = (-pen_mag).astype(np.float32)
    return pen, timer


def drop_penalty_once(
    cfg_r,
    has_cube_prev,
    has_cube_now,
    box_height,
    lift_phase,
    cooldown_prev=None,
    *,
    grasp_ready_prev=None,
    hold_counter_prev=None,
):
    """
    新增：掉落惩罚（一次性 + 冷却）
    返回 (penalty, cooldown)
    """
    has_cube_prev = has_cube_prev.astype(bool)
    has_cube_now = has_cube_now.astype(bool)
    box_height = box_height.astype(np.float32)
    lift_phase = lift_phase.astype(bool)

    h_drop = _getf(cfg_r, "drop_penalty_height_rwdfunc", _getf(cfg_r, "lift_band_low_rwdfunc", 0.03))

    # ✅ NEW（默认更温和）
    mag = _getf(cfg_r, "drop_penalty_mag_rwdfunc", 5.0)

    cooldown_steps = _geti(cfg_r, "drop_penalty_cooldown_steps_rwdfunc", 10)
    cooldown_steps = max(cooldown_steps, 0)

    if cooldown_prev is None:
        cooldown_prev = np.zeros_like(has_cube_prev, dtype=np.int32)
    else:
        cooldown_prev = cooldown_prev.astype(np.int32)

    cooldown = np.maximum(cooldown_prev - 1, 0).astype(np.int32)

    dropped = has_cube_prev & (~has_cube_now) & lift_phase & (box_height > h_drop) & (cooldown == 0)

    require_grasp_prev = bool(getattr(cfg_r, "drop_penalty_require_grasp_ready_prev_rwdfunc", True))
    if require_grasp_prev:
        if grasp_ready_prev is not None:
            dropped = dropped & np.asarray(grasp_ready_prev, dtype=bool)
        elif hold_counter_prev is not None:
            min_hold = _geti(cfg_r, "drop_penalty_min_hold_counter_rwdfunc", 1)
            dropped = dropped & (np.asarray(hold_counter_prev, dtype=np.int32) >= min_hold)

    pen = (-mag * dropped.astype(np.float32)).astype(np.float32)

    cooldown = np.where(dropped, cooldown_steps, cooldown).astype(np.int32)
    return pen, cooldown


# ----------------------------- origin bonus (optional) -----------------------------
def origin_bonus_reward(cfg_r, cube_disp_xy):
    """原地 bonus：返回 origin_bonus_r"""
    cube_disp_xy = cube_disp_xy.astype(np.float32)
    tol = _getf(cfg_r, "origin_tol_rwdfunc", 0.02, old_key="stage1_origin_tol")
    scale = _getf(cfg_r, "origin_bonus_scale_rwdfunc", 0.02, old_key="stage1_origin_bonus_scale")

    return (scale * tolerance(
        cube_disp_xy,
        bounds=(0.0, tol),
        margin=max(tol * 4.0, 1e-6),
    )).astype(np.float32)


# ----------------------------- smoothness (placeholder) -----------------------------
def smoothness_penalty(cfg_r, action=None):
    """smoothness_penalty 占位：返回 0"""
    if action is None:
        return np.array([0.0], dtype=np.float32)
    return np.zeros((action.shape[0],), dtype=np.float32)

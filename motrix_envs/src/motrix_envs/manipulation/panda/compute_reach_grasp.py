# ===== motrix_envs/src/motrix_envs/manipulation/panda/compute_reach_grasp.py =====
# motrix_envs/src/motrix_envs/manipulation/panda/compute_reach_grasp.py
#
# Stage1 (reach_grasp) — simplified reward:
#   - Only reward "improving best-so-far" progress (distance / pinch-quality / height)
#   - Milestones are one-time
#   - Safety penalties are sparse and event-like
#
# It implements the "best-so-far progress" formulation:
#   b_d(t)=min_{<=t} d,  Δb_d=b_d(t-1)-b_d(t) >= 0
#   b_q(t)=max_{<=t} q,  Δb_q=b_q(t)-b_q(t-1) >= 0
#   b_h(t)=max_{<=t} h,  Δb_h=b_h(t)-b_h(t-1) >= 0
#
# r_t =
#   1[~grasp_ready] * g_ori * (w_d Δb_d + w_q Δb_q)
# + 1[ grasp_ready] * (w_h Δb_h)
# + R_milestone(t)
# - P_safety(t)
#
import numpy as np

from .compute_common import _compute_common_terms

# Reuse existing detectors / helpers (keeps compatibility with reward_config keys).
from .reward_functions import (
    orient_upright_bonus_and_gate,
    floor_underbox_penalties,
    pinch_quality_geommean,
    stable_grasp_with_pinch,
    update_hold_counter,
    grasp_hold_rewards,
    lift_success_mask,
    success_once_reward,
    update_no_contact_no_grasp,
    drop_penalty_once,
)


def compute_reward_reach_grasp(env, data, info):
    """Public entry (kept for compatibility)."""
    return _compute_reward_reach_grasp(env, data, info)


def _getf(cfg_r, key: str, default: float, old_key: str | None = None) -> float:
    if old_key is None:
        return float(getattr(cfg_r, key, default))
    return float(getattr(cfg_r, key, getattr(cfg_r, old_key, default)))


def _geti(cfg_r, key: str, default: int, old_key: str | None = None) -> int:
    if old_key is None:
        return int(getattr(cfg_r, key, default))
    return int(getattr(cfg_r, key, getattr(cfg_r, old_key, default)))


def _getb(cfg_r, key: str, default: bool, old_key: str | None = None) -> bool:
    if old_key is None:
        return bool(getattr(cfg_r, key, default))
    return bool(getattr(cfg_r, key, getattr(cfg_r, old_key, default)))


def _compute_reward_reach_grasp(env, data, info):
    """
    Simplified stage1 reward implementation.

    Notes:
    - This implementation only touches this file (compute_reach_grasp.py).
    - It intentionally avoids "maintain" style rewards that can be exploited by standing still.
    - It keeps the same return signature and common info keys where practical.
    """
    num_envs = data.shape[0]
    cfg_r = env.cfg.reward_config

    c = _compute_common_terms(env, data, info)

    # -------------------- core states --------------------
    box_pos = c["box_pos"]
    ee_pos = c["ee_pos"]
    dof_pos = c["dof_pos"]
    dof_vel = c["dof_vel"]

    dist_center_box = c["dist_fingertip_center_box"].astype(np.float32)
    dist_ee_box = c["dist_ee_box"].astype(np.float32)

    box_height = c["box_height"].astype(np.float32)

    finger_gap_val = c["finger_gap_val"].astype(np.float32)
    left_force = c["left_force"].astype(np.float32)
    right_force = c["right_force"].astype(np.float32)

    has_cube_prev = c["has_cube_prev"].astype(bool)
    ever_had_cube_prev = c["ever_had_cube_prev"].astype(bool)

    orient_approach = c.get("orient_approach", None)
    if orient_approach is not None:
        orient_approach = orient_approach.astype(np.float32)

    # tip distance (optional)
    dist_left_tip_box = c.get("dist_left_fingertip_box", None)
    dist_right_tip_box = c.get("dist_right_fingertip_box", None)
    if dist_left_tip_box is not None and dist_right_tip_box is not None:
        tip_dist_max = np.maximum(
            np.asarray(dist_left_tip_box, dtype=np.float32),
            np.asarray(dist_right_tip_box, dtype=np.float32),
        ).astype(np.float32)
    else:
        tip_dist_max = None

    steps = info.get("steps_for_reward", info.get("steps", np.zeros(num_envs, dtype=np.int32)))
    steps = np.asarray(steps, dtype=np.int32)

    # ✅ 关键：在很多实现里 reset 后第一步 steps==1（而不是 0）
    # 为了避免 info 复用/部分 reset 导致 best_* 跨 episode 饱和，这里强制把 steps<=1 视为“新 episode 开始”
    reset_ep = (steps <= 1)

    max_abs = float(getattr(cfg_r, "max_step_reward", 200.0))

    # -------------------- thresholds for stable grasp (keep old keys) --------------------
    force_threshold = _getf(cfg_r, "stage1_force_threshold", 1.5)
    height_threshold = _getf(cfg_r, "stage1_height_threshold", 0.02)
    grasp_dist_threshold = _getf(cfg_r, "stage1_grasp_dist_threshold", 0.03)

    gap_min = _getf(cfg_r, "stage1_grasp_gap_min", 0.018)
    gap_max = _getf(cfg_r, "stage1_grasp_gap_max", 0.031)

    stay_dist_mult = _getf(cfg_r, "stage1_stay_dist_multiplier", 1.25)
    stay_force_mult = _getf(cfg_r, "stage1_stay_force_multiplier", 0.85)
    stay_gap_margin = _getf(cfg_r, "stage1_stay_gap_margin", 0.003)
    stay_h_margin = _getf(cfg_r, "stage1_stay_height_margin", 0.006)

    # used for safety/drop gating
    lift_start_h = _getf(cfg_r, "stage1_lift_start_height", 0.002)
    band_low = _getf(cfg_r, "lift_band_low_rwdfunc", 0.03)

    # -------------------- orientation gate (multiplier only) --------------------
    _, orient_gate = orient_upright_bonus_and_gate(
        cfg_r=cfg_r,
        orient_approach=orient_approach,
        dist_center_box=dist_center_box,
    )
    orient_gate = np.asarray(orient_gate, dtype=np.float32)

    use_ori_mult = _getb(cfg_r, "simple_use_orient_multiplier_rwdfunc", True)
    g_ori = orient_gate if use_ori_mult else np.ones_like(orient_gate, dtype=np.float32)

    # -------------------- pinch quality (q_t) --------------------
    # q_t is computed WITHOUT orientation gate, so orientation only affects reward via multiplier.
    pinch_quality, pinch_active, balance, tip_gate = pinch_quality_geommean(
        cfg_r=cfg_r,
        finger_gap_val=finger_gap_val,
        dist_center_box=dist_center_box,
        left_force=left_force,
        right_force=right_force,
        tip_dist_max=tip_dist_max,
        orient_gate=np.ones_like(dist_center_box, dtype=np.float32),
        gap_min=gap_min,
        gap_max=gap_max,
        grasp_dist_threshold=grasp_dist_threshold,
        stay_gap_margin=stay_gap_margin,
        force_threshold=force_threshold,
        eps=1e-6,
    )
    pinch_quality = np.asarray(pinch_quality, dtype=np.float32)
    pinch_active = np.asarray(pinch_active, dtype=bool)

    stable_grasp_candidate, stable_grasp_now = stable_grasp_with_pinch(
        cfg_r=cfg_r,
        has_cube_prev=has_cube_prev,
        finger_gap_val=finger_gap_val,
        left_force=left_force,
        right_force=right_force,
        dist_center_box=dist_center_box,
        center_z=c["fingertip_center_pos"][:, 2].astype(np.float32),
        tip_dist_max=tip_dist_max,
        orient_approach=orient_approach,
        pinch_quality=pinch_quality,
        gap_min=gap_min,
        gap_max=gap_max,
        force_threshold=force_threshold,
        height_threshold=height_threshold,
        grasp_dist_threshold=grasp_dist_threshold,
        stay_dist_mult=stay_dist_mult,
        stay_force_mult=stay_force_mult,
        stay_gap_margin=stay_gap_margin,
        stay_h_margin=stay_h_margin,
    )
    stable_grasp_now = np.asarray(stable_grasp_now, dtype=bool)
    has_cube_now = stable_grasp_now.astype(bool)
    ever_had_cube = np.logical_or(ever_had_cube_prev, has_cube_now).astype(bool)

    # -------------------- grasp_ready via hold counter --------------------
    hold_counter_prev = np.asarray(info.get("hold_counter", np.zeros(num_envs, dtype=np.int32)), dtype=np.int32)
    hold_counter_prev = np.where(reset_ep, 0, hold_counter_prev).astype(np.int32)

    hold_counter = update_hold_counter(cfg_r=cfg_r, stable_grasp_now=stable_grasp_now, hold_counter_prev=hold_counter_prev)

    grasp_pre_ready, grasp_ready, _hold_build_unused, _hold_maintain_unused = grasp_hold_rewards(
        cfg_r=cfg_r,
        stable_grasp_now=stable_grasp_now,
        hold_counter=hold_counter,
    )
    grasp_ready = np.asarray(grasp_ready, dtype=bool)

    grasp_ready_prev = np.asarray(info.get("stage1_grasp_ready_prev", np.zeros(num_envs, dtype=bool)), dtype=bool)
    grasp_ready_prev = np.where(reset_ep, False, grasp_ready_prev).astype(bool)

    grasp_ready_steps_prev = np.asarray(info.get("stage1_grasp_ready_steps", np.zeros(num_envs, dtype=np.int32)), dtype=np.int32)
    grasp_ready_steps_prev = np.where(reset_ep, 0, grasp_ready_steps_prev).astype(np.int32)
    grasp_ready_steps = np.where(grasp_ready, grasp_ready_steps_prev + 1, 0).astype(np.int32)

    # -------------------- contact / milestone signals --------------------
    single_force_th = _getf(cfg_r, "single_force_th_rwdfunc", force_threshold, old_key="stage1_single_force_threshold")
    single_dist_th = _getf(cfg_r, "single_dist_th_rwdfunc", 0.05, old_key="stage1_single_dist_threshold")

    left_contact = left_force >= single_force_th
    right_contact = right_force >= single_force_th
    any_contact_with_cube = (left_contact | right_contact) & (dist_center_box <= single_dist_th)

    # ---- one-time milestones (4 total) ----
    # 1) first_contact
    first_contact_prev = np.asarray(info.get("stage1_first_contact_achieved", np.zeros(num_envs, dtype=bool)), dtype=bool)
    first_contact_prev = np.where(reset_ep, False, first_contact_prev).astype(bool)
    first_contact_r, first_contact_achieved, _ = success_once_reward(
        cfg_r=cfg_r,
        success_mask=any_contact_with_cube,
        achieved_prev=first_contact_prev,
        reward_key="first_contact_reward_rwdfunc",
        default_reward=_getf(cfg_r, "first_contact_reward_default_rwdfunc", 1.5),
    )

    # 2) first_pinch_active
    pinch_once_prev = np.asarray(info.get("stage1_pinch_active_achieved", np.zeros(num_envs, dtype=bool)), dtype=bool)
    pinch_once_prev = np.where(reset_ep, False, pinch_once_prev).astype(bool)
    pinch_active_once_r, pinch_active_once_achieved, _ = success_once_reward(
        cfg_r=cfg_r,
        success_mask=pinch_active,
        achieved_prev=pinch_once_prev,
        reward_key="pinch_active_once_reward_rwdfunc",
        default_reward=_getf(cfg_r, "pinch_active_once_reward_default_rwdfunc", 3.0),
    )

    # 3) first_grasp_ready
    # ✅ 保护：如果有人把 grasp_success_reward_rwdfunc 设成负数，默认按 0 处理（除非显式允许）
    grasp_achieved_prev = np.asarray(info.get("stage1_grasp_achieved", np.zeros(num_envs, dtype=bool)), dtype=bool)
    grasp_achieved_prev = np.where(reset_ep, False, grasp_achieved_prev).astype(bool)

    default_grasp_once = _getf(cfg_r, "stage1_success_reward", 0.3 * max_abs)
    grasp_once_val = float(getattr(cfg_r, "grasp_success_reward_rwdfunc", default_grasp_once))
    allow_neg = bool(getattr(cfg_r, "allow_negative_grasp_ready_once_rwdfunc", False))
    if (grasp_once_val < 0.0) and (not allow_neg):
        grasp_once_val = 0.0

    grasp_ready_new = (grasp_ready & (~grasp_achieved_prev)).astype(bool)
    grasp_ready_once_r = (grasp_once_val * grasp_ready_new.astype(np.float32)).astype(np.float32)
    grasp_achieved = (grasp_achieved_prev | grasp_ready).astype(bool)

    # -------------------- best-so-far progress terms --------------------
    # ✅ steps<=1 时强制把 best_* “拉回当前”，避免跨 episode / 部分 reset 饱和
    # b_d: best (min) distance to cube center (fingertip center)
    best_dist_prev = np.asarray(info.get("stage1_best_dist_center_box", dist_center_box.copy()), dtype=np.float32)
    best_dist_prev = np.where(reset_ep, dist_center_box, best_dist_prev).astype(np.float32)
    best_dist_now = np.minimum(best_dist_prev, dist_center_box).astype(np.float32)
    delta_bd = (best_dist_prev - best_dist_now).astype(np.float32)  # >=0

    # b_q: best (max) pinch quality
    best_q_prev = np.asarray(info.get("stage1_best_pinch_quality", pinch_quality.copy()), dtype=np.float32)
    best_q_prev = np.where(reset_ep, pinch_quality, best_q_prev).astype(np.float32)
    best_q_now = np.maximum(best_q_prev, pinch_quality).astype(np.float32)
    delta_bq = (best_q_now - best_q_prev).astype(np.float32)  # >=0

    # b_h: best (max) box height (relative)
    best_h_prev = np.asarray(info.get("stage1_best_box_height", box_height.copy()), dtype=np.float32)
    best_h_prev = np.where(reset_ep, box_height, best_h_prev).astype(np.float32)
    best_h_now = np.maximum(best_h_prev, box_height).astype(np.float32)
    delta_bh = (best_h_now - best_h_prev).astype(np.float32)  # >=0

    # weights
    w_d = _getf(cfg_r, "simple_w_d_rwdfunc", 0.2)
    w_q = _getf(cfg_r, "simple_w_q_rwdfunc", 0.05)
    w_h = _getf(cfg_r, "simple_w_h_rwdfunc", 10.0)

    # ✅ 归一化尺度（把 tiny Δ 映射到更稳定量纲）
    dist_scale = _getf(cfg_r, "simple_dist_scale_rwdfunc", grasp_dist_threshold)
    dist_scale = float(max(dist_scale, 1e-6))
    # lift_goal_h 需要先拿到（见下方 lift 部分），这里先用一个合理默认，后面会覆盖
    lift_goal_h_tmp = _getf(cfg_r, "lift_goal_h_rwdfunc", _getf(cfg_r, "stage1_lift_goal_height", 0.03))
    height_scale = _getf(cfg_r, "simple_height_scale_rwdfunc", lift_goal_h_tmp)
    height_scale = float(max(height_scale, 1e-6))

    pre_mask = (~grasp_ready).astype(np.float32)
    post_mask = grasp_ready.astype(np.float32)

    progress_pre = (pre_mask * g_ori * (w_d * (delta_bd / dist_scale) + w_q * delta_bq)).astype(np.float32)
    progress_lift = (post_mask * (w_h * (delta_bh / height_scale))).astype(np.float32)

    # ✅ NEW：hold_counter build 进度奖励（只奖励 build 前几步，避免抓稳后原地刷分）
    hold_steps_cap = _geti(cfg_r, "hold_steps_rwdfunc", 3, old_key="stage1_hold_steps")
    hold_steps_cap = max(hold_steps_cap, 1)
    w_hold = _getf(cfg_r, "simple_w_hold_rwdfunc", 0.2)
    hold_build_mask = stable_grasp_now & (hold_counter_prev < hold_steps_cap)
    hold_progress = (w_hold * hold_build_mask.astype(np.float32)).astype(np.float32)

    # -------------------- lift success (success & terminate milestone) --------------------
    lift_hold_counter_prev = np.asarray(info.get("lift_hold_counter", np.zeros(num_envs, dtype=np.int32)), dtype=np.int32)
    lift_hold_counter_prev = np.where(reset_ep, 0, lift_hold_counter_prev).astype(np.int32)

    lift_goal_h = _getf(cfg_r, "lift_goal_h_rwdfunc", _getf(cfg_r, "stage1_lift_goal_height", 0.03))
    at_goal = grasp_ready & has_cube_now & (box_height >= lift_goal_h)
    lift_hold_counter = np.where(at_goal, lift_hold_counter_prev + 1, 0).astype(np.int32)

    lift_success = lift_success_mask(cfg_r=cfg_r, box_height=box_height, lift_hold_counter=lift_hold_counter)
    lift_achieved_prev = np.asarray(info.get("stage1_lift_achieved", np.zeros(num_envs, dtype=bool)), dtype=bool)
    lift_achieved_prev = np.where(reset_ep, False, lift_achieved_prev).astype(bool)
    lift_success_r, lift_achieved, lift_success_new = success_once_reward(
        cfg_r=cfg_r,
        success_mask=lift_success,
        achieved_prev=lift_achieved_prev,
        reward_key="lift_success_reward_rwdfunc",
        default_reward=_getf(cfg_r, "stage1_lift_success_reward", 0.6 * max_abs),
    )

    # -------------------- sparse safety penalties --------------------
    _fp, _up, deep_floor, floor_violation, under_box_mask = floor_underbox_penalties(
        cfg_r=cfg_r,
        ee_z=ee_pos[:, 2].astype(np.float32),
        center_z=c["fingertip_center_pos"][:, 2].astype(np.float32),
        dist_center_box=dist_center_box,
        box_pos_z=box_pos[:, 2].astype(np.float32),
        pregrasp_dist=_getf(cfg_r, "upright_bonus_near_dist_rwdfunc", 0.06),
    )
    deep_floor = np.asarray(deep_floor, dtype=bool)

    deep_floor_mag = _getf(cfg_r, "simple_deep_floor_penalty_mag_rwdfunc", 2.0)
    deep_floor_penalty = (-deep_floor_mag * deep_floor.astype(np.float32)).astype(np.float32)

    # drop event after lift (reuse existing drop detector; sparse + cooldown)
    lift_phase_prev = np.asarray(info.get("stage1_lift_phase", np.zeros(num_envs, dtype=bool)), dtype=bool)
    lift_phase_prev = np.where(reset_ep, False, lift_phase_prev).astype(bool)
    lift_phase_now = (lift_phase_prev | (best_h_now >= lift_start_h) | (box_height >= band_low)).astype(bool)

    drop_cd_prev = np.asarray(info.get("stage1_drop_cooldown", np.zeros(num_envs, dtype=np.int32)), dtype=np.int32)
    drop_cd_prev = np.where(reset_ep, 0, drop_cd_prev).astype(np.int32)
    drop_penalty, drop_cd = drop_penalty_once(
        cfg_r=cfg_r,
        has_cube_prev=has_cube_prev,
        has_cube_now=has_cube_now,
        box_height=box_height,
        lift_phase=lift_phase_now,
        cooldown_prev=drop_cd_prev,
        grasp_ready_prev=grasp_ready_prev,
        hold_counter_prev=hold_counter_prev,
    )

    # no_contact / no_grasp: terminate+once is recommended and controlled in cfg
    steps_since_contact_prev = np.asarray(info.get("steps_since_contact", np.zeros(num_envs, dtype=np.int32)), dtype=np.int32)
    steps_since_grasp_prev = np.asarray(info.get("steps_since_grasp", np.zeros(num_envs, dtype=np.int32)), dtype=np.int32)
    steps_since_contact_prev = np.where(reset_ep, 0, steps_since_contact_prev).astype(np.int32)
    steps_since_grasp_prev = np.where(reset_ep, 0, steps_since_grasp_prev).astype(np.int32)

    trig_contact_prev = np.asarray(info.get("stage1_no_contact_triggered", np.zeros(num_envs, dtype=bool)), dtype=bool)
    trig_grasp_prev = np.asarray(info.get("stage1_no_grasp_triggered", np.zeros(num_envs, dtype=bool)), dtype=bool)
    trig_contact_prev = np.where(reset_ep, False, trig_contact_prev).astype(bool)
    trig_grasp_prev = np.where(reset_ep, False, trig_grasp_prev).astype(bool)

    (
        no_contact_penalty,
        no_grasp_penalty,
        steps_since_contact,
        steps_since_grasp,
        no_contact_term,
        no_grasp_term,
        trig_contact,
        trig_grasp,
    ) = update_no_contact_no_grasp(
        cfg_r=cfg_r,
        any_contact_with_cube=any_contact_with_cube,
        has_cube_now=has_cube_now,
        ever_had_cube=ever_had_cube,
        steps_since_contact_prev=steps_since_contact_prev,
        steps_since_grasp_prev=steps_since_grasp_prev,
        triggered_contact_prev=trig_contact_prev,
        triggered_grasp_prev=trig_grasp_prev,
        return_termination=True,
    )

    # -------------------- total reward --------------------
    total_reward = (
        # progress (best-so-far deltas)
        progress_pre
        + progress_lift
        + hold_progress
        # milestones (one-time)
        + first_contact_r
        + pinch_active_once_r
        + grasp_ready_once_r
        + lift_success_r
        # safety (sparse)
        + deep_floor_penalty
        + drop_penalty
        + no_contact_penalty
        + no_grasp_penalty
    ).astype(np.float32)

    # -------------------- termination --------------------
    terminated = np.zeros(num_envs, dtype=bool)

    terminate_on_grasp = _getb(cfg_r, "stage1_terminate_on_grasp_success", False)
    terminate_on_lift = _getb(cfg_r, "stage1_terminate_on_lift_success", True)

    if terminate_on_grasp:
        terminated |= grasp_ready_new.astype(bool)
    if terminate_on_lift:
        terminated |= lift_success_new.astype(bool)

    # failure-type terminations
    terminated |= deep_floor
    terminated |= np.asarray(no_contact_term, dtype=bool)
    terminated |= np.asarray(no_grasp_term, dtype=bool)

    # -------------------- invalid states --------------------
    invalid_reward = np.isnan(total_reward) | np.isinf(total_reward)
    invalid_state = np.isnan(dof_pos).any(axis=1) | np.isnan(dof_vel).any(axis=1)
    invalid = invalid_reward | invalid_state
    if np.any(invalid):
        terminated |= invalid
        invalid_penalty = float(getattr(cfg_r, "invalid_penalty", -30.0))
        total_reward[invalid] = invalid_penalty

    total_reward = np.clip(total_reward, -max_abs, max_abs).astype(np.float32)

    # -------------------- update info (keep common keys) --------------------
    info["has_cube"] = has_cube_now.astype(bool)
    info["ever_had_cube"] = ever_had_cube.astype(bool)

    info["prev_dist_ee_box"] = dist_ee_box.astype(np.float32)
    info["prev_dist_center_box"] = dist_center_box.astype(np.float32)
    info["prev_box_height"] = box_height.astype(np.float32)

    info["hold_counter"] = hold_counter.astype(np.int32)
    info["stage1_grasp_ready_prev"] = grasp_ready.astype(bool)
    info["stage1_grasp_ready_steps"] = grasp_ready_steps.astype(np.int32)
    info["stage1_lift_phase"] = lift_phase_now.astype(bool)

    info["lift_hold_counter"] = lift_hold_counter.astype(np.int32)
    info["stage1_drop_cooldown"] = np.asarray(drop_cd, dtype=np.int32)

    info["steps_since_contact"] = np.asarray(steps_since_contact, dtype=np.int32)
    info["steps_since_grasp"] = np.asarray(steps_since_grasp, dtype=np.int32)
    info["stage1_no_contact_triggered"] = np.asarray(trig_contact, dtype=bool)
    info["stage1_no_grasp_triggered"] = np.asarray(trig_grasp, dtype=bool)

    # milestone achieved flags
    info["stage1_first_contact_achieved"] = np.asarray(first_contact_achieved, dtype=bool)
    info["stage1_pinch_active_achieved"] = np.asarray(pinch_active_once_achieved, dtype=bool)
    info["stage1_grasp_achieved"] = np.asarray(grasp_achieved, dtype=bool)
    info["stage1_lift_achieved"] = np.asarray(lift_achieved, dtype=bool)

    # best-so-far buffers
    info["stage1_best_dist_center_box"] = best_dist_now.astype(np.float32)
    info["stage1_best_pinch_quality"] = best_q_now.astype(np.float32)
    info["stage1_best_box_height"] = best_h_now.astype(np.float32)

    # -------------------- logs --------------------
    reward_terms = {
        # progress (best-so-far)
        "best_dist_improve": (w_d * (delta_bd / dist_scale) * pre_mask * g_ori).astype(np.float32),
        "best_pinchq_improve": (w_q * delta_bq * pre_mask * g_ori).astype(np.float32),
        "best_height_improve": (w_h * (delta_bh / height_scale) * post_mask).astype(np.float32),
        "hold_build_progress": np.asarray(hold_progress, dtype=np.float32),
        # milestones
        "first_contact_once": np.asarray(first_contact_r, dtype=np.float32),
        "pinch_active_once": np.asarray(pinch_active_once_r, dtype=np.float32),
        "grasp_ready_once": np.asarray(grasp_ready_once_r, dtype=np.float32),
        "lift_success_once": np.asarray(lift_success_r, dtype=np.float32),
        # safety
        "deep_floor_penalty": np.asarray(deep_floor_penalty, dtype=np.float32),
        "drop_penalty": np.asarray(drop_penalty, dtype=np.float32),
        "no_contact_penalty": np.asarray(no_contact_penalty, dtype=np.float32),
        "no_grasp_penalty": np.asarray(no_grasp_penalty, dtype=np.float32),
    }
    if np.any(invalid):
        reward_terms["invalid_penalty"] = (
            float(getattr(cfg_r, "invalid_penalty", -30.0)) * invalid.astype(np.float32)
        ).astype(np.float32)

    metrics = {
        "dist_center_box": dist_center_box.astype(np.float32),
        "dist_ee_box": dist_ee_box.astype(np.float32),
        "box_height": box_height.astype(np.float32),
        "finger_gap": finger_gap_val.astype(np.float32),
        "left_force": left_force.astype(np.float32),
        "right_force": right_force.astype(np.float32),
        "pinch_quality": pinch_quality.astype(np.float32),
        "pinch_active": pinch_active.astype(np.float32),
        "orient_gate": orient_gate.astype(np.float32),
        "grasp_ready": grasp_ready.astype(np.float32),
        "has_cube": has_cube_now.astype(np.float32),
        "ever_had_cube": ever_had_cube.astype(np.float32),
        "hold_counter": hold_counter.astype(np.float32),
        "best_dist": best_dist_now.astype(np.float32),
        "best_pinch_quality": best_q_now.astype(np.float32),
        "best_height": best_h_now.astype(np.float32),
        "delta_best_dist": delta_bd.astype(np.float32),
        "delta_best_pinch_quality": delta_bq.astype(np.float32),
        "delta_best_height": delta_bh.astype(np.float32),
        "any_contact": any_contact_with_cube.astype(np.float32),
        "deep_floor": deep_floor.astype(np.float32),
        "floor_violation": np.asarray(floor_violation, dtype=np.float32),
        "under_box": np.asarray(under_box_mask, dtype=np.float32),
        "no_contact_term": np.asarray(no_contact_term, dtype=np.float32),
        "no_grasp_term": np.asarray(no_grasp_term, dtype=np.float32),
        "reset_ep": reset_ep.astype(np.float32),
    }
    if np.any(invalid):
        metrics["invalid"] = invalid.astype(np.float32)

    return total_reward, terminated, has_cube_now.astype(bool), reward_terms, metrics

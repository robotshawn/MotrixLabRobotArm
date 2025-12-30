# motrix_envs/src/motrix_envs/manipulation/panda/drop_functions.py
import numpy as np


def _get_float_config_value(reward_configuration, key: str, default: float, old_key: str | None = None) -> float:
    if old_key is None:
        return float(getattr(reward_configuration, key, default))
    return float(getattr(reward_configuration, key, getattr(reward_configuration, old_key, default)))


def _clip_to_unit_interval(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def _score_near_distance(distance_value: np.ndarray, distance_scale, eps: float = 1e-6) -> np.ndarray:
    """score_near(distance; scale) = clip01(1 - distance / scale) ; scale can be float or (N,) array"""
    distance_value = np.asarray(distance_value, dtype=np.float32)
    distance_scale = np.asarray(distance_scale, dtype=np.float32)
    distance_scale = np.clip(distance_scale, eps, None)
    return _clip_to_unit_interval(1.0 - distance_value / distance_scale)


def _load_info_array(info: dict, key: str, fallback: np.ndarray, dtype, number_of_environments: int) -> np.ndarray:
    """Load 1D array from info with shape fallback."""
    v = info.get(key, fallback)
    arr = np.asarray(v, dtype=dtype)
    arr = np.atleast_1d(arr)
    if arr.shape[0] != number_of_environments:
        arr = np.asarray(fallback, dtype=dtype)
        arr = np.atleast_1d(arr)
        if arr.shape[0] != number_of_environments:
            arr = arr.reshape(number_of_environments)
    return arr


def _update_best_min_distance_state(
    *,
    info: dict,
    key_best_distance: str,
    distance_now: np.ndarray,
    number_of_environments: int,
    reset_episode: np.ndarray,
    segment_reset: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    维护“历史最小距离 best_distance”状态（min-distance progress 的必要状态）：
      - reset/切段：best_prev 对齐为当前 distance_now
      - 非 active 段：best_prev 也对齐为当前 distance_now（避免重新进入时白嫖）
      - active 段：best_current = min(best_prev, distance_now)
    """
    distance_now = np.asarray(distance_now, dtype=np.float32)
    reset_episode = np.asarray(reset_episode, dtype=bool)
    segment_reset = np.asarray(segment_reset, dtype=bool)
    active_mask = np.asarray(active_mask, dtype=bool)

    best_prev = _load_info_array(info, key_best_distance, distance_now.copy(), np.float32, number_of_environments)
    best_prev = np.where(reset_episode | segment_reset, distance_now, best_prev).astype(np.float32)
    best_prev = np.where(~active_mask, distance_now, best_prev).astype(np.float32)

    best_cur = np.where(active_mask, np.minimum(best_prev, distance_now), best_prev).astype(np.float32)
    return best_prev, best_cur


def compute_target_progress_before_drop(
    *,
    reward_configuration,
    info: dict,
    number_of_environments: int,
    reset_episode: np.ndarray,
    lift_success_just_achieved: np.ndarray,
    post_lift_before_drop_mask: np.ndarray,
    fingertip_center_to_target_distance_3d: np.ndarray,
) -> dict:
    """
    lift 成功后 & drop 前：
      - 使用 fingertip_center -> target 的 3D 距离
      - ✅ 仅保留 min-distance progress：只奖励刷新历史最小距离（节省计算资源）
    """
    reset_episode = np.asarray(reset_episode, dtype=bool)
    lift_success_just_achieved = np.asarray(lift_success_just_achieved, dtype=bool)
    post_lift_before_drop_mask = np.asarray(post_lift_before_drop_mask, dtype=bool)
    fingertip_center_to_target_distance_3d = np.asarray(fingertip_center_to_target_distance_3d, dtype=np.float32)

    # --- snapshot at lift success (for scale) ---
    lift_success_ftcenter_target_snapshot_previous = np.asarray(
        info.get(
            "reach_grasp_transport_lift_success_ftcenter_to_target_distance_3d_snapshot",
            fingertip_center_to_target_distance_3d.copy(),
        ),
        dtype=np.float32,
    )
    if lift_success_ftcenter_target_snapshot_previous.shape[0] != number_of_environments:
        lift_success_ftcenter_target_snapshot_previous = fingertip_center_to_target_distance_3d.copy().astype(np.float32)
    lift_success_ftcenter_target_snapshot_previous = np.where(
        reset_episode,
        fingertip_center_to_target_distance_3d,
        lift_success_ftcenter_target_snapshot_previous,
    ).astype(np.float32)

    lift_success_ftcenter_target_snapshot_current = np.where(
        lift_success_just_achieved,
        fingertip_center_to_target_distance_3d,
        lift_success_ftcenter_target_snapshot_previous,
    ).astype(np.float32)

    target_progress_distance_scale_previous = np.asarray(
        info.get(
            "reach_grasp_transport_target_progress_distance_scale",
            np.ones(number_of_environments, dtype=np.float32) * 0.30,
        ),
        dtype=np.float32,
    )
    if target_progress_distance_scale_previous.shape[0] != number_of_environments:
        target_progress_distance_scale_previous = np.ones((number_of_environments,), dtype=np.float32) * 0.30

    target_distance_scale_alpha = _get_float_config_value(reward_configuration, "rg3_m_t_alpha_rwdfunc", 0.9, old_key="rg3_m_xy_alpha_rwdfunc")
    target_distance_scale_minimum = _get_float_config_value(reward_configuration, "rg3_m_t_min_rwdfunc", 0.08, old_key="rg3_m_xy_min_rwdfunc")
    target_distance_scale_maximum = _get_float_config_value(reward_configuration, "rg3_m_t_max_rwdfunc", 0.45, old_key="rg3_m_xy_max_rwdfunc")

    target_progress_distance_scale_current = np.where(
        lift_success_just_achieved,
        np.clip(
            target_distance_scale_alpha * lift_success_ftcenter_target_snapshot_current,
            target_distance_scale_minimum,
            target_distance_scale_maximum,
        ).astype(np.float32),
        target_progress_distance_scale_previous,
    ).astype(np.float32)

    # --- score (for debug / best logging, still kept) ---
    target_progress_score = _score_near_distance(
        fingertip_center_to_target_distance_3d,
        target_progress_distance_scale_current,
    )

    # ------------------------------------------------------------------
    # min-distance progress state
    # ------------------------------------------------------------------
    best_target_distance_previous, best_target_distance_current = _update_best_min_distance_state(
        info=info,
        key_best_distance="reach_grasp_transport_best_target_distance_3d",
        distance_now=fingertip_center_to_target_distance_3d,
        number_of_environments=number_of_environments,
        reset_episode=reset_episode,
        segment_reset=lift_success_just_achieved,
        active_mask=post_lift_before_drop_mask,
    )

    min_dist_delta = _get_float_config_value(reward_configuration, "rg3_delta_t3_min_dist_rwdfunc", 1e-4)
    target_distance_best_improvement = np.maximum(
        0.0,
        (best_target_distance_previous - best_target_distance_current - float(max(min_dist_delta, 0.0))),
    ).astype(np.float32)

    best_target_progress_score_previous = np.asarray(
        info.get("reach_grasp_transport_best_target_progress_score", np.zeros(number_of_environments, dtype=np.float32)),
        dtype=np.float32,
    )
    if best_target_progress_score_previous.shape[0] != number_of_environments:
        best_target_progress_score_previous = np.zeros((number_of_environments,), dtype=np.float32)
    best_target_progress_score_previous = np.where(
        reset_episode | lift_success_just_achieved,
        0.0,
        best_target_progress_score_previous,
    ).astype(np.float32)

    best_target_progress_score_current = np.where(
        post_lift_before_drop_mask,
        np.maximum(best_target_progress_score_previous, target_progress_score),
        best_target_progress_score_previous,
    ).astype(np.float32)

    # 保留 best improvement（仅 debug）
    minimum_target_progress_delta = _get_float_config_value(reward_configuration, "rg3_delta_t3_rwdfunc", 1e-4)
    target_progress_best_improvement = np.maximum(
        0.0,
        (best_target_progress_score_current - best_target_progress_score_previous - float(max(minimum_target_progress_delta, 0.0))),
    ).astype(np.float32)

    # --- min-distance progress reward ---
    target_progress_weight = _get_float_config_value(reward_configuration, "rg3_w_t3_rwdfunc", 10.0)
    min_dist_weight = _get_float_config_value(reward_configuration, "rg3_w_t3_min_dist_rwdfunc", target_progress_weight)

    # ✅ 只保留 min-distance；step cap 用 min-dist 的 cap key
    step_cap = _get_float_config_value(reward_configuration, "rg3_t3_min_dist_step_cap_rwdfunc", 0.0)

    target_progress_reward_min_dist = (
        min_dist_weight
        * target_distance_best_improvement
        * post_lift_before_drop_mask.astype(np.float32)
    ).astype(np.float32)

    if step_cap > 0.0:
        cap = float(step_cap)
        target_progress_reward_min_dist = np.clip(target_progress_reward_min_dist, -cap, cap).astype(np.float32)

    target_progress_reward_before_drop = target_progress_reward_min_dist.astype(np.float32)

    return {
        "target_progress_reward_before_drop": target_progress_reward_before_drop,
        "best_target_progress_score_current": best_target_progress_score_current,
        "target_progress_distance_scale_current": target_progress_distance_scale_current,
        "lift_success_ftcenter_target_snapshot_current": lift_success_ftcenter_target_snapshot_current,
        "target_progress_score": target_progress_score,
        "target_progress_best_improvement": target_progress_best_improvement,

        # min-distance progress debug/state
        "best_target_distance_previous": best_target_distance_previous,
        "best_target_distance_current": best_target_distance_current,
        "target_distance_best_improvement": target_distance_best_improvement,
        "target_progress_reward_min_dist(debug)": target_progress_reward_min_dist,
        "target_progress_reward_mode_is_min_dist": (np.ones((number_of_environments,), dtype=np.float32) * 1.0).astype(np.float32),
    }


def compute_home_progress_after_drop(
    *,
    reward_configuration,
    info: dict,
    number_of_environments: int,
    reset_episode: np.ndarray,
    lift_success_just_achieved: np.ndarray,
    has_dropped_after_lift_current: np.ndarray,
    has_dropped_after_lift_just_now: np.ndarray,
    home_progress_active_mask: np.ndarray,
    end_effector_to_home_distance: np.ndarray,
    box_position_world: np.ndarray,
    target_position_world: np.ndarray,
    fingertip_center_position_world: np.ndarray,
) -> dict:
    """
    drop 后的 home progress：
      - 仅当 home_progress_active_mask=True（由上层 gating 决定）才启用
      - ✅ 仅保留 min-distance progress：只奖励刷新历史最小距离（节省计算资源）
    """
    reset_episode = np.asarray(reset_episode, dtype=bool)
    lift_success_just_achieved = np.asarray(lift_success_just_achieved, dtype=bool)
    has_dropped_after_lift_current = np.asarray(has_dropped_after_lift_current, dtype=bool)
    has_dropped_after_lift_just_now = np.asarray(has_dropped_after_lift_just_now, dtype=bool)
    home_progress_active_mask = np.asarray(home_progress_active_mask, dtype=bool)
    end_effector_to_home_distance = np.asarray(end_effector_to_home_distance, dtype=np.float32)

    # home phase start：drop 刚发生且 home_progress_active_mask=True
    home_phase_start = (has_dropped_after_lift_just_now & home_progress_active_mask).astype(bool)

    # -------------------------
    # best(min-distance) state（min-dist progress 必要状态；也可 debug）
    # -------------------------
    best_home_distance_previous, best_home_distance_current = _update_best_min_distance_state(
        info=info,
        key_best_distance="reach_grasp_transport_best_end_effector_to_home_distance",
        distance_now=end_effector_to_home_distance,
        number_of_environments=number_of_environments,
        reset_episode=reset_episode,
        segment_reset=(lift_success_just_achieved | home_phase_start),
        active_mask=home_progress_active_mask,
    )

    minimum_home_distance_delta = _get_float_config_value(
        reward_configuration,
        "rg3_delta_home_min_dist_rwdfunc",
        _get_float_config_value(reward_configuration, "rg3_home_dist_delta_rwdfunc", 1e-4),
    )
    home_distance_best_improvement = np.maximum(
        0.0,
        (best_home_distance_previous - best_home_distance_current - float(max(minimum_home_distance_delta, 0.0))),
    ).astype(np.float32)

    # ✅ only min-distance progress
    home_weight = _get_float_config_value(
        reward_configuration,
        "rg3_w_home_min_dist_rwdfunc",
        _get_float_config_value(reward_configuration, "rg3_w_home_dist_rwdfunc", 8.0),
    )

    home_step_cap = _get_float_config_value(
        reward_configuration,
        "rg3_home_min_dist_step_cap_rwdfunc",
        _get_float_config_value(reward_configuration, "rg3_home_dist_step_cap_rwdfunc", 0.0),
    )

    home_progress_reward = (
        home_weight
        * home_distance_best_improvement
        * home_progress_active_mask.astype(np.float32)
    ).astype(np.float32)

    if home_step_cap > 0.0:
        cap = float(home_step_cap)
        home_progress_reward = np.clip(home_progress_reward, -cap, cap).astype(np.float32)

    return {
        "best_home_distance_previous": best_home_distance_previous,
        "best_home_distance_current": best_home_distance_current,
        "home_distance_best_improvement": home_distance_best_improvement,
        "home_progress_reward": home_progress_reward,
        "home_progress_reward_mode_is_min_dist": (np.ones((number_of_environments,), dtype=np.float32) * 1.0).astype(np.float32),
    }

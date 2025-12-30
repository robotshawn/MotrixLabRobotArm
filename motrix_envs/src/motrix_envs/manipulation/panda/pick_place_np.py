# ===== motrix_envs/src/motrix_envs/manipulation/panda/pick_place_np.py =====
import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.manipulation.panda.cfg import PandaPickPlaceEnvCfg
from motrix_envs.np.env import NpEnv, NpEnvState
from .pick_place_reward_np import compute_reward_and_termination


@registry.env("panda-pick-place", sim_backend="np")
@registry.env("panda-pick-place-reach-only", sim_backend="np")
@registry.env("panda-pick-place-reach-grasp", sim_backend="np")
@registry.env("panda-pick-place-grasp-lift", sim_backend="np")
@registry.env("panda-pick-place-reach-grasp-transport", sim_backend="np")
class PandaPickPlaceTask(NpEnv):
    """
    Panda 抓取 & 放置任务（触觉 + 指令 + 连续夹爪开度控制 + 阶段化 reward）.
    """

    # ✅ ee_home_pos 固定偏移量（按你的要求写死）
    _EE_HOME_OFFSET_BASE = (0.2, 0.2, 0.6)

    def __init__(self, cfg: PandaPickPlaceEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)

        self._initialize_action_space()
        self._initialize_observation_space()

        self._number_of_action_dimensions = self._action_space.shape[0]
        self._number_of_observation_dimensions = self._observation_space.shape[0]
        self._number_of_degrees_of_freedom_positions = self._model.num_dof_pos
        self._number_of_degrees_of_freedom_velocities = self._model.num_dof_vel

        self._initial_degrees_of_freedom_velocities = np.zeros(
            (self._number_of_degrees_of_freedom_velocities,), dtype=np.float32
        )
        self._initial_degrees_of_freedom_positions = self._model.compute_init_dof_pos()

        # 模型末尾 7 个 dof 属于 box（或其它非机器人自由度），所以机器人 dof = total - 7
        self._number_of_robot_degrees_of_freedom = self._model.num_dof_pos - 7
        self._box_degrees_of_freedom_start_index = self._number_of_robot_degrees_of_freedom
        self._box_initial_height_world_z = float(
            self._initial_degrees_of_freedom_positions[self._box_degrees_of_freedom_start_index + 2]
        )

        # ✅ 统一“box 初始高度基准”字段名（compute_common.py 依赖）
        #   - 旧代码/新 reward 公共项统一用 env._box_z0 作为参考零点
        #   - 这里直接对齐为 box 的初始世界系高度（reset 时也会回填到该高度）
        self._box_z0 = float(self._box_initial_height_world_z)

        # 夹爪两个关节位于机器人 dof 的末尾两维
        self._finger_degrees_of_freedom_start_index = self._number_of_robot_degrees_of_freedom - 2
        self._finger_degrees_of_freedom_end_index = self._number_of_robot_degrees_of_freedom

        self._initialize_buffers()

    def get_base_pos(self, data: mtx.SceneData) -> np.ndarray:
        base_body = self._model.get_body(self.cfg.asset.base_body_name)
        return base_body.get_position(data).astype(np.float32)

    def _compute_target_pos(self, data: mtx.SceneData) -> np.ndarray:
        base_position_world = self.get_base_pos(data)
        target_offset_from_base = np.asarray(self.cfg.target_offset_base, dtype=np.float32)
        return base_position_world + target_offset_from_base[None, :]

    # ✅ NEW: ee_home_pos 的设置方式与 target_pos 完全一致（base + offset），只是不走 cfg，offset 写死为 (0.2,0.2,0.6)
    def _compute_ee_home_pos(self, data: mtx.SceneData) -> np.ndarray:
        base_position_world = self.get_base_pos(data)
        ee_home_offset_from_base = np.asarray(self._EE_HOME_OFFSET_BASE, dtype=np.float32)
        return base_position_world + ee_home_offset_from_base[None, :]

    def _initialize_action_space(self):
        model = self._model
        number_of_actuators = model.num_actuators
        self._action_space = gym.spaces.Box(
            low=-np.ones(number_of_actuators, dtype=np.float32),
            high=np.ones(number_of_actuators, dtype=np.float32),
            dtype=np.float32,
        )

    def _initialize_observation_space(self):
        model = self._model
        self._number_of_robot_degrees_of_freedom = model.num_dof_pos - 7

        number_of_joint_positions = self._number_of_robot_degrees_of_freedom
        number_of_joint_velocities = self._number_of_robot_degrees_of_freedom
        number_of_actions = model.num_actuators
        number_of_box_position_dimensions = 3
        number_of_target_position_dimensions = 3
        number_of_end_effector_position_dimensions = 3
        number_of_finger_force_values = 2

        number_of_observations = (
            number_of_joint_positions
            + number_of_joint_velocities
            + number_of_box_position_dimensions
            + number_of_target_position_dimensions
            + number_of_end_effector_position_dimensions
            + number_of_finger_force_values
            + number_of_actions
        )
        self._observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(number_of_observations,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def get_dof_pos(self, data: mtx.SceneData):
        return data.dof_pos

    def get_dof_vel(self, data: mtx.SceneData):
        return data.dof_vel

    def get_box_pos(self, data: mtx.SceneData):
        box_body = self._model.get_body("box")
        return box_body.get_position(data)

    def get_ee_pos(self, data: mtx.SceneData):
        end_effector_site = self._model.get_site(self.cfg.asset.ee_site_name)
        end_effector_pose = end_effector_site.get_pose(data)
        return end_effector_pose[:, :3]

    def get_finger_forces(self, data: mtx.SceneData) -> np.ndarray:
        asset = self.cfg.asset

        left_finger_force = self._model.get_sensor_value(asset.finger_left_force_sensor, data)
        right_finger_force = self._model.get_sensor_value(asset.finger_right_force_sensor, data)

        if left_finger_force.ndim == 1:
            left_finger_force = left_finger_force.reshape(-1, 1)
        if right_finger_force.ndim == 1:
            right_finger_force = right_finger_force.reshape(-1, 1)

        return np.concatenate([left_finger_force, right_finger_force], axis=1)

    def get_fingertip_positions(
        self,
        data: mtx.SceneData,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        asset = self.cfg.asset

        left_fingertip_site = self._model.get_site(asset.finger_left_touch_site)
        right_fingertip_site = self._model.get_site(asset.finger_right_touch_site)

        left_fingertip_pose = left_fingertip_site.get_pose(data)
        right_fingertip_pose = right_fingertip_site.get_pose(data)

        left_fingertip_position_world = left_fingertip_pose[:, :3]
        right_fingertip_position_world = right_fingertip_pose[:, :3]
        fingertip_center_position_world = 0.5 * (
            left_fingertip_position_world + right_fingertip_position_world
        )

        return (
            left_fingertip_position_world.astype(np.float32),
            right_fingertip_position_world.astype(np.float32),
            fingertip_center_position_world.astype(np.float32),
        )

    def _initialize_buffers(self):
        configuration = self._cfg
        assert isinstance(configuration, PandaPickPlaceEnvCfg)

        self.reset_buf = np.ones(self._num_envs, dtype=bool)

        number_of_actuators = self._model.num_actuators

        # 默认 actuator controls（用于 delta 控制累加）
        self.default_actuator_controls = np.zeros(number_of_actuators, dtype=np.float32)

        actuator_names = getattr(self._model, "actuator_names", [])
        for actuator_index, actuator_name in enumerate(actuator_names):
            for configured_joint_name, configured_value in configuration.init_state.default_joint_angles.items():
                if configured_joint_name in actuator_name:
                    self.default_actuator_controls[actuator_index] = float(configured_value)

        # 若没有匹配到，则至少给手指一个默认开度
        if np.allclose(self.default_actuator_controls, 0.0):
            self.default_actuator_controls[-1] = configuration.init_state.default_joint_angles.get(
                "actuator8", 0.04
            )

        self._finger_actuator_index = self._model.num_actuators - 1
        finger_default_open_value = configuration.init_state.default_joint_angles.get("actuator8", 0.04)
        if self.default_actuator_controls[self._finger_actuator_index] <= 0.0:
            self.default_actuator_controls[self._finger_actuator_index] = float(finger_default_open_value)

        # 最大张开（安全上限）
        self._finger_open_val = float(
            max(self.default_actuator_controls[self._finger_actuator_index], float(finger_default_open_value))
        )
        # 最小闭合（安全下限）
        self._finger_min_val = float(getattr(configuration.control_config, "finger_min_val", 0.0))

        # 连续控制下依然需要“判定开/关”的阈值
        self._finger_closed_threshold = float(
            getattr(
                configuration.control_config,
                "finger_closed_threshold",
                0.25 * self._finger_open_val,
            )
        )
        self._finger_fully_open_threshold = float(
            getattr(
                configuration.control_config,
                "finger_fully_open_threshold",
                0.95 * self._finger_open_val,
            )
        )

        self.default_action_values = np.zeros(number_of_actuators, dtype=np.float32)

        self.ground = self._model.get_geom_index(configuration.asset.ground)
        self.box_geom = self._model.get_geom_index(configuration.asset.box_geom_name)

        self._table_z = 0.0

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """
        连续夹爪控制：
          action[-1] in [-1, 1] -> target_gap in [finger_min_val, finger_open_val]
        同时加入速度限制 finger_speed（每个 ctrl step 最大变化量）和安全 clip。
        """
        clipped_action_values = np.clip(actions, -1.0, 1.0)

        number_of_environments = clipped_action_values.shape[0]
        number_of_actuators = self._model.num_actuators

        state.info.setdefault(
            "last_actions",
            np.zeros((number_of_environments, number_of_actuators), dtype=np.float32),
        )
        state.info.setdefault(
            "current_actions",
            np.zeros((number_of_environments, number_of_actuators), dtype=np.float32),
        )
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = clipped_action_values

        if "ctrls" not in state.info:
            actuator_controls = np.tile(self.default_actuator_controls, (number_of_environments, 1)).astype(np.float32)
        else:
            actuator_controls = state.info["ctrls"].astype(np.float32)

        # ----------- 关节（delta）控制 -----------
        if number_of_actuators > 1:
            joint_delta_actions = clipped_action_values[:, : number_of_actuators - 1]
            actuator_controls[:, : number_of_actuators - 1] += (
                joint_delta_actions * self.cfg.control_config.action_scale
            ).astype(np.float32)

            joint_range_radians = np.pi
            joint_default_controls = self.default_actuator_controls[: number_of_actuators - 1]
            joint_control_lower_bound = joint_default_controls - joint_range_radians
            joint_control_upper_bound = joint_default_controls + joint_range_radians
            actuator_controls[:, : number_of_actuators - 1] = np.clip(
                actuator_controls[:, : number_of_actuators - 1],
                joint_control_lower_bound,
                joint_control_upper_bound,
            )

        # ----------- 夹爪（连续开度）控制 -----------
        finger_action_raw = clipped_action_values[:, self._finger_actuator_index].astype(np.float32)

        # [-1, 1] -> [0, 1] -> [min_val, open_val]
        finger_interpolation_ratio = 0.5 * (finger_action_raw + 1.0)  # 0..1
        finger_target_gap_value = self._finger_min_val + finger_interpolation_ratio * (
            self._finger_open_val - self._finger_min_val
        )

        finger_target_gap_value = np.clip(
            finger_target_gap_value, self._finger_min_val, self._finger_open_val
        ).astype(np.float32)

        previous_finger_control_value = actuator_controls[:, self._finger_actuator_index].astype(np.float32)

        # 速度限制：每个控制步最大变化量
        maximum_gap_change_per_step = float(self.cfg.control_config.finger_speed) * (
            self._finger_open_val - self._finger_min_val
        )
        if maximum_gap_change_per_step <= 0.0:
            finger_control_value = finger_target_gap_value
        else:
            finger_gap_change = np.clip(
                finger_target_gap_value - previous_finger_control_value,
                -maximum_gap_change_per_step,
                maximum_gap_change_per_step,
            ).astype(np.float32)
            finger_control_value = (previous_finger_control_value + finger_gap_change).astype(np.float32)

        finger_control_value = np.clip(
            finger_control_value, self._finger_min_val, self._finger_open_val
        ).astype(np.float32)
        actuator_controls[:, self._finger_actuator_index] = finger_control_value

        state.data.actuator_ctrls = actuator_controls
        state.info["ctrls"] = actuator_controls

        return state

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        configuration = self.cfg

        degrees_of_freedom_positions = self.get_dof_pos(data)
        degrees_of_freedom_velocities = self.get_dof_vel(data)
        box_position_world = self.get_box_pos(data)

        target_position_world = info.get("target_pos", None)
        if target_position_world is None or target_position_world.shape[0] != box_position_world.shape[0]:
            target_position_world = self._compute_target_pos(data)
            info["target_pos"] = target_position_world.astype(np.float32)

        # ✅ NEW: ee_home_pos 缓存策略与 target_pos 完全一致（缺失/shape 不匹配就重算）
        ee_home_position_world = info.get("ee_home_pos", None)
        if ee_home_position_world is None or ee_home_position_world.shape[0] != box_position_world.shape[0]:
            ee_home_position_world = self._compute_ee_home_pos(data)
            info["ee_home_pos"] = ee_home_position_world.astype(np.float32)

        end_effector_position_world = self.get_ee_pos(data)
        current_action_values = info["current_actions"]
        finger_force_values = self.get_finger_forces(data)

        normalized_joint_positions = (
            degrees_of_freedom_positions[:, : self._number_of_robot_degrees_of_freedom]
            * configuration.normalization.joint_pos
        )
        normalized_joint_velocities = (
            degrees_of_freedom_velocities[:, : self._number_of_robot_degrees_of_freedom]
            * configuration.normalization.joint_vel
        )

        observation_vector = np.hstack(
            [
                normalized_joint_positions,
                normalized_joint_velocities,
                box_position_world * configuration.normalization.box_pos,
                target_position_world * configuration.normalization.target_pos,
                end_effector_position_world * configuration.normalization.ee_pos,
                finger_force_values * configuration.normalization.finger_force,
                current_action_values,
            ]
        )
        return observation_vector.astype(np.float32)

    def reset(
        self,
        data: mtx.SceneData,
        done: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        number_of_resets = data.shape[0]

        degrees_of_freedom_positions = np.tile(self._initial_degrees_of_freedom_positions, (number_of_resets, 1))
        degrees_of_freedom_velocities = np.tile(self._initial_degrees_of_freedom_velocities, (number_of_resets, 1))

        if getattr(self.cfg, "randomize_box", True):
            randomized_box_xy_positions = np.random.uniform(
                low=[0.45, -0.15],
                high=[0.65, 0.15],
                size=(number_of_resets, 2),
            )
            degrees_of_freedom_positions[
                :, self._box_degrees_of_freedom_start_index : self._box_degrees_of_freedom_start_index + 2
            ] = randomized_box_xy_positions

        degrees_of_freedom_positions[:, self._box_degrees_of_freedom_start_index + 2] = self._box_initial_height_world_z

        data.reset(self._model)
        data.set_dof_vel(degrees_of_freedom_velocities)
        data.set_dof_pos(degrees_of_freedom_positions, self._model)
        self._model.forward_kinematic(data)

        box_position_world = self.get_box_pos(data)
        target_position_world = self._compute_target_pos(data)

        box_to_target_distance_3d = np.linalg.norm(
            box_position_world - target_position_world,
            axis=1,
        ).astype(np.float32)

        actuator_controls = np.tile(self.default_actuator_controls, (number_of_resets, 1)).astype(np.float32)
        data.actuator_ctrls = actuator_controls

        end_effector_position_world = self.get_ee_pos(data)

        # ✅ NEW: reset 时 ee_home_pos 也按 base + offset 方式设置（与 target_pos 完全一致）
        ee_home_position_world = self._compute_ee_home_pos(data).astype(np.float32)

        # prev_box_height 使用相对高度增量（相对 self._box_initial_height_world_z）
        previous_box_height_relative = (box_position_world[:, 2] - self._box_initial_height_world_z).astype(np.float32)

        # 初始化“爪尖中心到 box 的距离”（保证第一步就有 approach_delta 梯度）
        _, _, fingertip_center_position_world = self.get_fingertip_positions(data)
        previous_fingertip_center_to_box_distance = np.linalg.norm(
            fingertip_center_position_world - box_position_world, axis=1
        ).astype(np.float32)

        previous_end_effector_to_box_distance = np.linalg.norm(
            end_effector_position_world - box_position_world, axis=1
        ).astype(np.float32)

        # --------- reach_grasp_transport 专用：用“语义清晰、不简写”的字段名 ---------
        reach_grasp_transport_previous_attempt_failed = np.zeros((number_of_resets,), dtype=bool)
        reach_grasp_transport_task_success_achieved = np.zeros((number_of_resets,), dtype=bool)
        reach_grasp_transport_lift_success_achieved = np.zeros((number_of_resets,), dtype=bool)

        reach_grasp_transport_release_debounce_counter = np.zeros((number_of_resets,), dtype=np.int32)
        reach_grasp_transport_release_debounce_active = np.zeros((number_of_resets,), dtype=bool)

        reach_grasp_transport_best_grasp_approach_score = np.zeros((number_of_resets,), dtype=np.float32)
        reach_grasp_transport_best_lift_score = np.zeros((number_of_resets,), dtype=np.float32)

        reach_grasp_transport_best_end_effector_to_home_distance = np.zeros((number_of_resets,), dtype=np.float32)
        reach_grasp_transport_idle_steps_counter = np.zeros((number_of_resets,), dtype=np.int32)
        reach_grasp_transport_progress_ledger = np.zeros((number_of_resets,), dtype=np.float32)

        reach_grasp_transport_has_dropped_after_lift = np.zeros((number_of_resets,), dtype=bool)

        info = {
            "current_actions": np.zeros((number_of_resets, self._model.num_actuators), dtype=np.float32),
            "last_actions": np.zeros((number_of_resets, self._model.num_actuators), dtype=np.float32),
            "ctrls": actuator_controls.copy(),
            "target_pos": target_position_world.astype(np.float32),

            # 公共状态（compute_common 依赖，保留原 key）
            "has_cube": np.zeros((number_of_resets,), dtype=bool),
            "ever_had_cube": np.zeros((number_of_resets,), dtype=bool),
            "box_init_pos": box_position_world.astype(np.float32),
            "prev_dist_ee_box": previous_end_effector_to_box_distance,
            "prev_dist_center_box": previous_fingertip_center_to_box_distance,
            "prev_dist_box_target_3d": box_to_target_distance_3d,
            "prev_box_xy": box_position_world[:, :2].astype(np.float32),
            "prev_box_height": previous_box_height_relative,
            "prev_cube_disp_xy": np.zeros((number_of_resets,), dtype=np.float32),
            "steps": np.zeros((number_of_resets,), dtype=np.int32),

            # stage1 counters（保留原 key，避免其它 stage 依赖）
            "hold_counter": np.zeros((number_of_resets,), dtype=np.int32),
            "lift_hold_counter": np.zeros((number_of_resets,), dtype=np.int32),
            "steps_since_contact": np.zeros((number_of_resets,), dtype=np.int32),
            "steps_since_grasp": np.zeros((number_of_resets,), dtype=np.int32),
            "stage1_grasp_step": -np.ones((number_of_resets,), dtype=np.int32),
            "stage1_grasp_achieved": np.zeros((number_of_resets,), dtype=bool),
            "stage1_lift_achieved": np.zeros((number_of_resets,), dtype=bool),

            # stage2 counters（保留）
            "stage2_grasp_step": -np.ones((number_of_resets,), dtype=np.int32),
            "stage2_lost_counter": np.zeros((number_of_resets,), dtype=np.int32),

            # ✅ ee_home_pos 现在按 base + offset 的方式设置（而不是 reset 时 ee 的当前位置）
            "ee_home_pos": ee_home_position_world.astype(np.float32),

            # ---------------- reach_grasp_transport: readable buffers ----------------
            "reach_grasp_transport_previous_attempt_failed": reach_grasp_transport_previous_attempt_failed,
            "reach_grasp_transport_task_success_achieved": reach_grasp_transport_task_success_achieved,
            "reach_grasp_transport_lift_success_achieved": reach_grasp_transport_lift_success_achieved,

            "reach_grasp_transport_release_debounce_counter": reach_grasp_transport_release_debounce_counter,
            "reach_grasp_transport_release_debounce_active": reach_grasp_transport_release_debounce_active,

            # 兼容旧字段（仍然存在但不再推荐使用）
            "reach_grasp_transport_release_box_height_snapshot": box_position_world[:, 2].copy().astype(np.float32),
            "reach_grasp_transport_release_finger_open_ratio_snapshot": np.zeros((number_of_resets,), dtype=np.float32),

            "reach_grasp_transport_best_grasp_approach_score": reach_grasp_transport_best_grasp_approach_score,
            "reach_grasp_transport_best_lift_score": reach_grasp_transport_best_lift_score,

            # lift -> drop -> home
            "reach_grasp_transport_has_dropped_after_lift": reach_grasp_transport_has_dropped_after_lift,

            "reach_grasp_transport_best_end_effector_to_home_distance": reach_grasp_transport_best_end_effector_to_home_distance,
            "reach_grasp_transport_idle_steps_counter": reach_grasp_transport_idle_steps_counter,
            "reach_grasp_transport_progress_ledger": reach_grasp_transport_progress_ledger,
        }

        observation_vector = self._get_obs(data, info)
        return observation_vector, info

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        info = state.info

        number_of_environments = data.shape[0]
        if "has_cube" not in info:
            info["has_cube"] = np.zeros((number_of_environments,), dtype=bool)

        step_counters = info.get("steps", None)
        if (
            step_counters is None
            or not isinstance(step_counters, np.ndarray)
            or step_counters.shape[0] != number_of_environments
        ):
            step_counters = np.zeros((number_of_environments,), dtype=np.int32)

        next_step_counters = (step_counters + 1).astype(np.int32)

        info["steps"] = next_step_counters
        info["steps_for_reward"] = next_step_counters

        reward, terminated, has_cube_next, reward_terms, metrics = compute_reward_and_termination(
            self, data, info
        )

        maximum_seconds = float(getattr(self.cfg, "max_episode_seconds", 10.0))
        control_timestep_seconds = float(getattr(self.cfg, "ctrl_dt", 0.01))
        maximum_steps = max(int(maximum_seconds / max(control_timestep_seconds, 1e-6)), 1)
        timeout = next_step_counters >= maximum_steps

        truncated = timeout.astype(bool)
        metrics["timeout"] = timeout.astype(np.float32)

        info["has_cube"] = has_cube_next
        info["Reward"] = reward_terms
        info["metrics"] = metrics

        observation_vector = self._get_obs(data, info)

        return state.replace(
            obs=observation_vector, reward=reward, terminated=terminated, truncated=truncated
        )

    def get_finger_gap(self, data: np.ndarray):
        degrees_of_freedom_positions = self.get_dof_pos(data)
        finger_joint_positions = degrees_of_freedom_positions[
            :, self._finger_degrees_of_freedom_start_index : self._finger_degrees_of_freedom_end_index
        ]
        finger_gap_value = np.mean(finger_joint_positions, axis=1)
        return finger_gap_value

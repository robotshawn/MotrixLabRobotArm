from motrix_envs import registry
import motrixsim as mtx
import numpy as np

from .pick_place_np import PandaPickPlaceTask
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.manipulation.panda.cfg import PandaPickPlaceEnvCfg

# ✅ 统一调用同一个“夹爪朝向”函数（几何：normalize(tip_center - ee_site_pos)）
from .compute_common import compute_gripper_orientation_world

# ======================================================================
#  脚本抓取版本环境：完全忽略外部 action，用固定脚本抓 cube
#  环境名：panda-pick-place-scripted
# ======================================================================


def _safe_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize last-dim vectors safely. v: (..., 3)"""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


@registry.env("panda-pick-place-scripted", sim_backend="np")
class PandaPickPlaceScriptedTask(PandaPickPlaceTask):
    """
    简单脚本抓取：
      1) home
      2) 移动到 cube 上方
      3) 继续下压（张开夹爪）
      4) 合爪
      5) 抬起

    这里额外：
      - 每步输出 finger 触觉和左右爪尖的世界坐标；
      - 将当前爪尖位置写入 info["fingertip_debug"]；
      - ✅ 输出“夹爪朝向轴(世界系)”与 world_up/down 的对齐（与 reward 计算保持一致）：
        gripper_axis_world = normalize(tip_center - ee_site_pos)
    """

    def __init__(self, cfg: PandaPickPlaceEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)
        assert num_envs == 1, "脚本抓取测试请先用 num_envs=1"

        # ---------------- debug knobs ----------------
        self._debug_print = True
        self._debug_every = 20  # 每隔多少步打印一次；关键相位边界也会强制打印

        # 从 keyframe 中抄来的 3 个关键姿态 ctrl
        # actuator 顺序：joint1..joint7, finger_tendon
        self._home_ctrl = np.array(
            [0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.04],
            dtype=np.float32,
        )
        self._approach_ctrl = np.array(
            [0.2097, 0.423, -0.144392, -2.13105, -0.0291743, 2.52586, -0.492492, 0.04],
            dtype=np.float32,
        )
        self._pickup1_ctrl = np.array(
            [0.2097, 0.458, -0.144392, -2.13105, -0.0291743, 2.52586, -0.492492, 0.04],
            dtype=np.float32,
        )

        # 下压一点：在 approach 的基础上把肘（joint4）再弯一些，让 EE 再往下
        self._press_ctrl = self._approach_ctrl.copy()
        self._press_ctrl[3] -= 0.30
        self._press_ctrl[0] = 0.20

        # 合爪、抬起时手指闭合（最后一个维度）
        self._closed_press_ctrl = self._press_ctrl.copy()
        self._closed_press_ctrl[-1] = 0.0

        self._lift_ctrl = self._pickup1_ctrl.copy()
        self._lift_ctrl[1] = 0.1
        self._lift_ctrl[-1] = 0.0

        self._phase_steps = {
            "home": 150,
            "move_to_pick": 300,
            "press": 150,
            "close": 10,
            "lift": 10,
        }
        self._phase_order = ["home", "move_to_pick", "press", "close", "lift"]
        self._phase_boundaries = np.cumsum([self._phase_steps[p] for p in self._phase_order])
        self._script_total_steps = int(self._phase_boundaries[-1])
        self._script_step = 0

    def reset(
        self,
        data: mtx.SceneData,
        done: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        self._script_step = 0
        obs, info = super().reset(data, done=done)
        ctrl = self._home_ctrl[None, :]
        data.actuator_ctrls = ctrl.astype(np.float32)
        info["ctrls"][:] = ctrl
        info["current_actions"][:] = 0.0
        info["last_actions"][:] = 0.0
        return obs, info

    def _interp_ctrl(
        self,
        t: int,
        t0: int,
        t1: int,
        c0: np.ndarray,
        c1: np.ndarray,
    ) -> np.ndarray:
        if t1 <= t0 + 1:
            return c1
        alpha = (t - t0) / float(max(1, (t1 - t0 - 1)))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return (1.0 - alpha) * c0 + alpha * c1

    def _script_ctrl(self, t: int) -> np.ndarray:
        b = self._phase_boundaries

        if t < b[0]:
            return self._home_ctrl

        if t < b[1]:
            return self._interp_ctrl(t, 0, int(b[1]), self._home_ctrl, self._approach_ctrl)

        if t < b[2]:
            return self._interp_ctrl(t, int(b[1]), int(b[2]), self._approach_ctrl, self._press_ctrl)

        if t < b[3]:
            return self._interp_ctrl(t, int(b[2]), int(b[3]), self._press_ctrl, self._closed_press_ctrl)

        if t < b[4]:
            return self._interp_ctrl(t, int(b[3]), int(b[4]), self._closed_press_ctrl, self._lift_ctrl)

        return self._lift_ctrl

    def _should_print(self, t: int) -> bool:
        if not self._debug_print:
            return False
        if t % max(int(self._debug_every), 1) == 0:
            return True
        if t in set(int(x) for x in self._phase_boundaries.tolist()):
            return True
        return False

    def _debug_orientation(
        self,
        state: NpEnvState,
        t: int,
        left_tip: np.ndarray,
        right_tip: np.ndarray,
        center_tip: np.ndarray,
    ):
        """
        ✅ 统一调用 compute_gripper_orientation_world(...)：
            gripper_axis_world = normalize(tip_center - ee_site_pos)
        """
        ee_pos = super().get_ee_pos(state.data)
        box_pos = super().get_box_pos(state.data)
        ee_pos = np.atleast_2d(ee_pos).astype(np.float32)
        box_pos = np.atleast_2d(box_pos).astype(np.float32)

        center_tip = np.atleast_2d(center_tip).astype(np.float32)

        ori = compute_gripper_orientation_world(
            model=self._model,
            ee_site_name=self.cfg.asset.ee_site_name,
            data=state.data,
            fingertip_center_pos_world=center_tip,
        )
        ee_site_pos = ori["ee_site_pos"]
        gripper_axis_world = ori["gripper_axis_world"]
        cos_down = ori["cos_down"]
        cos_up = ori["cos_up"]
        orient_approach = ori["orient_approach"]
        orient_carry = ori["orient_carry"]

        # 这里的 approach_dir 与 gripper_axis_world 完全一致（保留用于对照打印）
        approach_dir = _safe_normalize(center_tip - ee_site_pos)

        if self._should_print(t):
            print(f"\n[ORIENT DEBUG step={t}]")
            print("  ee_pos(get_ee_pos):", ee_pos)
            print("  ee_site_pos:", ee_site_pos)
            print("  box_pos:", box_pos)
            print("  fingertip_center:", center_tip)
            print("  gripper_axis_world (=EE_site->tip_center):", gripper_axis_world)
            print("  cos(down)=", float(cos_down[0]), " cos(up)=", float(cos_up[0]))
            print("  orient_approach=", float(orient_approach[0]), " orient_carry=", float(orient_carry[0]))
            print("  left_tip:", left_tip)
            print("  right_tip:", right_tip)

        state.info.setdefault("fingertip_debug", {})
        state.info["fingertip_debug"]["ee_pos"] = ee_pos
        state.info["fingertip_debug"]["ee_site_pos"] = ee_site_pos
        state.info["fingertip_debug"]["box_pos"] = box_pos
        state.info["fingertip_debug"]["gripper_axis_world"] = gripper_axis_world
        state.info["fingertip_debug"]["cos_gripper_down"] = cos_down.astype(np.float32)
        state.info["fingertip_debug"]["cos_gripper_up"] = cos_up.astype(np.float32)
        state.info["fingertip_debug"]["orient_approach"] = orient_approach.astype(np.float32)
        state.info["fingertip_debug"]["orient_carry"] = orient_carry.astype(np.float32)
        state.info["fingertip_debug"]["approach_dir"] = approach_dir.astype(np.float32)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        t = self._script_step
        self._script_step = min(self._script_step + 1, self._script_total_steps - 1)

        ctrl = self._script_ctrl(t).astype(np.float32)
        state.data.actuator_ctrls = ctrl[None, :]

        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"][:] = 0.0
        state.info["ctrls"][:] = ctrl

        finger_forces = super().get_finger_forces(state.data)
        finger_forces = np.atleast_2d(finger_forces).astype(np.float32)
        contact_force = np.linalg.norm(finger_forces, axis=1)

        left_tip, right_tip, center_tip = self.get_fingertip_positions(state.data)
        left_tip = np.atleast_2d(left_tip).astype(np.float32)
        right_tip = np.atleast_2d(right_tip).astype(np.float32)
        center_tip = np.atleast_2d(center_tip).astype(np.float32)

        if self._should_print(t):
            print(f"\n[script step {t}]")
            print("  finger_forces:", finger_forces, " contact_force:", contact_force)

        self._debug_orientation(state, t, left_tip, right_tip, center_tip)

        state.info.setdefault("fingertip_debug", {})
        state.info["fingertip_debug"]["left_tip"] = left_tip
        state.info["fingertip_debug"]["right_tip"] = right_tip
        state.info["fingertip_debug"]["center_tip"] = center_tip
        state.info["fingertip_debug"]["finger_forces"] = finger_forces
        state.info["fingertip_debug"]["contact_force"] = contact_force.astype(np.float32)

        return state


# ======================================================================
#  关节调试版本环境：使用 RL 环境的关节增量控制，不再自动执行脚本
#  环境名：panda-pick-place-joint-debug
# ======================================================================


@registry.env("panda-pick-place-joint-debug", sim_backend="np")
class PandaPickPlaceJointDebugTask(PandaPickPlaceTask):
    def __init__(self, cfg: PandaPickPlaceEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)
        assert num_envs == 1, "脚本抓取测试请先用 num_envs=1"

        self._debug_print = True
        self._debug_every = 1

        self._home_ctrl = np.array(
            [0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.04],
            dtype=np.float32,
        )
        self._approach_ctrl = np.array(
            [0.2097, 0.423, -0.144392, -2.13105, -0.0291743, 2.52586, -0.492492, 0.04],
            dtype=np.float32,
        )
        self._pickup1_ctrl = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.1, 0.7, 0.04],
            dtype=np.float32,
        )

    def reset(
        self,
        data: mtx.SceneData,
        done: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        obs, info = super().reset(data, done=done)
        ctrl = self._pickup1_ctrl[None, :]
        data.actuator_ctrls = ctrl.astype(np.float32)
        info["ctrls"][:] = ctrl
        return obs, info

    def _should_print(self, t: int) -> bool:
        if not self._debug_print:
            return False
        return (t % max(int(self._debug_every), 1)) == 0

    def _debug_orientation(
        self,
        state: NpEnvState,
        t: int,
        left_tip: np.ndarray,
        right_tip: np.ndarray,
        center_tip: np.ndarray,
    ):
        ee_pos = super().get_ee_pos(state.data)
        box_pos = super().get_box_pos(state.data)
        ee_pos = np.atleast_2d(ee_pos).astype(np.float32)
        box_pos = np.atleast_2d(box_pos).astype(np.float32)

        center_tip = np.atleast_2d(center_tip).astype(np.float32)

        ori = compute_gripper_orientation_world(
            model=self._model,
            ee_site_name=self.cfg.asset.ee_site_name,
            data=state.data,
            fingertip_center_pos_world=center_tip,
        )
        ee_site_pos = ori["ee_site_pos"]
        gripper_axis_world = ori["gripper_axis_world"]
        cos_down = ori["cos_down"]
        cos_up = ori["cos_up"]
        orient_approach = ori["orient_approach"]
        orient_carry = ori["orient_carry"]

        if self._should_print(t):
            print(f"\n[JOINT DEBUG ORIENT step={t}]")
            print("  ee_pos(get_ee_pos):", ee_pos, " box_pos:", box_pos)
            print("  ee_site_pos:", ee_site_pos)
            print("  center_tip:", center_tip)
            print("  gripper_axis_world (=EE_site->tip_center):", gripper_axis_world)
            print("  cos(down)=", float(cos_down[0]), " cos(up)=", float(cos_up[0]))
            print("  orient_approach=", float(orient_approach[0]), " orient_carry=", float(orient_carry[0]))
            print("  left_tip:", left_tip)
            print("  right_tip:", right_tip)

        state.info.setdefault("fingertip_debug", {})
        state.info["fingertip_debug"]["ee_pos"] = ee_pos
        state.info["fingertip_debug"]["ee_site_pos"] = ee_site_pos
        state.info["fingertip_debug"]["box_pos"] = box_pos
        state.info["fingertip_debug"]["gripper_axis_world"] = gripper_axis_world
        state.info["fingertip_debug"]["cos_gripper_down"] = cos_down.astype(np.float32)
        state.info["fingertip_debug"]["cos_gripper_up"] = cos_up.astype(np.float32)
        state.info["fingertip_debug"]["orient_approach"] = orient_approach.astype(np.float32)
        state.info["fingertip_debug"]["orient_carry"] = orient_carry.astype(np.float32)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        ctrl = self._pickup1_ctrl.astype(np.float32)
        state.data.actuator_ctrls = ctrl[None, :]
        state.info["ctrls"][:] = ctrl

        finger_forces = super().get_finger_forces(state.data)
        finger_forces = np.atleast_2d(finger_forces).astype(np.float32)
        contact_force = np.linalg.norm(finger_forces, axis=1)

        ee_pos = super().get_ee_pos(state.data)
        box_pos = super().get_box_pos(state.data)

        left_tip, right_tip, center_tip = self.get_fingertip_positions(state.data)
        left_tip = np.atleast_2d(left_tip).astype(np.float32)
        right_tip = np.atleast_2d(right_tip).astype(np.float32)
        center_tip = np.atleast_2d(center_tip).astype(np.float32)

        t = int(state.info.get("steps_for_reward", state.info.get("steps", 0)))
        if self._should_print(t):
            print("  box_pos:", box_pos)

        self._debug_orientation(state, t, left_tip, right_tip, center_tip)

        state.info.setdefault("fingertip_debug", {})
        state.info["fingertip_debug"]["left_tip"] = left_tip
        state.info["fingertip_debug"]["right_tip"] = right_tip
        state.info["fingertip_debug"]["center_tip"] = center_tip
        state.info["fingertip_debug"]["finger_forces"] = finger_forces
        state.info["fingertip_debug"]["contact_force"] = contact_force.astype(np.float32)
        print("  finger_forces:", finger_forces)

        return state

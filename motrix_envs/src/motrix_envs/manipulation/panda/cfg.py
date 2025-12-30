# motrix_envs/src/motrix_envs/manipulation/panda/cfg.py
import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg
from .cfg_reward import RewardConfig

model_file = os.path.dirname(__file__) + "/xmls/panda_scene_pickplace.xml"


@dataclass
class ControlConfig:
    # 关节做 delta-angle 控制，手指单独做 [open, close] 插值
    action_scale: float = 0.02
    stiffness: float = 0.0
    damping: float = 0.0
    finger_speed: float = 0.30

    # ✅ NEW：连续夹爪控制的显式参数（pick_place_np.py 里一直用 getattr 兜底，这里补齐默认）
    finger_min_val: float = 0.0
    # 默认 open=0.04 -> 0.25 * open = 0.01
    finger_closed_threshold: float = 0.01
    # 默认 open=0.04 -> 0.95 * open = 0.038
    finger_fully_open_threshold: float = 0.038


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.0]
    default_joint_angles = {
        "actuator1": 0.0,
        "actuator2": 0.3,
        "actuator3": 0.0,
        "actuator4": -1.57,
        "actuator5": 0.0,
        "actuator6": 2.0,
        "actuator7": -0.785,
        "actuator8": 0.04,  # 手指张开
    }


@dataclass
class Normalization:
    joint_pos = 1.0
    joint_vel = 0.1
    box_pos = 1.0
    target_pos = 1.0
    ee_pos = 1.0
    finger_force = 0.1


@dataclass
class Asset:
    base_body_name = "link0"
    ground = "floor"
    box_geom_name = "box"
    ee_site_name = "gripper"
    finger_left_geom_name = "left_finger_pad"
    finger_right_geom_name = "right_finger_pad"
    finger_left_force_sensor = "left_finger_touch"
    finger_right_force_sensor = "right_finger_touch"
    finger_left_touch_site = "left_finger_touch_site"
    finger_right_touch_site = "right_finger_touch_site"


@dataclass
class Commands:
    pass


@registry.envcfg("panda-pick-place")
@dataclass
class PandaPickPlaceEnvCfg(EnvCfg):
    """
    基础 Panda Pick & Place 配置，默认 task_mode 为 reach_only。
    """

    max_episode_seconds: float = 25.0

    model_file: str = model_file
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    commands: Commands = field(default_factory=Commands)

    sim_dt: float = 0.01
    ctrl_dt: float = 0.01

    randomize_box: bool = True

    task_mode: str = "reach_only"
    target_offset_base: tuple[float, float, float] = (0.2, 0.5, 0.03)


@registry.envcfg("panda-pick-place-reach-only")
@dataclass
class PandaPickPlaceReachOnlyEnvCfg(PandaPickPlaceEnvCfg):
    task_mode: str = "reach_only"


@registry.envcfg("panda-pick-place-reach-grasp")
@dataclass
class PandaPickPlaceReachGraspEnvCfg(PandaPickPlaceEnvCfg):
    task_mode: str = "reach_grasp"
    max_episode_seconds: float = 25.0


@registry.envcfg("panda-pick-place-grasp-lift")
@dataclass
class PandaPickPlaceGraspLiftEnvCfg(PandaPickPlaceEnvCfg):
    task_mode: str = "grasp_lift"


@registry.envcfg("panda-pick-place-reach-grasp-transport")
@dataclass
class PandaPickPlaceReachGraspTransportEnvCfg(PandaPickPlaceEnvCfg):
    task_mode: str = "reach_grasp_transport"
    # ✅ NEW：stage3 给足时间，让“抓稳-搬运-放下-回 home”可完成
    max_episode_seconds: float = 25.0


@registry.envcfg("panda-pick-place-scripted")
@dataclass
class PandaPickPlaceScriptedEnvCfg(PandaPickPlaceEnvCfg):
    randomize_box: bool = False


@registry.envcfg("panda-pick-place-joint-debug")
@dataclass
class PandaPickPlaceJointDebugEnvCfg(PandaPickPlaceEnvCfg):
    randomize_box: bool = False

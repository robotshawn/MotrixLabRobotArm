# ===== motrix_rl/src/motrix_rl/cfgs.py =====
# motrix_rl/src/motrix_rl/cfgs.py
# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


class basic:
    @rlcfg("cartpole")
    @dataclass
    class CartPolePPO(PPOCfg):
        max_env_steps: int = 10_000_000
        check_point_interval: int = 500

        policy_hidden_layer_sizes: tuple[int, ...] = (32, 32)
        value_hidden_layer_sizes: tuple[int, ...] = (32, 32)
        rollouts: int = 32
        learning_epochs: int = 5
        mini_batches: int = 4

    @rlcfg("dm-walker", backend="jax")
    @rlcfg("dm-stander", backend="jax")
    @rlcfg("dm-runner", backend="jax")
    @dataclass
    class WalkerPPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 4

    @rlcfg("dm-stander", backend="torch")
    @rlcfg("dm-walker", backend="torch")
    @dataclass
    class WalkerPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 4
        mini_batches: int = 32

    @rlcfg("dm-runner", backend="torch")
    @dataclass
    class RunnerPPOTorch(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 2048

        learning_rate: float = 2e-4
        rollouts: int = 24
        learning_epochs: int = 2
        mini_batches: int = 32


class locomotion:
    @rlcfg("go1-flat-terrain-walk")
    @dataclass
    class Go1WalkPPO(PPOCfg):
        seed: int = 42
        share_policy_value_features: bool = False
        max_env_steps: int = 1024 * 60000
        num_envs: int = 2048


class manipulation:
    # ---------------------- JAX 版本（基础配置） ---------------------- #
    @rlcfg("panda-pick-place", backend="jax")
    @dataclass
    class PandaPickPlacePPO(PPOCfg):
        seed: int = 42
        max_env_steps: int = 1024 * 40000
        num_envs: int = 1024
        rollouts: int = 32
        learning_epochs: int = 4
        mini_batches: int = 8
        policy_hidden_layer_sizes: tuple[int, ...] = (512, 512)
        value_hidden_layer_sizes: tuple[int, ...] = (512, 512)
        entropy_loss_scale: float = 0.02

    # ---------------------- JAX 版本各个 curriculum stage ---------------------- #
    @rlcfg("panda-pick-place-reach-only", backend="jax")
    @dataclass
    class PandaPickPlaceReachOnlyPPO(PandaPickPlacePPO):
        max_env_steps: int = 1024 * 20000
        learning_rate: float = 2e-4

    @rlcfg("panda-pick-place-reach-grasp", backend="jax")
    @dataclass
    class PandaPickPlaceReachGraspPPO(PandaPickPlacePPO):
        max_env_steps: int = 1024 * 160000
        learning_rate: float = 2e-4
        # ✅ NEW：略增熵，减缓过早变得确定性（对应你图里的 std/entropy 快速下滑）
        entropy_loss_scale: float = 0.03

    @rlcfg("panda-pick-place-grasp-lift", backend="jax")
    @dataclass
    class PandaPickPlaceGraspLiftPPO(PandaPickPlacePPO):
        max_env_steps: int = 1024 * 160000
        learning_rate: float = 5e-5

    @rlcfg("panda-pick-place-reach-grasp-transport", backend="jax")
    @dataclass
    class PandaPickPlaceReachGraspTransportPPO(PandaPickPlacePPO):
        # ✅ NEW：stage3 难度高且要学放下+回 home，训练步数拉到与 stage2 对齐
        max_env_steps: int = 1024 * 300000
        learning_rate: float = 2e-4
        # ✅ NEW：略降熵，减少“抡臂甩飞/高抛”的探索幅度
        entropy_loss_scale: float = 0.03

    # ---------------------- Torch 版本基础配置 ---------------------- #
    @rlcfg("panda-pick-place", backend="torch")
    @dataclass
    class PandaPickPlaceTorchPPO(PPOCfg):
        max_env_steps: int = 1024 * 40000
        num_envs: int = 1024
        seed: int = 42
        rollouts: int = 32
        learning_epochs: int = 4
        mini_batches: int = 8
        policy_hidden_layer_sizes: tuple[int, ...] = (256, 256)
        value_hidden_layer_sizes: tuple[int, ...] = (256, 256)
        entropy_loss_scale: float = 0.02
        share_policy_value_features: bool = False

    # ---------------------- Torch 版本各个 curriculum stage ---------------------- #
    @rlcfg("panda-pick-place-reach-only", backend="torch")
    @dataclass
    class PandaPickPlaceReachOnlyPPO_Torch(PandaPickPlaceTorchPPO):
        max_env_steps: int = 1024 * 20000
        learning_rate: float = 2e-4

    @rlcfg("panda-pick-place-reach-grasp", backend="torch")
    @dataclass
    class PandaPickPlaceReachGraspPPO_Torch(PandaPickPlaceTorchPPO):
        max_env_steps: int = 1024 * 160000
        learning_rate: float = 2e-4
        # ✅ NEW：略增熵，减缓过早变得确定性（对应你图里的 std/entropy 快速下滑）
        entropy_loss_scale: float = 0.03

    @rlcfg("panda-pick-place-grasp-lift", backend="torch")
    @dataclass
    class PandaPickPlaceGraspLiftPPO_Torch(PandaPickPlaceTorchPPO):
        max_env_steps: int = 1024 * 160000
        learning_rate: float = 5e-5

    @rlcfg("panda-pick-place-reach-grasp-transport", backend="torch")
    @dataclass
    class PandaPickPlaceReachGraspTransportPPO_Torch(PandaPickPlaceTorchPPO):
        max_env_steps: int = 1024 * 300000
        learning_rate: float = 2e-4
        entropy_loss_scale: float = 0.03

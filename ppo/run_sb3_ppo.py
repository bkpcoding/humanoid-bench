import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from fast_td3.environments.humanoid_bench_env import HumanoidBenchEnv

# Set up headless rendering before importing anything MuJoCo-related
if sys.platform != "darwin":  # Not macOS
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"

# Also set these for additional compatibility
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

import gymnasium as gym
import torch
import torch.nn as nn

from gymnasium.wrappers import TimeLimit
import wandb
from wandb.integration.sb3 import WandbCallback

import humanoid_bench
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--max_steps", default=20000000, type=int)
parser.add_argument("--wandb_entity", default="robot-learning", type=str)
parser.add_argument("--enable_profiling", action="store_true", help="Enable PyTorch profiling")
parser.add_argument("--profile_steps", default=100, type=int, help="Number of steps to profile")
parser.add_argument("--profile_warmup", default=10, type=int, help="Warmup steps for profiling")
parser.add_argument("--profile_active", default=20, type=int, help="Active profiling steps")
ARGS = parser.parse_args()


def calculate_network_norms(network: nn.Module, prefix: str = ""):
    """
    Calculate various norms of network parameters for logging.
    
    Args:
        network: PyTorch network module
        prefix: String prefix for metric names
        
    Returns:
        Dictionary of norm metrics
    """
    metrics = {}
    
    # Calculate total parameter norm
    total_norm = 0.0
    param_count = 0
    
    # Calculate layer-wise norms
    layer_norms = {}
    
    for name, param in network.named_parameters():
        if param.requires_grad:
            param_norm = param.data.norm(2).item()
            layer_norms[f"{prefix}_{name}_norm"] = param_norm
            total_norm += param_norm ** 2
            param_count += param.numel()
    
    # Total parameter norm
    total_norm = total_norm ** 0.5
    metrics[f"{prefix}_total_param_norm"] = total_norm
    metrics[f"{prefix}_param_count"] = param_count
    
    # Add layer-wise norms
    metrics.update(layer_norms)
    
    return metrics


def make_env(
    rank,
    seed=0
):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    def _init():
        
        env = gym.make(ARGS.env_name, render_mode=None)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        
        env.action_space.seed(ARGS.seed + rank)
        
        return env

    return _init

class EvalCallback(BaseCallback):
    
    def __init__(self, eval_every: int = 500000, verbose: int = 0):
        super(EvalCallback, self).__init__(verbose=verbose)
        # Create separate eval env with render_mode for video recording
        def make_eval_env():
            env = gym.make(ARGS.env_name, render_mode="rgb_array")
            env = TimeLimit(env, max_episode_steps=1000)
            env = Monitor(env)
            return env

        self.eval_every = eval_every
        self.eval_env = DummyVecEnv([make_eval_env])

    def _on_step(self) -> bool:
        
        if self.num_timesteps % self.eval_every == 0:
            self.record_video()

        return True
    
    def record_video(self) -> None:

        print("recording video")
        video = []

        obs = self.eval_env.reset()
        for i in range(1000):
            action = self.model.predict(obs, deterministic=True)[0]
            obs, _, _, _ = self.eval_env.step(action)
            pixels = self.eval_env.render().transpose(2,0,1)
            video.append(pixels)

        video = np.stack(video)
        wandb.log({"render_video": wandb.Video(video, fps=100, format="gif")})


class LogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.aux_rewards = {}
        self.aux_returns = {}
        for key in info_keywords:
            self.aux_rewards[key] = np.zeros(ARGS.num_envs)
            self.aux_returns[key] = deque(maxlen=100)


    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx in range(len(infos)):
            for key in self.aux_rewards.keys():
                self.aux_rewards[key][idx] += infos[idx][key]

            if self.locals['dones'][idx]:
                for key in self.aux_rewards.keys():
                    self.aux_returns[key].append(self.aux_rewards[key][idx])
                    self.aux_rewards[key][idx] = 0
        return True

    def _on_rollout_end(self) -> None:
        
        for key in self.aux_returns.keys():
            self.logger.record("aux_returns_{}/mean".format(key), np.mean(self.aux_returns[key]))


class EpisodeLogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.returns_info = {
            "eval_avg_return": [],
            "eval_avg_length": [],
            "results/success": [],
            "results/success_subtasks": [],
        }


    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx in range(len(infos)):
            curr_info = infos[idx]
            if "episode" in curr_info:
                self.returns_info["eval_avg_return"].append(curr_info["episode"]["r"])
                self.returns_info["eval_avg_length"].append(curr_info["episode"]["l"])
                cur_info_success = 0
                if "success" in curr_info:
                    cur_info_success = curr_info["success"]
                self.returns_info["results/success"].append(cur_info_success)
                cur_info_success_subtasks = 0
                if "success_subtasks" in curr_info:
                    cur_info_success_subtasks = curr_info["success_subtasks"]
                self.returns_info["results/success_subtasks"].append(cur_info_success_subtasks)
        return True

    def _on_rollout_end(self) -> None:
        
        for key in self.returns_info.keys():
            if self.returns_info[key]:
                self.logger.record(key, np.mean(self.returns_info[key]))
                self.returns_info[key] = []


class NetworkNormCallback(BaseCallback):
    """
    Custom callback for logging network parameter norms to match FastTD3 logging.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # This method is required by BaseCallback but we don't need to do anything here
        return True

    def _on_rollout_end(self) -> None:
        # Calculate and log network parameter norms
        actor_norms = calculate_network_norms(self.model.policy.mlp_extractor, "actor")
        critic_norms = calculate_network_norms(self.model.policy.value_net, "critic")
        
        # Log the norms
        for key, value in actor_norms.items():
            self.logger.record(key, value)
        for key, value in critic_norms.items():
            self.logger.record(key, value)


class ProfilerCallback(BaseCallback):
    """
    Custom callback for PyTorch profiling to understand GPU utilization.
    """

    def __init__(self, 
                 enable_profiling=True,
                 profile_steps=100,
                 warmup_steps=10,
                 active_steps=20,
                 verbose=0):
        super().__init__(verbose)
        self.enable_profiling = enable_profiling
        self.profile_steps = profile_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.profiler = None
        self.step_count = 0
        
        if self.enable_profiling:
            print(f"Profiling enabled: will profile for {profile_steps} steps")
            print(f"Warmup: {warmup_steps}, Active: {active_steps}")

    def _on_training_start(self) -> None:
        if self.enable_profiling:
            # Create profiler activities based on available hardware
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            
            # Create the profiler
            self.profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=self.warmup_steps,
                    active=self.active_steps,
                    repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profile'),
                record_shapes=True,
                with_stack=True,
                with_flops=True
            )
            self.profiler.start()
            print("PyTorch profiler started")

    def _on_step(self) -> bool:
        if self.enable_profiling and self.profiler is not None:
            self.step_count += 1
            self.profiler.step()
            
            # Stop profiling after specified number of steps
            if self.step_count >= self.profile_steps:
                self.profiler.stop()
                print(f"Profiling completed after {self.step_count} steps")
                print("Profiling results saved to ./log/profile")
                print("View with: tensorboard --logdir=./log/profile")
                self.profiler = None
                self.enable_profiling = False
        
        return True

    def _on_training_end(self) -> None:
        if self.profiler is not None:
            self.profiler.stop()
            print("Profiling stopped at training end")


def main(argv):
    env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])
    # Replace the SubprocVecEnv creation with FastTD3's GPU wrapper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create GPU-accelerated environment
    env = HumanoidBenchEnv(
        env_name=ARGS.env_name,
        num_envs=ARGS.num_envs,
        render_mode=None,
        device=device
    )
    
    steps = 1000
        
    run = wandb.init(
        entity=ARGS.wandb_entity,
        project="rl_scratch",
        name=f"ppo_{ARGS.env_name}",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    
    # Create callbacks list
    callbacks = [
        WandbCallback(model_save_path=f"models/{run.id}", verbose=2),
        EvalCallback(),
        LogCallback(info_keywords=[]),
        EpisodeLogCallback(),
        NetworkNormCallback()
    ]
    
    # Add profiler callback if enabled
    if ARGS.enable_profiling:
        callbacks.append(ProfilerCallback(
            enable_profiling=True,
            profile_steps=ARGS.profile_steps,
            warmup_steps=ARGS.profile_warmup,
            active_steps=ARGS.profile_active
        ))
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", learning_rate=float(ARGS.learning_rate), batch_size=32768)
    model.learn(total_timesteps=ARGS.max_steps, log_interval=1, callback=callbacks)
    
    
    model.save("ppo")
    print("Training finished")

if __name__ == '__main__':
#   app.run(main)
    main(None)
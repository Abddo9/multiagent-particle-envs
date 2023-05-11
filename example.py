
import time

import numpy as np
import torch
import argparse
import sys

import os
from env_wrappers import SubprocVecEnv, DummyVecEnv
from mpe.MPE_env import MPEEnv

os.environ["SUPPRESS_MA_PROMPT"] = "1"



#0 nothing   --- 	0,0
#1 left      	-1,0
#2 right   	1,0
#3 down	0,-1
#4 up		0, 1

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def rand():
    return int(np.random.rand()*5)

def use_mpe_env(all_args, render: bool = False, save_render: bool = False):

 
    n_steps = 20

    envs = make_train_env(all_args)

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    for s in range(n_steps):
        actions = []
        step += 1
        print(f"Step {step}")
        
        obs = envs.reset()

        for i in range(len(obs)):
            env_ac = []
            for j in range(all_args.num_agents):
                env_ac.append(rand())
            actions.append(env_ac)
        
        actions = np.array(actions).reshape(len(obs), all_args.num_agents, -1)
        actions = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
        obs, rews, dones, info = envs.step(actions)
                        
        if render:       
            img = envs.render(mode="rgb_array")      
            frame_list.append(envs.render(mode="rgb_array")[0][0])  # Can give the camera an agent index to focus on

    if render and save_render:
        import cv2

        video_name = all_args.scenario_name + ".mp4"

        # Produce a video
        video = cv2.VideoWriter(
            video_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            1 / envs.envs[0].world.dt,  # FPS
            (frame_list[0].shape[1], frame_list[0].shape[0]),
        )
        for img in frame_list:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img)
        video.release()

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps}"
    )


def main(args):
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=3, help="number of players")
    parser.add_argument('--episode_length', type=int, default=25, help="Max length for any episode")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--env_name", type=str, default='MPE', help="specify the name of environment")
    parser.add_argument("--n_rollout_threads", type=int, default=1, help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1, help="Number of parallel envs for evaluating rollouts")

    all_args = parser.parse_known_args(args)[0]

    use_mpe_env(all_args, render=True, save_render=True)

if __name__ == "__main__":
    main(sys.argv[1:])
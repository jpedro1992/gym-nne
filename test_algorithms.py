import logging

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from tqdm import tqdm
from envs.dqn_deepset import DQN_DeepSets
from envs.ppo_deepset import PPO_DeepSets
from envs.nne_scheduling_env import NNESchedulingEnv
from sb3_contrib import MaskablePPO

SEED = 2

# Logging
logging.basicConfig(filename='run_test.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

env_kwargs = {"n_nodes": 4, "arrival_rate_r": 100, "call_duration_r": 1, "episode_length": 100}
MONITOR_PATH = f"./results/test/ppo_deepset_{SEED}_n{env_kwargs['n_nodes']}_lam{env_kwargs['arrival_rate_r']}_mu{env_kwargs['call_duration_r']}.monitor.csv"

if __name__ == "__main__":
    # Define here variables for testing
    num_nodes = 4  # 4, 8, 12, 16, 32
    reward_function = 'multi'
    alg = 'dqn_deepsets'
    strategy = "cost/"
    path = "data/train/v2-nov-dec/nodes/"

    cost_weight = 1.0
    bandwidth_weight = 0.0
    latency_weight = 0.0
    gini_weight = 0.0

    episodes = 100
    episode_length = 100
    call_duration_r = 1

    env = NNESchedulingEnv(num_nodes=num_nodes, arrival_rate_r=100, call_duration_r=1,
                           episode_length=100,
                           reward_function=reward_function,
                           latency_weight=latency_weight,
                           cost_weight=cost_weight,
                           gini_weight=gini_weight,
                           path_csv_files=path,
                           bandwidth_weight=bandwidth_weight)
    env.reset()
    _, _, _, info = env.step(0)
    info_keywords = tuple(info.keys())

    envs = DummyVecEnv([lambda: NNESchedulingEnv(num_nodes=num_nodes, arrival_rate_r=100, call_duration_r=1,
                                                 episode_length=100,
                                                 reward_function=reward_function,
                                                 latency_weight=latency_weight,
                                                 cost_weight=cost_weight,
                                                 gini_weight=gini_weight,
                                                 path_csv_files=path,
                                                 bandwidth_weight=bandwidth_weight)])
    
    envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)

    # Algos supported
    agent = None
    if alg == "ppo_deepsets":
        agent = PPO_DeepSets(envs, seed=SEED, tensorboard_log=None)
    elif alg == 'dqn_deepsets':
        agent = DQN_DeepSets(envs, seed=SEED, tensorboard_log=None)
    elif alg == 'mask_ppo':
        agent = MaskablePPO("MlpPolicy", envs, gamma=0.95, verbose=1, tensorboard_log=None)
    else:
        print('Invalid algorithm!')

    # Adapt the path accordingly
    agent.load(f"./results/nne/multi/"
               + strategy + "/" + alg + "_env_nne_num_nodes_4_reward_multi_totalSteps_200000_run_1/"
               + alg + "_env_nne_num_nodes_4_reward_multi_totalSteps_200000")

    # Test the agent for 100 episodes
    for _ in tqdm(range(episodes)):
        obs = envs.reset()
        action_mask = np.array(envs.env_method("action_masks"))
        done = False
        while not done:
            action = agent.predict(obs, action_mask)
            obs, reward, dones, info = envs.step(action)
            action_mask = np.array(envs.env_method("action_masks"))
            done = dones[0]

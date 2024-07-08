import logging

import numpy as np
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from envs.nne_scheduling_env import NNESchedulingEnv
from envs.baselines import latency_greedy_policy, cost_greedy_policy, bandwidth_greedy_policy

MONITOR_PATH = "./results/greedy_monitor.csv"

# Logging
logging.basicConfig(filename='run_baselines.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

NUM_NODES = 4

# Defaults for Weights -> change here to run
LATENCY_WEIGHT = 0.0
CPU_WEIGHT = 0.0
GINI_WEIGHT = 0.0
BANDWIDTH_WEIGHT = 1.0

LATENCY_GREEDY = 'lat'
COST_GREEDY = 'cost'
BANDWIDTH_GREEDY = 'band'

if __name__ == "__main__":
    policy = LATENCY_GREEDY
    num_nodes = [4]  # 12, 16, 24, 32, 48, 64, 80, 128, 150, 180]
    n_episodes = 100
    path = "data/train/v2-jan-feb/nodes/"

    factors = [1, 2, 4, 6, 8, 10, 12]
    TEST_FACTORS = True

    i = 0
    for n in num_nodes:
        print("Initiating run for {} with: nodes: {} | ".format(policy, n))
        if TEST_FACTORS:
            for f in factors:
                env = NNESchedulingEnv(num_nodes=n,
                                        arrival_rate_r=100, call_duration_r=1,
                                        episode_length=100,
                                        reward_function='multi',
                                        factor=f,
                                        path_csv_files=path,
                                        file_results_name=str(i) + "_" + policy + '_baselines_num_nodes_' + str(n) + '_factor_' + str(f))
                env.reset()
                _, _, _, info = env.step(0)
                info_keywords = tuple(info.keys())
                env = NNESchedulingEnv(num_nodes=n,
                                       arrival_rate_r=100, call_duration_r=1,
                                       episode_length=100,
                                       reward_function='multi',
                                       factor=f,
                                       path_csv_files=path,
                                       file_results_name=str(i) + "_" + policy + '_baselines_num_nodes_' + str(n) + '_factor_' + str(f))

                # env = Monitor(env, filename=MONITOR_PATH, info_keywords=info_keywords)
                returns = []
                for _ in tqdm(range(n_episodes)):
                    obs = env.reset()
                    action_mask = env.action_masks()
                    return_ = 0.0
                    done = False
                    while not done:
                        if policy == LATENCY_GREEDY:
                            action = latency_greedy_policy(env, action_mask)
                        elif policy == COST_GREEDY:
                            action = cost_greedy_policy(env, action_mask)
                        elif policy == BANDWIDTH_GREEDY:
                            action = bandwidth_greedy_policy(env, action_mask)
                        else:
                            print("unrecognized policy!")

                        obs, reward, done, info = env.step(action)
                        action_mask = env.action_masks()
                        return_ += reward
                    returns.append(return_)

                i += 1

        else:
            env = NNESchedulingEnv(num_nodes=n,
                                   arrival_rate_r=100, call_duration_r=1,
                                   episode_length=100,
                                   reward_function='multi',
                                   path_csv_files=path,
                                   file_results_name=str(i) + "_" + policy + '_baselines_num_nodes_' + str(n))
            env.reset()
            _, _, _, info = env.step(0)
            info_keywords = tuple(info.keys())
            env = NNESchedulingEnv(num_nodes=n,
                                   arrival_rate_r=100, call_duration_r=1,
                                   episode_length=100,
                                   reward_function='multi',
                                   path_csv_files=path,
                                   file_results_name=str(i) + "_" + policy + '_baselines_num_nodes_' + str(n))

            # env = Monitor(env, filename=MONITOR_PATH, info_keywords=info_keywords)
            returns = []
            for _ in tqdm(range(n_episodes)):
                obs = env.reset()
                action_mask = env.action_masks()
                return_ = 0.0
                done = False
                while not done:
                    if policy == LATENCY_GREEDY:
                        action = latency_greedy_policy(env, action_mask)
                    elif policy == COST_GREEDY:
                        action = cost_greedy_policy(env, action_mask)
                    elif policy == BANDWIDTH_GREEDY:
                        action = bandwidth_greedy_policy(env, action_mask)
                    else:
                        print("unrecognized policy!")

                    obs, reward, done, info = env.step(action)
                    action_mask = env.action_masks()
                    return_ += reward
                returns.append(return_)

            i += 1

import logging
import argparse

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO, MaskablePPO, TRPO, TQC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from envs.nne_scheduling_env import NNESchedulingEnv
from envs.ppo_deepset import PPO_DeepSets
from envs.dqn_deepset import DQN_DeepSets
from sb3_contrib.common.maskable.utils import get_action_masks

matplotlib.use('TkAgg')

# Logging
logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run RL Agent!')
parser.add_argument('--alg', default='mask_ppo',
                    help='The algorithm: ["ppo_deepsets", "recurrent_ppo", "ppo", "mask_ppo", "ppo_deepsets", "dqn_deepsets", "trpo", "tqc"]')
parser.add_argument('--env_name', default='nne', help='Env: ["nne"]')
parser.add_argument('--num_nodes', default=4, help='num_nodes: 4, 6, 8, etc')
parser.add_argument('--reward', default='multi', help='reward: ["naive", "risk", "cost", "latency"]')
parser.add_argument('--training', default=False, action="store_true", help='Training mode')
parser.add_argument('--testing', default=True, action="store_true", help='Testing mode')
parser.add_argument('--loading', default=False, action="store_true", help='Loading mode')
parser.add_argument('--load_path',
                    default='results/v2/nne/multi/latineq/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000_run_1/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000.zip',
                    help='Loading path, ex: logs/model/test.zip')
parser.add_argument('--test_path',
                    default='results/v2/nne/multi/latineq/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000_run_1/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000.zip',
                    help='Testing path, ex: logs/model/test.zip')
parser.add_argument('--steps', default=200000, help='Save model after X steps')
parser.add_argument('--total_steps', default=200000, help='The total number of steps.')

# TODO: add other arguments if needed
# parser.add_argument('--k8s', default=False, action="store_true", help='K8s mode')
# parser.add_argument('--goal', default='cost', help='Reward Goal: ["cost", "latency"]')

args = parser.parse_args()

TESTING_FACTORS = True


def get_model(alg, env, tensorboard_log):
    model = 0
    if alg == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)  # , n_steps=steps
    elif alg == 'mask_ppo':
        model = MaskablePPO("MlpPolicy", env, gamma=0.95, verbose=1, tensorboard_log=tensorboard_log)  # , n_steps=steps
    elif alg == 'ppo_deepsets':
        model = PPO_DeepSets(env, num_steps=100, n_minibatches=8, ent_coef=0.001, tensorboard_log=None, seed=2)
    elif alg == 'trpo':
        model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=None)
    elif alg == 'tqc':
        policy_kwargs = dict(n_critics=2, n_quantiles=25)
        model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs)
    elif alg == 'dqn_deepsets':
        model = DQN_DeepSets(env, num_steps=100, n_minibatches=8, tensorboard_log=None)
    else:
        logging.info('Invalid algorithm!')

    return model


def get_load_model(env, alg, tensorboard_log, load_path):
    if alg == 'ppo':
        return PPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        return RecurrentPPO.load(load_path, reset_num_timesteps=False, verbose=1,
                                 tensorboard_log=tensorboard_log)  # n_steps=steps
    elif alg == 'a2c':
        return A2C.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'mask_ppo':
        return MaskablePPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'ppo_deepsets':
        agent = PPO_DeepSets(env, tensorboard_log=None)
        return agent.load(f"" + load_path)
    elif alg == 'dqn_deepsets':
        agent = DQN_DeepSets(env, tensorboard_log=None)
        return agent.load(f"" + load_path)
    elif alg == 'trpo':
        agent = TRPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
        return agent.load(f"" + load_path)
    elif alg == 'tqc':
        agent = TQC.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
        return agent.load(f"" + load_path)
    else:
        logging.info('Invalid algorithm!')


def get_env(env_name, num_nodes, reward_function):
    envs = 0
    latency_weight = 0.0
    cost_weight = 0.0
    gini_weight = 0.0
    bandwidth_weight = 1.0
    factor = 1
    path = "data/train/v1/nodes/"


    if env_name == "nne":
        env = NNESchedulingEnv(num_nodes=num_nodes, arrival_rate_r=100, call_duration_r=1,
                               episode_length=100,
                               reward_function=reward_function,
                               latency_weight=latency_weight,
                               cost_weight=cost_weight,
                               gini_weight=gini_weight,
                               factor=factor,
                               path_csv_files=path,
                               bandwidth_weight=bandwidth_weight)
        # For faster training!
        # otherwise just comment the following lines

        env.reset()
        _, _, _, info = env.step(0)
        info_keywords = tuple(info.keys())

        env = SubprocVecEnv([lambda: NNESchedulingEnv(num_nodes=num_nodes, arrival_rate_r=100,
                                                      call_duration_r=1, episode_length=100,
                                                      reward_function=reward_function,
                                                      latency_weight=latency_weight,
                                                      cost_weight=cost_weight,
                                                      gini_weight=gini_weight,
                                                      factor=factor,
                                                      path_csv_files=path,
                                                      bandwidth_weight=bandwidth_weight)
                             for i in range(1)])
        envs = VecMonitor(env, filename="vec_nne_gym_results", info_keywords=info_keywords)


    else:
        logging.info('Invalid environment!')

    return envs


def test_model(model, env, n_episodes, n_steps, smoothing_window, fig_name):
    episode_rewards = []
    reward_sum = 0
    obs = env.reset()

    print("------------Testing -----------------")
    for e in range(n_episodes):
        for step in range(n_steps):
            action_masks = get_action_masks(env)
            # print(f"Step: {step}, Action Masks: {action_masks}")

            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, info = env.step(action)
            reward_sum += float(reward)

            # print(f"Action: {action}, Reward: {reward}, Done: {done}")

            if done:
                episode_rewards.append(reward_sum)
                print(f"Episode {e} | Total reward: {reward_sum}")
                reward_sum = 0
                obs = env.reset()
                break

    env.close()

    # Free memory
    del model, env

    # Plot the episode reward over time
    '''
    plt.figure()
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(fig_name, dpi=250, bbox_inches='tight')
    '''

def main():
    # Import and initialize Environment
    logging.info(args)

    alg = args.alg
    env_name = args.env_name
    reward = args.reward
    num_nodes = int(args.num_nodes)
    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path

    steps = int(args.steps)
    total_steps = int(args.total_steps)

    env = get_env(env_name, num_nodes, reward)
    print("env: {}".format(env))

    tensorboard_log = "results/" + env_name + "/" + reward + "/"

    name = alg + "_env_" + env_name + "_num_nodes_" + str(num_nodes) \
           + "_reward_" + reward + "_totalSteps_" + str(total_steps)

    # callback: does not work with multiple envs
    checkpoint_callback = CheckpointCallback(save_freq=steps, save_path="logs/" + name, name_prefix=name)

    # Training selected
    if training:
        if loading:  # resume training
            model = get_load_model(alg, tensorboard_log, load_path)
            model.set_env(env)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)
        else:
            if alg == "ppo_deepsets" or alg == 'dqn_deepsets':
                model = get_model(alg, env, tensorboard_log)
                print("model: {}".format(model))
                model.learn(total_timesteps=total_steps)
            else:
                model = get_model(alg, env, tensorboard_log)
                model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)

        model.save(name)

    # Testing selected
    if testing:
        if TESTING_FACTORS:
            path = "data/train/v1/nodes/"
            factors = [1, 2, 4, 6, 8, 10, 12]
            i = 0
            for f in factors:
                env = NNESchedulingEnv(num_nodes=num_nodes,
                                       arrival_rate_r=100, call_duration_r=1,
                                       episode_length=100,
                                       reward_function='multi',
                                       factor=f,
                                       path_csv_files=path,
                                       file_results_name=str(i) + '_nne_gym_num_nodes_' + str(
                                           num_nodes) + '_factor_' + str(f))
                env.reset()
                _, _, _, info = env.step(0)
                info_keywords = tuple(info.keys())
                env = NNESchedulingEnv(num_nodes=num_nodes,
                                       arrival_rate_r=100, call_duration_r=1,
                                       episode_length=100,
                                       reward_function='multi',
                                       factor=f,
                                       path_csv_files=path,
                                       file_results_name=str(i) + '_nne_gym_num_nodes_' + str(
                                           num_nodes) + '_factor_' + str(f))

                model = get_load_model(env, alg, tensorboard_log, test_path)
                test_model(model, env, n_episodes=100, n_steps=100, smoothing_window=5, fig_name=name + "_test_reward.png")
                i += 1
        else:
            model = get_load_model(env, alg, tensorboard_log, test_path)
            test_model(model, env, n_episodes=100, n_steps=100, smoothing_window=5, fig_name=name + "_test_reward.png")



if __name__ == "__main__":
    main()

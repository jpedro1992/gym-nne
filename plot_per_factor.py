import glob
import logging
import os
from collections import namedtuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


def get_statistics(df, alg_name,
                   avg_reward, ci_avg_reward,
                   avg_ep_block_prob, ci_avg_ep_block_prob,
                   avg_deployment_cost, ci_avg_deployment_cost,
                   avg_total_latency, ci_avg_total_latency,
                   avg_dl, ci_avg_dl,
                   avg_ul, ci_avg_ul,
                   avg_jitter, ci_avg_jitter,
                   avg_gini, ci_avg_gini,
                   avg_executionTime, ci_avg_executionTime):

    avg_reward.append(np.mean(df["reward"]))
    ci_avg_reward.append(1.96 * np.std(df["reward"]) / np.sqrt(len(df["reward"])))
    avg_ep_block_prob.append(np.mean(df["ep_block_prob"]))
    ci_avg_ep_block_prob.append(1.96 * np.std(df["ep_block_prob"]) / np.sqrt(len(df["ep_block_prob"])))
    avg_deployment_cost.append(np.mean(df["avg_deployment_cost"]))
    ci_avg_deployment_cost.append(1.96 * np.std(df["avg_deployment_cost"]) / np.sqrt(len(df["avg_deployment_cost"])))
    avg_total_latency.append(np.mean(df["avg_total_latency"]))
    ci_avg_total_latency.append(1.96 * np.std(df["avg_total_latency"]) / np.sqrt(len(df["avg_total_latency"])))
    avg_dl.append(np.mean(df["avg_dl"]))
    ci_avg_dl.append(1.96 * np.std(df["avg_dl"]) / np.sqrt(len(df["avg_dl"])))
    avg_ul.append(np.mean(df["avg_ul"]))
    ci_avg_ul.append(1.96 * np.std(df["avg_ul"]) / np.sqrt(len(df["avg_ul"])))
    avg_jitter.append(np.mean(df["avg_jitter"]))
    ci_avg_jitter.append(1.96 * np.std(df["avg_jitter"]) / np.sqrt(len(df["avg_jitter"])))
    avg_gini.append(np.mean(df["gini"]))
    ci_avg_gini.append(1.96 * np.std(df["gini"]) / np.sqrt(len(df["gini"])))
    avg_executionTime.append(np.mean(df["executionTime"]))
    ci_avg_executionTime.append(1.96 * np.std(df["executionTime"]) / np.sqrt(len(df["executionTime"])))


if __name__ == "__main__":
    reward = 'latency'  # cost, risk or latency
    max_reward = 100  # cost= 1500, risk and latency= 100
    ylim = 120  # 1700 for cost and 120 for rest

    version = 'v1' # or v1
    path = "results/"
    # testing
    path_ppo_cost = path + version + "/nne/multi/cost/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000_run_1/testing/scalability/"
    path_ppo_latency = path + version + "/nne/multi/latency/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000_run_1/testing/scalability/"
    path_ppo_inequality = path + version + "/nne/multi/inequality/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000_run_1/testing/scalability/"
    path_ppo_bandwidth = path + version + "/nne/multi/bandwidth/mask_ppo_env_nne_num_nodes_4_reward_multi_totalSteps_200000_run_1/testing/scalability/"

    avg_reward_ppo_cost = []
    ci_avg_reward_ppo_cost = []
    avg_ep_block_prob_ppo_cost = []
    ci_avg_ep_block_prob_ppo_cost = []
    avg_deployment_cost_ppo_cost = []
    ci_avg_deployment_cost_ppo_cost = []
    avg_total_latency_ppo_cost = []
    ci_avg_total_latency_ppo_cost = []
    avg_dl_ppo_cost = []
    ci_avg_dl_ppo_cost = []
    avg_ul_ppo_cost = []
    ci_avg_ul_ppo_cost = []
    avg_jitter_ppo_cost = []
    ci_avg_jitter_ppo_cost = []
    avg_gini_ppo_cost = []
    ci_avg_gini_ppo_cost = []
    avg_executionTime_ppo_cost = []
    ci_executionTime_ppo_cost = []

    avg_reward_ppo_latency = []
    ci_avg_reward_ppo_latency = []
    avg_ep_block_prob_ppo_latency = []
    ci_avg_ep_block_prob_ppo_latency = []
    avg_deployment_cost_ppo_latency = []
    ci_avg_deployment_cost_ppo_latency = []
    avg_total_latency_ppo_latency = []
    ci_avg_total_latency_ppo_latency = []
    avg_dl_ppo_latency = []
    ci_avg_dl_ppo_latency = []
    avg_ul_ppo_latency = []
    ci_avg_ul_ppo_latency = []
    avg_jitter_ppo_latency = []
    ci_avg_jitter_ppo_latency = []
    avg_gini_ppo_latency = []
    ci_avg_gini_ppo_latency = []
    avg_executionTime_ppo_latency = []
    ci_executionTime_ppo_latency = []

    avg_reward_ppo_inequality = []
    ci_avg_reward_ppo_inequality = []
    avg_ep_block_prob_ppo_inequality = []
    ci_avg_ep_block_prob_ppo_inequality = []
    avg_deployment_cost_ppo_inequality = []
    ci_avg_deployment_cost_ppo_inequality = []
    avg_total_latency_ppo_inequality = []
    ci_avg_total_latency_ppo_inequality = []
    avg_dl_ppo_inequality = []
    ci_avg_dl_ppo_inequality = []
    avg_ul_ppo_inequality = []
    ci_avg_ul_ppo_inequality = []
    avg_jitter_ppo_inequality = []
    ci_avg_jitter_ppo_inequality = []
    avg_gini_ppo_inequality = []
    ci_avg_gini_ppo_inequality = []
    avg_executionTime_ppo_inequality = []
    ci_executionTime_ppo_inequality = []

    avg_reward_ppo_bandwidth = []
    ci_avg_reward_ppo_bandwidth = []
    avg_ep_block_prob_ppo_bandwidth = []
    ci_avg_ep_block_prob_ppo_bandwidth = []
    avg_deployment_cost_ppo_bandwidth = []
    ci_avg_deployment_cost_ppo_bandwidth = []
    avg_total_latency_ppo_bandwidth = []
    ci_avg_total_latency_ppo_bandwidth = []
    avg_dl_ppo_bandwidth = []
    ci_avg_dl_ppo_bandwidth = []
    avg_ul_ppo_bandwidth = []
    ci_avg_ul_ppo_bandwidth = []
    avg_jitter_ppo_bandwidth = []
    ci_avg_jitter_ppo_bandwidth = []
    avg_gini_ppo_bandwidth = []
    ci_avg_gini_ppo_bandwidth = []
    avg_executionTime_ppo_bandwidth = []
    ci_executionTime_ppo_bandwidth = []

    # Baselines: for cpu, latency, cost, binpack, karmada
    path_cost = path + version + "/baselines/cost/scalability/"
    path_bandwidth = path + version + "/baselines/bandwidth/scalability/"
    path_latency = path + version + "/baselines/latency/scalability/"
    baseline = 'baseline'

    avg_reward_cost = []
    ci_avg_reward_cost = []
    avg_ep_block_prob_cost = []
    ci_avg_ep_block_prob_cost = []
    avg_deployment_cost_cost = []
    ci_avg_deployment_cost_cost = []
    avg_total_latency_cost = []
    ci_avg_total_latency_cost = []
    avg_dl_cost = []
    ci_avg_dl_cost = []
    avg_ul_cost = []
    ci_avg_ul_cost = []
    avg_jitter_cost = []
    ci_avg_jitter_cost = []
    avg_gini_cost = []
    ci_avg_gini_cost = []
    avg_executionTime_cost = []
    ci_executionTime_cost = []

    avg_reward_bandwidth = []
    ci_avg_reward_bandwidth = []
    avg_ep_block_prob_bandwidth = []
    ci_avg_ep_block_prob_bandwidth = []
    avg_deployment_cost_bandwidth = []
    ci_avg_deployment_cost_bandwidth = []
    avg_total_latency_bandwidth = []
    ci_avg_total_latency_bandwidth = []
    avg_dl_bandwidth = []
    ci_avg_dl_bandwidth = []
    avg_ul_bandwidth = []
    ci_avg_ul_bandwidth = []
    avg_jitter_bandwidth = []
    ci_avg_jitter_bandwidth = []
    avg_gini_bandwidth = []
    ci_avg_gini_bandwidth = []
    avg_executionTime_bandwidth = []
    ci_executionTime_bandwidth = []

    avg_reward_latency = []
    ci_avg_reward_latency = []
    avg_ep_block_prob_latency = []
    ci_avg_ep_block_prob_latency = []
    avg_deployment_cost_latency = []
    ci_avg_deployment_cost_latency = []
    avg_total_latency_latency = []
    ci_avg_total_latency_latency = []
    avg_dl_latency = []
    ci_avg_dl_latency = []
    avg_ul_latency = []
    ci_avg_ul_latency = []
    avg_jitter_latency = []
    ci_avg_jitter_latency = []
    avg_gini_latency = []
    ci_avg_gini_latency = []
    avg_executionTime_latency = []
    ci_executionTime_latency = []

    if os.path.exists(path_ppo_cost):
        for file in glob.glob(f"{path_ppo_cost}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_ppo_cost, ci_avg_reward_ppo_cost,
                           avg_ep_block_prob_ppo_cost, ci_avg_ep_block_prob_ppo_cost,
                           avg_deployment_cost_ppo_cost, ci_avg_deployment_cost_ppo_cost,
                           avg_total_latency_ppo_cost, ci_avg_total_latency_ppo_cost,
                           avg_dl_ppo_cost, ci_avg_dl_ppo_cost,
                           avg_ul_ppo_cost, ci_avg_ul_ppo_cost,
                           avg_jitter_ppo_cost, ci_avg_jitter_ppo_cost,
                           avg_gini_ppo_cost, ci_avg_gini_ppo_cost,
                           avg_executionTime_ppo_cost, ci_executionTime_ppo_cost
                           )

    if os.path.exists(path_ppo_latency):
        for file in glob.glob(f"{path_ppo_latency}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                            avg_reward_ppo_latency, ci_avg_reward_ppo_latency,
                            avg_ep_block_prob_ppo_latency, ci_avg_ep_block_prob_ppo_latency,
                            avg_deployment_cost_ppo_latency, ci_avg_deployment_cost_ppo_latency,
                            avg_total_latency_ppo_latency, ci_avg_total_latency_ppo_latency,
                            avg_dl_ppo_latency, ci_avg_dl_ppo_latency,
                            avg_ul_ppo_latency, ci_avg_ul_ppo_latency,
                            avg_jitter_ppo_latency, ci_avg_jitter_ppo_latency,
                            avg_gini_ppo_latency, ci_avg_gini_ppo_latency,
                           avg_executionTime_ppo_latency, ci_executionTime_ppo_latency
                           )

    if os.path.exists(path_ppo_inequality):
        for file in glob.glob(f"{path_ppo_inequality}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                            avg_reward_ppo_inequality, ci_avg_reward_ppo_inequality,
                            avg_ep_block_prob_ppo_inequality, ci_avg_ep_block_prob_ppo_inequality,
                            avg_deployment_cost_ppo_inequality, ci_avg_deployment_cost_ppo_inequality,
                            avg_total_latency_ppo_inequality, ci_avg_total_latency_ppo_inequality,
                            avg_dl_ppo_inequality, ci_avg_dl_ppo_inequality,
                            avg_ul_ppo_inequality, ci_avg_ul_ppo_inequality,
                            avg_jitter_ppo_inequality, ci_avg_jitter_ppo_inequality,
                            avg_gini_ppo_inequality, ci_avg_gini_ppo_inequality,
                            avg_executionTime_ppo_inequality, ci_executionTime_ppo_inequality
                            )

    if os.path.exists(path_ppo_bandwidth):
        for file in glob.glob(f"{path_ppo_bandwidth}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                            avg_reward_ppo_bandwidth, ci_avg_reward_ppo_bandwidth,
                            avg_ep_block_prob_ppo_bandwidth, ci_avg_ep_block_prob_ppo_bandwidth,
                            avg_deployment_cost_ppo_bandwidth, ci_avg_deployment_cost_ppo_bandwidth,
                            avg_total_latency_ppo_bandwidth, ci_avg_total_latency_ppo_bandwidth,
                            avg_dl_ppo_bandwidth, ci_avg_dl_ppo_bandwidth,
                            avg_ul_ppo_bandwidth, ci_avg_ul_ppo_bandwidth,
                            avg_jitter_ppo_bandwidth, ci_avg_jitter_ppo_bandwidth,
                            avg_gini_ppo_bandwidth, ci_avg_gini_ppo_bandwidth,
                            avg_executionTime_ppo_bandwidth, ci_executionTime_ppo_bandwidth
                            )

    # Baselines
    if os.path.exists(path_cost):
        for file in glob.glob(f"{path_cost}/*_baselines_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, baseline,
                            avg_reward_cost, ci_avg_reward_cost,
                            avg_ep_block_prob_cost, ci_avg_ep_block_prob_cost,
                            avg_deployment_cost_cost, ci_avg_deployment_cost_cost,
                            avg_total_latency_cost, ci_avg_total_latency_cost,
                            avg_dl_cost, ci_avg_dl_cost,
                            avg_ul_cost, ci_avg_ul_cost,
                            avg_jitter_cost, ci_avg_jitter_cost,
                            avg_gini_cost, ci_avg_gini_cost,
                            avg_executionTime_cost, ci_executionTime_cost
                            )

    if os.path.exists(path_bandwidth):
        for file in glob.glob(f"{path_bandwidth}/*_baselines_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, baseline,
                            avg_reward_bandwidth, ci_avg_reward_bandwidth,
                            avg_ep_block_prob_bandwidth, ci_avg_ep_block_prob_bandwidth,
                            avg_deployment_cost_bandwidth, ci_avg_deployment_cost_bandwidth,
                            avg_total_latency_bandwidth, ci_avg_total_latency_bandwidth,
                            avg_dl_bandwidth, ci_avg_dl_bandwidth,
                            avg_ul_bandwidth, ci_avg_ul_bandwidth,
                            avg_jitter_bandwidth, ci_avg_jitter_bandwidth,
                            avg_gini_bandwidth, ci_avg_gini_bandwidth,
                            avg_executionTime_bandwidth, ci_executionTime_bandwidth
                            )

    if os.path.exists(path_latency):
        for file in glob.glob(f"{path_latency}/*_baselines_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, baseline,
                            avg_reward_latency, ci_avg_reward_latency,
                            avg_ep_block_prob_latency, ci_avg_ep_block_prob_latency,
                            avg_deployment_cost_latency, ci_avg_deployment_cost_latency,
                            avg_total_latency_latency, ci_avg_total_latency_latency,
                            avg_dl_latency, ci_avg_dl_latency,
                            avg_ul_latency, ci_avg_ul_latency,
                            avg_jitter_latency, ci_avg_jitter_latency,
                            avg_gini_latency, ci_avg_gini_latency,
                            avg_executionTime_latency, ci_executionTime_latency
                            )

    # Accumulated Reward
    fig = plt.figure()
    x = [1, 2, 4, 6, 8, 10, 12]

    plt.errorbar(x, avg_reward_ppo_cost, yerr=ci_avg_reward_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='MaskPPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_reward_ppo_latency, yerr=ci_avg_reward_ppo_latency,
                 marker='o', linestyle='dotted',
                 color='#D95319', label='MaskPPO (Latency)', markersize=6)

    plt.errorbar(x, avg_reward_ppo_inequality, yerr=ci_avg_reward_ppo_inequality,
                 marker='^', linestyle='dashed',
                 color='#3399FF', label='MaskPPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_reward_ppo_bandwidth, yerr=ci_avg_reward_ppo_bandwidth,
                    marker='x', linestyle='dashdot',
                    color='#FFA500', label='MaskPPO (Bandwidth)', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=max_reward, color='black', linestyle='--', label="max reward= " + str(max_reward))
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 100)

    # set x-axis label
    plt.xlabel("Factor", fontsize=14)

    # set y-axis label
    plt.ylabel("Accumulated Reward", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_factor_reward.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Avg. Cost
    plt.errorbar(x, avg_deployment_cost_ppo_cost, yerr=ci_avg_deployment_cost_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='MaskPPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_deployment_cost_ppo_latency, yerr=ci_avg_deployment_cost_ppo_latency,
                 marker='o', linestyle='dotted',
                 color='#D95319', label='MaskPPO (Latency)', markersize=6)

    plt.errorbar(x, avg_deployment_cost_ppo_inequality, yerr=ci_avg_deployment_cost_ppo_inequality,
                 marker='^', linestyle='dashed',
                 color='#3399FF', label='MaskPPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_deployment_cost_ppo_bandwidth, yerr=ci_avg_deployment_cost_ppo_bandwidth,
                 marker='x', linestyle='dashdot',
                 color='#FFA500', label='MaskPPO (Bandwidth)', markersize=6)

    # Baselines
    plt.errorbar(x, avg_deployment_cost_cost, yerr=ci_avg_deployment_cost_cost,
                    linestyle='-.',
                    marker="s", color='#E897E8', label='Cost-Greedy',
                    markersize=6)

    plt.errorbar(x, avg_deployment_cost_bandwidth, yerr=ci_avg_deployment_cost_bandwidth,
                    marker='o', linestyle='--',
                    color='#808080', label='Bandwidth Greedy', markersize=6)

    plt.errorbar(x, avg_deployment_cost_latency, yerr=ci_avg_deployment_cost_latency,
                    marker='^', linestyle='dotted',
                    color='#DAB9AA', label='Latency Greedy', markersize=6)

    # plt.errorbar(x, avg_cost_ppo, yerr=ci_avg_cost_ppo, linestyle=None, marker="s", color='#3399FF',
    #             label='Deepsets PPO', markersize=6)

    # plt.errorbar(x, avg_cost_dqn, yerr=ci_avg_cost_dqn, color='#EDB120',
    #             linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 16)

    # set x-axis label
    plt.xlabel("Factor", fontsize=14)

    # set y-axis label
    plt.ylabel("Deployment Cost", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_factor_cost.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Avg latency
    plt.errorbar(x, avg_total_latency_ppo_cost, yerr=ci_avg_total_latency_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='MaskPPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_total_latency_ppo_latency, yerr=ci_avg_total_latency_ppo_latency,
                    marker='o', linestyle='dotted',
                    color='#D95319', label='MaskPPO (Latency)', markersize=6)

    plt.errorbar(x, avg_total_latency_ppo_inequality, yerr=ci_avg_total_latency_ppo_inequality,
                    marker='^', linestyle='dashed',
                    color='#3399FF', label='MaskPPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_total_latency_ppo_bandwidth, yerr=ci_avg_total_latency_ppo_bandwidth,
                    marker='x', linestyle='dashdot',
                    color='#FFA500', label='MaskPPO (Bandwidth)', markersize=6)

    # Baselines
    plt.errorbar(x, avg_total_latency_cost, yerr=ci_avg_total_latency_cost,
                    linestyle='-.',
                    marker="s", color='#E897E8', label='Cost-Greedy',
                    markersize=6)

    plt.errorbar(x, avg_total_latency_bandwidth, yerr=ci_avg_total_latency_bandwidth,
                    marker='o', linestyle='--',
                    color='#808080', label='Bandwidth Greedy', markersize=6)

    plt.errorbar(x, avg_total_latency_latency, yerr=ci_avg_total_latency_latency,
                    marker='^', linestyle='dotted',
                    color='#DAB9AA', label='Latency Greedy', markersize=6)

    # plt.errorbar(x, avg_latency_ppo, yerr=ci_avg_latency_ppo, linestyle=None, marker="s", color='#3399FF',
    #              label='Deepsets PPO', markersize=6)

    # plt.errorbar(x, avg_latency_dqn, yerr=ci_avg_latency_dqn, color='#EDB120',
    #              linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 120)

    # set x-axis label
    plt.xlabel("Factor", fontsize=14)

    # set y-axis label
    plt.ylabel("Avg. Total Latency (in ms)", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_factor_latency.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Episode Block Prob
    print("PPO: {}".format(avg_ep_block_prob_ppo_cost))
    print("Cost-greedy: {}".format(avg_ep_block_prob_cost))

    plt.errorbar(x, avg_ep_block_prob_ppo_cost, yerr=ci_avg_ep_block_prob_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='MaskPPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_ep_block_prob_ppo_latency, yerr=ci_avg_ep_block_prob_ppo_latency,
                    marker='o', linestyle='dotted',
                    color='#D95319', label='MaskPPO (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_block_prob_ppo_inequality, yerr=ci_avg_ep_block_prob_ppo_inequality,
                    marker='^', linestyle='dashed',
                    color='#3399FF', label='MaskPPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_ep_block_prob_ppo_bandwidth, yerr=ci_avg_ep_block_prob_ppo_bandwidth,
                    marker='x', linestyle='dashdot',
                    color='#FFA500', label='MaskPPO (Bandwidth)', markersize=6)

    # Baselines
    plt.errorbar(x, avg_ep_block_prob_cost, yerr=ci_avg_ep_block_prob_cost,
                    linestyle='-.',
                    marker="s", color='#E897E8', label='Cost-Greedy',
                    markersize=6)

    plt.errorbar(x, avg_ep_block_prob_bandwidth, yerr=ci_avg_ep_block_prob_bandwidth,
                    marker='o', linestyle='--',
                    color='#808080', label='Bandwidth Greedy', markersize=6)

    plt.errorbar(x, avg_ep_block_prob_latency, yerr=ci_avg_ep_block_prob_latency,
                    marker='^', linestyle='dotted',
                    color='#DAB9AA', label='Latency Greedy', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 1.0)

    # set x-axis label
    plt.xlabel("Factor", fontsize=14)

    # set y-axis label
    plt.ylabel("Percentage of Rejected Requests", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_factor_rejected_requests.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Gini Coefficient
    plt.errorbar(x, avg_gini_ppo_cost, yerr=ci_avg_gini_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='MaskPPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_gini_ppo_latency, yerr=ci_avg_gini_ppo_latency,
                    marker='o', linestyle='dotted',
                    color='#D95319', label='MaskPPO (Latency)', markersize=6)

    plt.errorbar(x, avg_gini_ppo_inequality, yerr=ci_avg_gini_ppo_inequality,
                    marker='^', linestyle='dashed',
                    color='#3399FF', label='MaskPPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_gini_ppo_bandwidth, yerr=ci_avg_gini_ppo_bandwidth,
                    marker='x', linestyle='dashdot',
                    color='#FFA500', label='MaskPPO (Bandwidth)', markersize=6)


    # Baselines
    plt.errorbar(x, avg_gini_cost, yerr=ci_avg_gini_cost,
                    linestyle='-.',
                    marker="s", color='#E897E8', label='Cost-Greedy',
                    markersize=6)

    plt.errorbar(x, avg_gini_bandwidth, yerr=ci_avg_gini_bandwidth,
                    marker='o', linestyle='--',
                    color='#808080', label='Bandwidth Greedy', markersize=6)

    plt.errorbar(x, avg_gini_latency, yerr=ci_avg_gini_latency,
                    marker='^', linestyle='dotted',
                    color='#DAB9AA', label='Latency Greedy', markersize=6)

    # plt.errorbar(x, avg_ep_block_prob_ppo, yerr=ci_avg_ep_block_prob_ppo, linestyle=None, marker="s", color='#3399FF',
    #              label='Deepsets PPO', markersize=6)
    # plt.errorbar(x, avg_ep_block_prob_dqn, yerr=ci_avg_ep_block_prob_dqn, color='#EDB120',
    #              linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 0.8)

    # set x-axis label
    plt.xlabel("Factor", fontsize=14)

    # set y-axis label
    plt.ylabel("Gini Coefficient", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_factor_gini.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # avg_dl
    plt.errorbar(x, avg_dl_ppo_cost, yerr=ci_avg_dl_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='MaskPPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_dl_ppo_latency, yerr=ci_avg_dl_ppo_latency,
                    marker='o', linestyle='dotted',
                    color='#D95319', label='MaskPPO (Latency)', markersize=6)

    plt.errorbar(x, avg_dl_ppo_inequality, yerr=ci_avg_dl_ppo_inequality,
                    marker='^', linestyle='dashed',
                    color='#3399FF', label='MaskPPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_dl_ppo_bandwidth, yerr=ci_avg_dl_ppo_bandwidth,
                    marker='x', linestyle='dashdot',
                    color='#FFA500', label='MaskPPO (Bandwidth)', markersize=6)

    # Baselines

    plt.errorbar(x, avg_dl_cost, yerr=ci_avg_dl_cost,
                    linestyle='-.',
                    marker="s", color='#E897E8', label='Cost-Greedy',
                    markersize=6)

    plt.errorbar(x, avg_dl_bandwidth, yerr=ci_avg_dl_bandwidth,
                    marker='o', linestyle='--',
                    color='#808080', label='Bandwidth Greedy', markersize=6)

    plt.errorbar(x, avg_dl_latency, yerr=ci_avg_dl_latency,
                    marker='^', linestyle='dotted',
                    color='#DAB9AA', label='Latency Greedy', markersize=6)

    # plt.errorbar(x, avg_dl_ppo, yerr=ci_avg_dl_ppo, linestyle=None, marker="s", color='#3399FF',
    #              label='Deepsets PPO', markersize=6)

    # plt.errorbar(x, avg_dl_dqn, yerr=ci_avg_dl_dqn, color='#EDB120',
    #              linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 330)

    # set x-axis label
    plt.xlabel("Factor", fontsize=14)
    plt.ylabel("Avg. Downlink (in Mbps)", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_factor_dl.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # avg_ul
    plt.errorbar(x, avg_ul_ppo_cost, yerr=ci_avg_ul_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='MaskPPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_ul_ppo_latency, yerr=ci_avg_ul_ppo_latency,
                    marker='o', linestyle='dotted',
                    color='#D95319', label='MaskPPO (Latency)', markersize=6)

    plt.errorbar(x, avg_ul_ppo_inequality, yerr=ci_avg_ul_ppo_inequality,
                    marker='^', linestyle='dashed',
                    color='#3399FF', label='MaskPPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_ul_ppo_bandwidth, yerr=ci_avg_ul_ppo_bandwidth,
                    marker='x', linestyle='dashdot',
                    color='#FFA500', label='MaskPPO (Bandwidth)', markersize=6)

    # Baselines

    plt.errorbar(x, avg_ul_cost, yerr=ci_avg_ul_cost,
                    linestyle='-.',
                    marker="s", color='#E897E8', label='Cost-Greedy',
                    markersize=6)

    plt.errorbar(x, avg_ul_bandwidth, yerr=ci_avg_ul_bandwidth,
                    marker='o', linestyle='--',
                    color='#808080', label='Bandwidth Greedy', markersize=6)

    plt.errorbar(x, avg_ul_latency, yerr=ci_avg_ul_latency,
                    marker='^', linestyle='dotted',
                    color='#DAB9AA', label='Latency Greedy', markersize=6)

    # plt.errorbar(x, avg_ul_ppo, yerr=ci_avg_ul_ppo, linestyle=None, marker="s", color='#3399FF',
    #              label='Deepsets PPO', markersize=6)

    # plt.errorbar(x, avg_ul_dqn, yerr=ci_avg_ul_dqn, color='#EDB120',
    #              linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 50)

    # set x-axis label
    plt.xlabel("Factor", fontsize=14)
    plt.ylabel("Avg. Uplink (in Mbps)", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_factor_ul.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # avg_jitter
    plt.errorbar(x, avg_jitter_ppo_cost, yerr=ci_avg_jitter_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='MaskPPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_jitter_ppo_latency, yerr=ci_avg_jitter_ppo_latency,
                    marker='o', linestyle='dotted',
                    color='#D95319', label='MaskPPO (Latency)', markersize=6)

    plt.errorbar(x, avg_jitter_ppo_inequality, yerr=ci_avg_jitter_ppo_inequality,
                    marker='^', linestyle='dashed',
                    color='#3399FF', label='MaskPPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_jitter_ppo_bandwidth, yerr=ci_avg_jitter_ppo_bandwidth,
                    marker='x', linestyle='dashdot',
                    color='#FFA500', label='MaskPPO (Bandwidth)', markersize=6)

    # Baselines

    plt.errorbar(x, avg_jitter_cost, yerr=ci_avg_jitter_cost,
                    linestyle='-.',
                    marker="s", color='#E897E8', label='Cost-Greedy',
                    markersize=6)

    plt.errorbar(x, avg_jitter_bandwidth, yerr=ci_avg_jitter_bandwidth,
                    marker='o', linestyle='--',
                    color='#808080', label='Bandwidth Greedy', markersize=6)

    plt.errorbar(x, avg_jitter_latency, yerr=ci_avg_jitter_latency,
                    marker='^', linestyle='dotted',
                    color='#DAB9AA', label='Latency Greedy', markersize=6)

    # plt.errorbar(x, avg_jitter_ppo, yerr=ci_avg_jitter_ppo, linestyle=None, marker="s", color='#3399FF',
    #              label='Deepsets PPO', markersize=6)

    # plt.errorbar(x, avg_jitter_dqn, yerr=ci_avg_jitter_dqn, color='#EDB120',
    #              linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 10)

    # set x-axis label
    plt.xlabel("Factor", fontsize=14)
    plt.ylabel("Avg. Jitter (in ms)", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_factor_jitter.pdf', dpi=250, bbox_inches='tight')
    plt.close()




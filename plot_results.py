import logging
from collections import namedtuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')

stats = namedtuple("episode_stats",
                   ["ppo_cost_rewards", "ppo_latency_rewards", "ppo_inequality_rewards", "ppo_bandwidth_rewards",
                    "ppo_bandcost_rewards", "ppo_bandlat_rewards", "ppo_costineq_rewards", "ppo_latcost_rewards",
                    "ppo_latineq_rewards",

                    # block probability
                    "ppo_cost_ep_block_prob", "ppo_latency_ep_block_prob", "ppo_inequality_ep_block_prob",
                    "ppo_bandwidth_ep_block_prob",
                    "ppo_bandcost_ep_block_prob", "ppo_bandlat_ep_block_prob", "ppo_costineq_ep_block_prob",
                    "ppo_latcost_ep_block_prob",
                    "ppo_latineq_ep_block_prob",

                    # "ppo_latcost_ep_block_prob",
                    # "ppo_latineq_ep_block_prob", "ppo_costineq_ep_block_prob", "ppo_balanced_ep_block_prob",
                    # "ppo_favorlat_ep_block_prob",
                    # latency
                    "ppo_cost_latency", "ppo_latency_latency", "ppo_inequality_latency", "ppo_bandwidth_latency",
                    "ppo_bandcost_latency",
                    "ppo_bandlat_latency", "ppo_costineq_latency", "ppo_latcost_latency",
                    "ppo_latineq_latency",

                    "ppo_cost_cost", "ppo_latency_cost", "ppo_inequality_cost", "ppo_bandwidth_cost",
                    "ppo_bandcost_cost",
                    "ppo_bandlat_cost", "ppo_costineq_cost", "ppo_latcost_cost", "ppo_latineq_cost",

                    "ppo_cost_ul", "ppo_latency_ul", "ppo_inequality_ul", "ppo_bandwidth_ul", "ppo_bandcost_ul",
                    "ppo_bandlat_ul", "ppo_costineq_ul", "ppo_latcost_ul", "ppo_latineq_ul",

                    "ppo_cost_dl", "ppo_latency_dl", "ppo_inequality_dl", "ppo_bandwidth_dl", "ppo_bandcost_dl",
                    "ppo_bandlat_dl", "ppo_costineq_dl", "ppo_latcost_dl", "ppo_latineq_dl",

                    # gini
                    "ppo_cost_gini", "ppo_latency_gini", "ppo_inequality_gini", "ppo_bandwidth_gini",
                    "ppo_bandcost_gini", "ppo_bandlat_gini", "ppo_costineq_gini", "ppo_latcost_gini",
                    "ppo_latineq_gini",

                    ])


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def plot_stats(figName, stats, max_reward, xlim, smoothing_window=10):
    # latency greedy: C521EE
    # resource greedy: 7A21EE

    # Plot the episode reward over time
    ppo_cost = pd.Series(stats.ppo_cost_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_rewards).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    ppo_bandwidth = pd.Series(stats.ppo_bandwidth_rewards).rolling(smoothing_window,
                                                                   min_periods=smoothing_window).mean()
    ppo_bandlat = pd.Series(stats.ppo_bandlat_rewards).rolling(smoothing_window,
                                                               min_periods=smoothing_window).mean()
    ppo_bandcost = pd.Series(stats.ppo_bandcost_rewards).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_rewards).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_rewards).rolling(smoothing_window,
                                                               min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_rewards).rolling(smoothing_window,
                                                               min_periods=smoothing_window).mean()

    fig = plt.figure()
    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='MaskPPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='MaskPPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='MaskPPO (Inequality)')
    plt.plot(ppo_bandwidth,
             linestyle='dashdot', color='#FFA500', label='MaskPPO (Bandwidth)')

    plt.plot(ppo_bandcost,
             linestyle='-.', color='#EDB120', label='MaskPPO (BandCost)')
    plt.plot(ppo_bandlat,
             linestyle='dashdot', color='#7A21EE', label='MaskPPO (BandLat)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='MaskPPO (CostIneq)')
    plt.plot(ppo_latcost,
             linestyle='dashed', color='#D74281', label='MaskPPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='-.', color='#434DD7', label='MaskPPO (LatIneq)')

    # specifying horizontal line type
    plt.axhline(y=max_reward, color='black', linestyle='--', label="max reward= " + str(max_reward))
    # plt.yscale('log')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.xlim(smoothing_window, xlim)
    plt.ylim(0, 150)
    plt.legend(ncol=2)

    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_reward.pdf', dpi=250, bbox_inches='tight')

    ppo_cost = pd.Series(stats.ppo_cost_ep_block_prob).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_ep_block_prob).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_ep_block_prob).rolling(smoothing_window,
                                                                           min_periods=smoothing_window).mean()
    ppo_bandwidth = pd.Series(stats.ppo_bandwidth_ep_block_prob).rolling(smoothing_window,
                                                                         min_periods=smoothing_window).mean()
    ppo_bandcost = pd.Series(stats.ppo_bandcost_ep_block_prob).rolling(smoothing_window,
                                                                       min_periods=smoothing_window).mean()
    ppo_bandlat = pd.Series(stats.ppo_bandlat_ep_block_prob).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_ep_block_prob).rolling(smoothing_window,
                                                                       min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_ep_block_prob).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_ep_block_prob).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()

    fig = plt.figure()
    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='MaskPPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='MaskPPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='MaskPPO (Inequality)')
    plt.plot(ppo_bandwidth,
             linestyle='dashdot', color='#FFA500', label='MaskPPO (Bandwidth)')
    plt.plot(ppo_bandcost,
             linestyle='-.', color='#EDB120', label='MaskPPO (BandCost)')
    plt.plot(ppo_bandlat,
             linestyle='dashdot', color='#7A21EE', label='MaskPPO (BandLat)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='MaskPPO (CostIneq)')
    plt.plot(ppo_latcost,
             linestyle='dashed', color='#D74281', label='MaskPPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='-.', color='#434DD7', label='MaskPPO (LatIneq)')

    plt.xlabel("Episode")
    plt.ylabel("Percentage of Rejected Requests")
    plt.xlim(smoothing_window, xlim)
    plt.ylim(0, 0.2)
    plt.legend(ncol=2)
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_block_probability.pdf', dpi=250, bbox_inches='tight')

    # Avg latency
    fig = plt.figure()
    ppo_cost = pd.Series(stats.ppo_cost_latency).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_latency).rolling(smoothing_window,
                                                               min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_latency).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    ppo_bandwidth = pd.Series(stats.ppo_bandwidth_latency).rolling(smoothing_window,
                                                                   min_periods=smoothing_window).mean()

    ppo_bandcost = pd.Series(stats.ppo_bandcost_latency).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()
    ppo_bandlat = pd.Series(stats.ppo_bandlat_latency).rolling(smoothing_window,
                                                               min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_latency).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_latency).rolling(smoothing_window,
                                                               min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_latency).rolling(smoothing_window,
                                                               min_periods=smoothing_window).mean()

    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='MaskPPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='MaskPPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='MaskPPO (Inequality)')
    plt.plot(ppo_bandwidth,
             linestyle='dashdot', color='#FFA500', label='MaskPPO (Bandwidth)')
    plt.plot(ppo_bandcost,
             linestyle='-.', color='#EDB120', label='MaskPPO (BandCost)')
    plt.plot(ppo_bandlat,
             linestyle='dashdot', color='#7A21EE', label='MaskPPO (BandLat)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='MaskPPO (CostIneq)')
    plt.plot(ppo_latcost,
             linestyle='dashed', color='#D74281', label='MaskPPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='-.', color='#434DD7', label='MaskPPO (LatIneq)')

    plt.xlabel("Episode")
    plt.ylabel("Avg. Latency (in ms)")
    plt.xlim(smoothing_window, xlim)
    # plt.ylim(0, 100)
    plt.legend(ncol=2)
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_latency.pdf', dpi=250, bbox_inches='tight')

    # Avg cost
    fig = plt.figure()
    ppo_cost = pd.Series(stats.ppo_cost_cost).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_cost).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_cost).rolling(smoothing_window,
                                                                  min_periods=smoothing_window).mean()
    ppo_bandwidth = pd.Series(stats.ppo_bandwidth_cost).rolling(smoothing_window,
                                                                min_periods=smoothing_window).mean()

    ppo_bandcost = pd.Series(stats.ppo_bandcost_cost).rolling(smoothing_window,
                                                              min_periods=smoothing_window).mean()
    ppo_bandlat = pd.Series(stats.ppo_bandlat_cost).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_cost).rolling(smoothing_window,
                                                              min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_cost).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_cost).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()

    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='MaskPPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='MaskPPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='MaskPPO (Inequality)')
    plt.plot(ppo_bandwidth,
             linestyle='dashdot', color='#FFA500', label='MaskPPO (Bandwidth)')
    plt.plot(ppo_bandcost,
             linestyle='-.', color='#EDB120', label='MaskPPO (BandCost)')
    plt.plot(ppo_bandlat,
             linestyle='dashdot', color='#7A21EE', label='MaskPPO (BandLat)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='MaskPPO (CostIneq)')
    plt.plot(ppo_latcost,
             linestyle='dashed', color='#D74281', label='MaskPPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='-.', color='#434DD7', label='MaskPPO (LatIneq)')

    plt.xlabel("Episode")
    plt.ylabel("Avg. Cost (in units)")
    plt.xlim(smoothing_window, xlim)
    plt.ylim(0, 16)
    plt.legend(ncol=2)
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_cost.pdf', dpi=250, bbox_inches='tight')

    # Avg UL
    fig = plt.figure()
    ppo_cost = pd.Series(stats.ppo_cost_ul).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_ul).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_ul).rolling(smoothing_window,
                                                                min_periods=smoothing_window).mean()
    ppo_bandwidth = pd.Series(stats.ppo_bandwidth_ul).rolling(smoothing_window,
                                                              min_periods=smoothing_window).mean()
    ppo_bandcost = pd.Series(stats.ppo_bandcost_ul).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_bandlat = pd.Series(stats.ppo_bandlat_ul).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_ul).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_ul).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_ul).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()

    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='MaskPPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='MaskPPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='MaskPPO (Inequality)')
    plt.plot(ppo_bandwidth,
             linestyle='dashdot', color='#FFA500', label='MaskPPO (Bandwidth)')
    plt.plot(ppo_bandcost,
             linestyle='-.', color='#EDB120', label='MaskPPO (BandCost)')
    plt.plot(ppo_bandlat,
             linestyle='dashdot', color='#7A21EE', label='MaskPPO (BandLat)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='MaskPPO (CostIneq)')
    plt.plot(ppo_latcost,
             linestyle='dashed', color='#D74281', label='MaskPPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='-.', color='#434DD7', label='MaskPPO (LatIneq)')

    plt.xlabel("Episode")
    plt.ylabel("Avg. UL (in Mbits/s)")
    plt.xlim(smoothing_window, xlim)
    plt.ylim(0, 60)
    plt.legend(ncol=2)
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_ul.pdf', dpi=250, bbox_inches='tight')

    # Avg DL
    fig = plt.figure()
    ppo_cost = pd.Series(stats.ppo_cost_dl).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_dl).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_dl).rolling(smoothing_window,
                                                                min_periods=smoothing_window).mean()
    ppo_bandwidth = pd.Series(stats.ppo_bandwidth_dl).rolling(smoothing_window,
                                                              min_periods=smoothing_window).mean()
    ppo_bandlat = pd.Series(stats.ppo_bandlat_dl).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()
    ppo_bandcost = pd.Series(stats.ppo_bandcost_dl).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_dl).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_dl).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_dl).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()

    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='MaskPPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='MaskPPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='MaskPPO (Inequality)')
    plt.plot(ppo_bandwidth,
             linestyle='dashdot', color='#FFA500', label='MaskPPO (Bandwidth)')
    plt.plot(ppo_bandcost,
             linestyle='-.', color='#EDB120', label='MaskPPO (BandCost)')
    plt.plot(ppo_bandlat,
             linestyle='dashdot', color='#7A21EE', label='MaskPPO (BandLat)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='MaskPPO (CostIneq)')
    plt.plot(ppo_latcost,
             linestyle='dashed', color='#D74281', label='MaskPPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='-.', color='#434DD7', label='MaskPPO (LatIneq)')

    plt.xlabel("Episode")
    plt.ylabel("Avg. DL (in Mbits/s)")
    plt.xlim(smoothing_window, xlim)
    # plt.ylim(0, 100)
    plt.legend(ncol=2)
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_dl.pdf', dpi=250, bbox_inches='tight')

    # Gini
    fig = plt.figure()
    ppo_cost = pd.Series(stats.ppo_cost_gini).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_gini).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_gini).rolling(smoothing_window,
                                                                  min_periods=smoothing_window).mean()
    ppo_bandwidth = pd.Series(stats.ppo_bandwidth_gini).rolling(smoothing_window,
                                                                min_periods=smoothing_window).mean()
    ppo_bandcost = pd.Series(stats.ppo_bandcost_gini).rolling(smoothing_window,
                                                              min_periods=smoothing_window).mean()
    ppo_bandlat = pd.Series(stats.ppo_bandlat_gini).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_gini).rolling(smoothing_window,
                                                              min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_gini).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_gini).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()

    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='MaskPPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='MaskPPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='MaskPPO (Inequality)')
    plt.plot(ppo_bandwidth,
             linestyle='dashdot', color='#FFA500', label='MaskPPO (Bandwidth)')
    plt.plot(ppo_bandcost,
             linestyle='-.', color='#EDB120', label='MaskPPO (BandCost)')
    plt.plot(ppo_bandlat,
             linestyle='dashdot', color='#7A21EE', label='MaskPPO (BandLat)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='MaskPPO (CostIneq)')
    plt.plot(ppo_latcost,
             linestyle='dashed', color='#D74281', label='MaskPPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='-.', color='#434DD7', label='MaskPPO (LatIneq)')

    plt.xlabel("Episode")
    plt.ylabel("Gini Coefficient")
    plt.xlim(smoothing_window, xlim)
    plt.ylim(0, 1)
    plt.legend(ncol=2)
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_gini.pdf', dpi=250, bbox_inches='tight')


def remove_duplicates(df, column_name):
    modified = df.drop_duplicates(subset=[column_name])
    modified.to_csv('karmada_gym_results.csv', index=False)
    return modified


def remove_empty_lines(df):
    print(df.isnull().sum())
    # Droping the empty rows
    modified = df.dropna()
    # Saving it to the csv file
    modified.to_csv('karmada_gym_results.csv', index=False)
    return modified


def print_statistics(df, alg_name):
    print("{} reward Mean: {}".format(alg_name, np.mean(df["reward"])))
    print("{} reward Std: {}".format(alg_name, 1.96 * np.std(df["reward"]) / np.sqrt(
        len(df["reward"]))))

    print("{} avg_deployment_cost Mean: {}".format(alg_name, np.mean(df["avg_deployment_cost"])))
    print("{} avg_deployment_cost Std: {}".format(alg_name, 1.96 * np.std(df["avg_deployment_cost"]) / np.sqrt(
        len(df["avg_deployment_cost"]))))

    print("{} rejected requests Mean: {}".format(alg_name, 100 * np.mean(df["ep_block_prob"])))
    print("{} rejected requests Std: {}".format(alg_name, 100 * 1.96 * np.std(df["ep_block_prob"]) / np.sqrt(
        len(df["ep_block_prob"]))))

    print("{} avg_processing_latency Mean: {}".format(alg_name, np.mean(df["avg_processing_latency"])))
    print("{} avg_processing_latency Std: {}".format(alg_name, 1.96 * np.std(df["avg_processing_latency"]) / np.sqrt(
        len(df["avg_processing_latency"]))))

    print("{} avg_access_latency Mean: {}".format(alg_name, np.mean(df["avg_access_latency"])))
    print("{} avg_access_latency Std: {}".format(alg_name, 1.96 * np.std(df["avg_access_latency"]) / np.sqrt(
        len(df["avg_access_latency"]))))

    print("{} avg_rtt Mean: {}".format(alg_name, np.mean(df["avg_rtt"])))
    print("{} avg_rtt Std: {}".format(alg_name, 1.96 * np.std(df["avg_rtt"]) / np.sqrt(
        len(df["avg_rtt"]))))

    print("{} avg_total_latency Mean: {}".format(alg_name, np.mean(df["avg_total_latency"])))
    print("{} avg_total_latency Std: {}".format(alg_name, 1.96 * np.std(df["avg_total_latency"]) / np.sqrt(
        len(df["avg_total_latency"]))))

    print("{} avg_ul Mean: {}".format(alg_name, np.mean(df["avg_ul"])))
    print("{} avg_ul Std: {}".format(alg_name,
                                     1.96 * np.std(df["avg_ul"]) / np.sqrt(
                                         len(df["avg_ul"]))))

    print("{} avg_dl Mean: {}".format(alg_name, np.mean(df["avg_dl"])))
    print("{} avg_dl Std: {}".format(alg_name,
                                     1.96 * np.std(df["avg_dl"]) / np.sqrt(
                                         len(df["avg_dl"]))))

    print("{} avg_jitter Mean: {}".format(alg_name, np.mean(df["avg_jitter"])))
    print("{} avg_jitter Std: {}".format(alg_name, 1.96 * np.std(df["avg_jitter"]) / np.sqrt(len(df["avg_jitter"]))))

    print("{} gini Mean: {}".format(alg_name, np.mean(df["gini"])))
    print("{} gini Std: {}".format(alg_name, 1.96 * np.std(df["gini"]) / np.sqrt(len(df["gini"]))))

    print("{} telia_requests Mean: {}".format(alg_name, np.mean(df["telia_requests"])))
    print("{} telia_requests Std: {}".format(alg_name, 1.96 * np.std(df["telia_requests"]) / np.sqrt(len(df["telia_requests"]))))

    print("{} telenor_requests Mean: {}".format(alg_name, np.mean(df["telenor_requests"])))
    print("{} telenor_requests Std: {}".format(alg_name, 1.96 * np.std(df["telenor_requests"]) / np.sqrt(len(df["telenor_requests"]))))

    print("{} ice_requests Mean: {}".format(alg_name, np.mean(df["ice_requests"])))
    print("{} ice_requests Std: {}".format(alg_name, 1.96 * np.std(df["ice_requests"]) / np.sqrt(len(df["ice_requests"]))))

    print("{} executionTime Mean: {}".format(alg_name, np.mean(df["executionTime"])))
    print("{} executionTime Std: {}".format(alg_name,
                                            1.96 * np.std(df["executionTime"]) / np.sqrt(len(df["executionTime"]))))


if __name__ == "__main__":
    reward = 'multi'  # cost, risk or latency
    version = 'v1'  # or v1
    path = "results/" + version + "/nne/"
    path_baselines = "results/" + version
    path_model = "_env_nne_num_nodes_4_reward_"
    testing_path = "testing/"  # "testing/"
    alg = 'mask_ppo'

    file_results = "vec_nne_gym_results.monitor.csv"
    # file_results_testing = "0_nne_gym_results_num_nodes_4.csv"

    window = 100  # 5 for testing / 100 for training
    max_reward = 100
    xlim = 2000

    # Training
    file_latency = path + reward + "/latency/" + alg + path_model + reward + "_totalSteps_200000_run_1/" + file_results  # testing_path +
    file_cost = path + reward + "/cost/" + alg + path_model + reward + "_totalSteps_200000_run_1/" + file_results
    file_inequality = path + reward + "/inequality/" + alg + path_model + reward + "_totalSteps_200000_run_1/" + file_results
    file_bandwidth = path + reward + "/bandwidth/" + alg + path_model + reward + "_totalSteps_200000_run_1/" + file_results

    file_bandcost = path + reward + "/bandcost/" + alg + path_model + reward + "_totalSteps_200000_run_1/" + file_results
    file_bandlat = path + reward + "/bandlat/" + alg + path_model + reward + "_totalSteps_200000_run_1/" + file_results
    file_costineq = path + reward + "/costineq/" + alg + path_model + reward + "_totalSteps_200000_run_1/" + file_results
    file_latineq = path + reward + "/latineq/" + alg + path_model + reward + "_totalSteps_200000_run_1/" + file_results
    file_latcost = path + reward + "/latcost/" + alg + path_model + reward + "_totalSteps_200000_run_1/" + file_results

    file_cost_greedy = path_baselines + "/baselines/cost/0_cost_baselines_num_nodes_4.csv"
    file_bandwidth_greedy = path_baselines + "/baselines/bandwidth/0_band_baselines_num_nodes_4.csv"
    file_latency_greedy = path_baselines + "/baselines/latency/0_lat_baselines_num_nodes_4.csv"

    df_latency = pd.read_csv(file_latency)
    df_cost = pd.read_csv(file_cost)
    df_inequality = pd.read_csv(file_inequality)
    df_bandwidth = pd.read_csv(file_bandwidth)
    df_cost_greedy = pd.read_csv(file_cost_greedy)
    df_bandwidth_greedy = pd.read_csv(file_bandwidth_greedy)
    df_latency_greedy = pd.read_csv(file_latency_greedy)

    df_bandcost = pd.read_csv(file_bandcost)
    df_bandlat = pd.read_csv(file_bandlat)
    df_costineq = pd.read_csv(file_costineq)
    df_latineq = pd.read_csv(file_latineq)
    df_latcost = pd.read_csv(file_latcost)


    print_statistics(df_cost, "df_cost")
    print_statistics(df_latency, "df_latency")
    print_statistics(df_inequality, "df_inequality")
    print_statistics(df_bandwidth, "df_bandwidth")
    print_statistics(df_bandcost, "df_bandcost")
    print_statistics(df_bandlat, "df_bandlat")
    print_statistics(df_costineq, "df_costineq")
    print_statistics(df_latineq, "df_latineq")
    print_statistics(df_latcost, "df_latcost")

    print_statistics(df_cost_greedy, "df_cost_greedy")
    print_statistics(df_bandwidth_greedy, "df_bandwidth_greedy")
    print_statistics(df_latency_greedy, "df_latency_greedy")


    # remove_empty_lines(df_a2c)
    # remove_empty_lines(df_mask_ppo)
    # remove_empty_lines(df_deepsets_ppo)
    # remove_empty_lines(df_deepsets_dqn)

    # remove_duplicates(df_deepsets_dqn, 'episode')

    stats = stats(
        ppo_cost_rewards=df_cost['reward'],
        ppo_latency_rewards=df_latency['reward'],
        ppo_inequality_rewards=df_inequality['reward'],
        ppo_bandwidth_rewards=df_bandwidth['reward'],
        ppo_bandcost_rewards=df_bandcost['reward'],
        ppo_bandlat_rewards=df_bandlat['reward'],
        ppo_costineq_rewards=df_costineq['reward'],
        ppo_latineq_rewards=df_latineq['reward'],
        ppo_latcost_rewards=df_latcost['reward'],

        ppo_cost_ep_block_prob=df_cost['ep_block_prob'],
        ppo_latency_ep_block_prob=df_latency['ep_block_prob'],
        ppo_inequality_ep_block_prob=df_inequality['ep_block_prob'],
        ppo_bandwidth_ep_block_prob=df_bandwidth['ep_block_prob'],
        ppo_bandcost_ep_block_prob=df_bandcost['ep_block_prob'],
        ppo_bandlat_ep_block_prob=df_bandlat['ep_block_prob'],
        ppo_costineq_ep_block_prob=df_costineq['ep_block_prob'],
        ppo_latineq_ep_block_prob=df_latineq['ep_block_prob'],
        ppo_latcost_ep_block_prob=df_latcost['ep_block_prob'],

        ppo_cost_latency=df_cost['avg_total_latency'],
        ppo_latency_latency=df_latency['avg_total_latency'],
        ppo_inequality_latency=df_inequality['avg_total_latency'],
        ppo_bandwidth_latency=df_bandwidth['avg_total_latency'],
        ppo_bandcost_latency=df_bandcost['avg_total_latency'],
        ppo_bandlat_latency=df_bandlat['avg_total_latency'],
        ppo_costineq_latency=df_costineq['avg_total_latency'],
        ppo_latineq_latency=df_latineq['avg_total_latency'],
        ppo_latcost_latency=df_latcost['avg_total_latency'],

        ppo_cost_cost=df_cost['avg_deployment_cost'],
        ppo_latency_cost=df_latency['avg_deployment_cost'],
        ppo_inequality_cost=df_inequality['avg_deployment_cost'],
        ppo_bandwidth_cost=df_bandwidth['avg_deployment_cost'],
        ppo_bandcost_cost=df_bandcost['avg_deployment_cost'],
        ppo_bandlat_cost=df_bandlat['avg_deployment_cost'],
        ppo_costineq_cost=df_costineq['avg_deployment_cost'],
        ppo_latineq_cost=df_latineq['avg_deployment_cost'],
        ppo_latcost_cost=df_latcost['avg_deployment_cost'],

        ppo_cost_ul=df_cost['avg_ul'],
        ppo_latency_ul=df_latency['avg_ul'],
        ppo_inequality_ul=df_inequality['avg_ul'],
        ppo_bandwidth_ul=df_bandwidth['avg_ul'],
        ppo_bandlat_ul=df_bandlat['avg_ul'],
        ppo_bandcost_ul=df_bandcost['avg_ul'],
        ppo_costineq_ul=df_costineq['avg_ul'],
        ppo_latineq_ul=df_latineq['avg_ul'],
        ppo_latcost_ul=df_latcost['avg_ul'],

        ppo_cost_dl=df_cost['avg_dl'],
        ppo_latency_dl=df_latency['avg_dl'],
        ppo_inequality_dl=df_inequality['avg_dl'],
        ppo_bandwidth_dl=df_bandwidth['avg_dl'],
        ppo_bandlat_dl=df_bandlat['avg_dl'],
        ppo_bandcost_dl=df_bandcost['avg_dl'],
        ppo_costineq_dl=df_costineq['avg_dl'],
        ppo_latineq_dl=df_latineq['avg_dl'],
        ppo_latcost_dl=df_latcost['avg_dl'],

        ppo_cost_gini=df_cost['gini'],
        ppo_latency_gini=df_latency['gini'],
        ppo_inequality_gini=df_inequality['gini'],
        ppo_bandwidth_gini=df_bandwidth['gini'],
        ppo_bandcost_gini=df_bandcost['gini'],
        ppo_bandlat_gini=df_bandlat['gini'],
        ppo_costineq_gini=df_costineq['gini'],
        ppo_latineq_gini=df_latineq['gini'],
        ppo_latcost_gini=df_latcost['gini'],
    )

    plot_stats("nne_training_" + reward, stats, max_reward=max_reward, xlim=xlim, smoothing_window=window)

    # ecdf latency
    fig = plt.figure()
    sns.ecdfplot(data=df_cost['avg_total_latency'], color='#77AC30', label='MaskPPO (Cost)')
    sns.ecdfplot(data=df_latency['avg_total_latency'], color='#D95319', label='MaskPPO (Latency)')
    sns.ecdfplot(data=df_inequality['avg_total_latency'], color='#3399FF', label='MaskPPO (Inequality)')
    sns.ecdfplot(data=df_bandwidth['avg_total_latency'], color='#FFA500', label='MaskPPO (Bandwidth)')
    sns.ecdfplot(data=df_bandcost['avg_total_latency'], color='#EDB120', label='MaskPPO (BandCost)')
    sns.ecdfplot(data=df_bandlat['avg_total_latency'], color='#7A21EE', label='MaskPPO (BandLat)')
    sns.ecdfplot(data=df_costineq['avg_total_latency'], color='#C521EE', label='MaskPPO (CostIneq)')
    sns.ecdfplot(data=df_latineq['avg_total_latency'], color='#D74281', label='MaskPPO (LatIneq)')
    sns.ecdfplot(data=df_latcost['avg_total_latency'], color='#434DD7', label='MaskPPO (LatCost)')

    # sns.ecdfplot(data=df_cost_greedy['avg_total_latency'], color='#E897E8', label='Cost-Greedy')
    # sns.ecdfplot(data=df_bandwidth_greedy['avg_total_latency'], color='#BCCE61', label='Bandwidth-Greedy')
    # sns.ecdfplot(data=df_latency_greedy['avg_total_latency'], color='#DAB9AA', label='Latency-Greedy')

    '''
    sns.ecdfplot(data=df_dqn_cost['avg_latency'], color='#94E827', label='Deepsets DQN (Cost)')
    sns.ecdfplot(data=df_dqn_latency['avg_latency'], color='#F5520C', label='Deepsets DQN (Latency)')
    sns.ecdfplot(data=df_dqn_inequality['avg_latency'], color='#0481FD', label='Deepsets DQN (Inequality)')
    '''
    plt.xlabel("Latency (in ms)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend()

    plt.savefig('cdf_seaborn_total_latency.pdf', dpi=250, bbox_inches='tight')

    # ecdf access latency
    fig = plt.figure()
    sns.ecdfplot(data=df_cost['avg_access_latency'], color='#77AC30', label='MaskPPO (Cost)')
    sns.ecdfplot(data=df_latency['avg_access_latency'], color='#D95319', label='MaskPPO (Latency)')
    sns.ecdfplot(data=df_inequality['avg_access_latency'], color='#3399FF', label='MaskPPO (Inequality)')
    sns.ecdfplot(data=df_bandwidth['avg_access_latency'], color='#FFA500', label='MaskPPO (Bandwidth)')
    sns.ecdfplot(data=df_bandcost['avg_access_latency'], color='#EDB120', label='MaskPPO (BandCost)')
    sns.ecdfplot(data=df_bandlat['avg_access_latency'], color='#7A21EE', label='MaskPPO (BandLat)')
    sns.ecdfplot(data=df_costineq['avg_access_latency'], color='#C521EE', label='MaskPPO (CostIneq)')
    sns.ecdfplot(data=df_latineq['avg_access_latency'], color='#D74281', label='MaskPPO (LatIneq)')
    sns.ecdfplot(data=df_latcost['avg_access_latency'], color='#434DD7', label='MaskPPO (LatCost)')

    # sns.ecdfplot(data=df_cost_greedy['avg_access_latency'], color='#E897E8', label='Cost-Greedy')
    # sns.ecdfplot(data=df_bandwidth_greedy['avg_access_latency'], color='#BCCE61', label='Bandwidth-Greedy')
    # sns.ecdfplot(data=df_latency_greedy['avg_access_latency'], color='#DAB9AA', label='Latency-Greedy')

    '''
    sns.ecdfplot(data=df_dqn_cost['avg_latency'], color='#94E827', label='Deepsets DQN (Cost)')
    sns.ecdfplot(data=df_dqn_latency['avg_latency'], color='#F5520C', label='Deepsets DQN (Latency)')
    sns.ecdfplot(data=df_dqn_inequality['avg_latency'], color='#0481FD', label='Deepsets DQN (Inequality)')
    '''
    plt.xlabel("Latency (in ms)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend()

    plt.savefig('cdf_seaborn_access_latency.pdf', dpi=250, bbox_inches='tight')

    # ecdf rtt
    fig = plt.figure()
    sns.ecdfplot(data=df_cost['avg_rtt'], color='#77AC30', label='MaskPPO (Cost)')
    sns.ecdfplot(data=df_latency['avg_rtt'], color='#D95319', label='MaskPPO (Latency)')
    sns.ecdfplot(data=df_inequality['avg_rtt'], color='#3399FF', label='MaskPPO (Inequality)')
    sns.ecdfplot(data=df_bandwidth['avg_rtt'], color='#FFA500', label='MaskPPO (Bandwidth)')
    sns.ecdfplot(data=df_bandcost['avg_rtt'], color='#EDB120', label='MaskPPO (BandCost)')
    sns.ecdfplot(data=df_bandlat['avg_rtt'], color='#7A21EE', label='MaskPPO (BandLat)')
    sns.ecdfplot(data=df_costineq['avg_rtt'], color='#C521EE', label='MaskPPO (CostIneq)')
    sns.ecdfplot(data=df_latineq['avg_rtt'], color='#D74281', label='MaskPPO (LatIneq)')
    sns.ecdfplot(data=df_latcost['avg_rtt'], color='#434DD7', label='MaskPPO (LatCost)')

    # sns.ecdfplot(data=df_cost_greedy['avg_rtt'], color='#E897E8', label='Cost-Greedy')
    # sns.ecdfplot(data=df_bandwidth_greedy['avg_rtt'], color='#BCCE61', label='Bandwidth-Greedy')
    # sns.ecdfplot(data=df_latency_greedy['avg_rtt'], color='#DAB9AA', label='Latency-Greedy')

    '''
    sns.ecdfplot(data=df_dqn_cost['avg_latency'], color='#94E827', label='Deepsets DQN (Cost)')
    sns.ecdfplot(data=df_dqn_latency['avg_latency'], color='#F5520C', label='Deepsets DQN (Latency)')
    sns.ecdfplot(data=df_dqn_inequality['avg_latency'], color='#0481FD', label='Deepsets DQN (Inequality)')
    '''
    plt.xlabel("RTT (in ms)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend()

    plt.savefig('cdf_seaborn_rtt.pdf', dpi=250, bbox_inches='tight')

    # ecdf avg_processing_latency
    fig = plt.figure()
    sns.ecdfplot(data=df_cost['avg_processing_latency'], color='#77AC30', label='MaskPPO (Cost)')
    sns.ecdfplot(data=df_latency['avg_processing_latency'], color='#D95319', label='MaskPPO (Latency)')
    sns.ecdfplot(data=df_inequality['avg_processing_latency'], color='#3399FF', label='MaskPPO (Inequality)')
    sns.ecdfplot(data=df_bandwidth['avg_processing_latency'], color='#FFA500', label='MaskPPO (Bandwidth)')
    sns.ecdfplot(data=df_bandcost['avg_processing_latency'], color='#EDB120', label='MaskPPO (BandCost)')
    sns.ecdfplot(data=df_bandlat['avg_processing_latency'], color='#7A21EE', label='MaskPPO (BandLat)')
    sns.ecdfplot(data=df_costineq['avg_processing_latency'], color='#C521EE', label='MaskPPO (CostIneq)')
    sns.ecdfplot(data=df_latineq['avg_processing_latency'], color='#D74281', label='MaskPPO (LatIneq)')
    sns.ecdfplot(data=df_latcost['avg_processing_latency'], color='#434DD7', label='MaskPPO (LatCost)')

    # sns.ecdfplot(data=df_cost_greedy['avg_processing_latency'], color='#E897E8', label='Cost-Greedy')
    # sns.ecdfplot(data=df_bandwidth_greedy['avg_processing_latency'], color='#BCCE61', label='Bandwidth-Greedy')
    # sns.ecdfplot(data=df_latency_greedy['avg_processing_latency'], color='#DAB9AA', label='Latency-Greedy')

    '''
    sns.ecdfplot(data=df_dqn_cost['avg_latency'], color='#94E827', label='Deepsets DQN (Cost)')
    sns.ecdfplot(data=df_dqn_latency['avg_latency'], color='#F5520C', label='Deepsets DQN (Latency)')
    sns.ecdfplot(data=df_dqn_inequality['avg_latency'], color='#0481FD', label='Deepsets DQN (Inequality)')
    '''

    plt.xlabel("Latency (in ms)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend()

    plt.savefig('cdf_seaborn_avg_processing_latency.pdf', dpi=250, bbox_inches='tight')

    # ecdf cost
    fig = plt.figure()
    sns.ecdfplot(data=df_cost['avg_deployment_cost'], color='#77AC30', label='MaskPPO (Cost)')
    sns.ecdfplot(data=df_latency['avg_deployment_cost'], color='#D95319', label='MaskPPO (Latency)')
    sns.ecdfplot(data=df_inequality['avg_deployment_cost'], color='#3399FF', label='MaskPPO (Inequality)')
    sns.ecdfplot(data=df_bandwidth['avg_deployment_cost'], color='#FFA500', label='MaskPPO (Bandwidth)')
    sns.ecdfplot(data=df_bandcost['avg_deployment_cost'], color='#EDB120', label='MaskPPO (BandCost)')
    sns.ecdfplot(data=df_bandlat['avg_deployment_cost'], color='#7A21EE', label='MaskPPO (BandLat)')
    sns.ecdfplot(data=df_costineq['avg_deployment_cost'], color='#C521EE', label='MaskPPO (CostIneq)')
    sns.ecdfplot(data=df_latineq['avg_deployment_cost'], color='#D74281', label='MaskPPO (LatIneq)')
    sns.ecdfplot(data=df_latcost['avg_deployment_cost'], color='#434DD7', label='MaskPPO (LatCost)')

    # sns.ecdfplot(data=df_cost_greedy['avg_deployment_cost'], color='#E897E8', label='Cost-Greedy')
    # sns.ecdfplot(data=df_bandwidth_greedy['avg_deployment_cost'], color='#BCCE61', label='Bandwidth-Greedy')
    # sns.ecdfplot(data=df_latency_greedy['avg_deployment_cost'], color='#DAB9AA', label='Latency-Greedy')

    plt.xlabel("Avg. Deployment Cost (in units)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend()

    plt.savefig('cdf_seaborn_cost.pdf', dpi=250, bbox_inches='tight')

    # ecdf ul
    fig = plt.figure()
    sns.ecdfplot(data=df_cost['avg_ul'], color='#77AC30', label='MaskPPO (Cost)')
    sns.ecdfplot(data=df_latency['avg_ul'], color='#D95319', label='MaskPPO (Latency)')
    sns.ecdfplot(data=df_inequality['avg_ul'], color='#3399FF', label='MaskPPO (Inequality)')
    sns.ecdfplot(data=df_bandwidth['avg_ul'], color='#FFA500', label='MaskPPO (Bandwidth)')
    sns.ecdfplot(data=df_bandcost['avg_ul'], color='#EDB120', label='MaskPPO (BandCost)')
    sns.ecdfplot(data=df_bandlat['avg_ul'], color='#7A21EE', label='MaskPPO (BandLat)')
    sns.ecdfplot(data=df_costineq['avg_ul'], color='#C521EE', label='MaskPPO (CostIneq)')
    sns.ecdfplot(data=df_latineq['avg_ul'], color='#D74281', label='MaskPPO (LatIneq)')
    sns.ecdfplot(data=df_latcost['avg_ul'], color='#434DD7', label='MaskPPO (LatCost)')

    # sns.ecdfplot(data=df_cost_greedy['avg_ul'], color='#E897E8', label='Cost-Greedy')
    # sns.ecdfplot(data=df_bandwidth_greedy['avg_ul'], color='#BCCE61', label='Bandwidth-Greedy')
    # sns.ecdfplot(data=df_latency_greedy['avg_ul'], color='#DAB9AA', label='Latency-Greedy')

    plt.xlabel("Avg. UL (in Mbits/s)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend()

    plt.savefig('cdf_seaborn_ul.pdf', dpi=250, bbox_inches='tight')

    # ecdf dl
    fig = plt.figure()
    sns.ecdfplot(data=df_cost['avg_dl'], color='#77AC30', label='MaskPPO (Cost)')
    sns.ecdfplot(data=df_latency['avg_dl'], color='#D95319', label='MaskPPO (Latency)')
    sns.ecdfplot(data=df_inequality['avg_dl'], color='#3399FF', label='MaskPPO (Inequality)')
    sns.ecdfplot(data=df_bandwidth['avg_dl'], color='#FFA500', label='MaskPPO (Bandwidth)')
    sns.ecdfplot(data=df_bandcost['avg_dl'], color='#EDB120', label='MaskPPO (BandCost)')
    sns.ecdfplot(data=df_bandlat['avg_dl'], color='#7A21EE', label='MaskPPO (BandLat)')
    sns.ecdfplot(data=df_costineq['avg_dl'], color='#C521EE', label='MaskPPO (CostIneq)')
    sns.ecdfplot(data=df_latineq['avg_dl'], color='#D74281', label='MaskPPO (LatIneq)')
    sns.ecdfplot(data=df_latcost['avg_dl'], color='#434DD7', label='MaskPPO (LatCost)')

    # sns.ecdfplot(data=df_cost_greedy['avg_dl'], color='#E897E8', label='Cost-Greedy')
    # sns.ecdfplot(data=df_bandwidth_greedy['avg_dl'], color='#BCCE61', label='Bandwidth-Greedy')
    # sns.ecdfplot(data=df_latency_greedy['avg_dl'], color='#DAB9AA', label='Latency-Greedy')

    plt.xlabel("Avg. DL (in Mbits/s)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend()

    plt.savefig('cdf_seaborn_dl.pdf', dpi=250, bbox_inches='tight')

    # ecdf jitter
    fig = plt.figure()
    sns.ecdfplot(data=df_cost['avg_jitter'], color='#77AC30', label='MaskPPO (Cost)')
    sns.ecdfplot(data=df_latency['avg_jitter'], color='#D95319', label='MaskPPO (Latency)')
    sns.ecdfplot(data=df_inequality['avg_jitter'], color='#3399FF', label='MaskPPO (Inequality)')
    sns.ecdfplot(data=df_bandwidth['avg_jitter'], color='#FFA500', label='MaskPPO (Bandwidth)')
    sns.ecdfplot(data=df_bandcost['avg_jitter'], color='#EDB120', label='MaskPPO (BandCost)')
    sns.ecdfplot(data=df_bandlat['avg_jitter'], color='#7A21EE', label='MaskPPO (BandLat)')
    sns.ecdfplot(data=df_costineq['avg_jitter'], color='#C521EE', label='MaskPPO (CostIneq)')
    sns.ecdfplot(data=df_latineq['avg_jitter'], color='#D74281', label='MaskPPO (LatIneq)')
    sns.ecdfplot(data=df_latcost['avg_jitter'], color='#434DD7', label='MaskPPO (LatCost)')

    # sns.ecdfplot(data=df_cost_greedy['avg_jitter'], color='#E897E8', label='Cost-Greedy')
    # sns.ecdfplot(data=df_bandwidth_greedy['avg_jitter'], color='#BCCE61', label='Bandwidth-Greedy')
    # sns.ecdfplot(data=df_latency_greedy['avg_jitter'], color='#DAB9AA', label='Latency-Greedy')

    plt.xlabel("Avg. DL (in Mbits/s)")
    plt.ylabel("Jitter (in ms)")
    plt.legend()

    plt.savefig('cdf_seaborn_jitter.pdf', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    width = 0.4
    red_square = dict(markerfacecolor='r', marker='s')

    ticks = ['MaskPPO (Cost)', 'MaskPPO (Latency)', 'MaskPPO (Inequality)', 'MaskPPO (Bandwidth)',
             'Cost-Greedy', 'Bandwidth-Greedy', 'Latency-Greedy']

    data_ppo_cost = [df_cost['avg_deployment_cost'].tolist()]
    data_ppo_latency = [df_latency['avg_deployment_cost'].tolist()]
    data_ppo_inequality = [df_inequality['avg_deployment_cost'].tolist()]
    data_ppo_bandwidth = [df_bandwidth['avg_deployment_cost'].tolist()]
    data_cost_greedy = [df_cost_greedy['avg_deployment_cost'].tolist()]
    data_bandwidth_greedy = [df_bandwidth_greedy['avg_deployment_cost'].tolist()]
    data_latency_greedy = [df_latency_greedy['avg_deployment_cost'].tolist()]

    a = plt.boxplot(data_ppo_cost, positions=[1], widths=width, flierprops=red_square)
    b = plt.boxplot(data_ppo_latency, positions=[5], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    c = plt.boxplot(data_ppo_inequality, positions=[10], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    d = plt.boxplot(data_ppo_bandwidth, positions=[15], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)
    e = plt.boxplot(data_cost_greedy, positions=[20], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    f = plt.boxplot(data_bandwidth_greedy, positions=[25], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    g = plt.boxplot(data_latency_greedy, positions=[30], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)

    # h = plt.boxplot(data_binpack_greedy, positions=[35], whiskerprops=dict(ls='-.'), widths=width,
    #                flierprops=red_square)
    # i = plt.boxplot(data_latency_greedy, positions=[40], whiskerprops=dict(ls='--'), widths=width,
    #                flierprops=red_square)
    # j = plt.boxplot(data_karmada_greedy, positions=[45], whiskerprops=dict(ls='dotted'), widths=width,
    #                flierprops=redsquare)

    set_box_color(a, '#77AC30')
    set_box_color(b, '#D95319')
    set_box_color(c, '#3399FF')
    set_box_color(d, '#FFA500')
    set_box_color(e, '#E897E8')
    set_box_color(f, '#808080')
    set_box_color(g, '#DAB9AA')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#77AC30', label='MaskPPO (Cost)')
    plt.plot([], c='#D95319', ls='-.', label='MaskPPO (Latency)')
    plt.plot([], c='#3399FF', ls='--', label='MaskPPO (Inequality)')
    plt.plot([], c='#FFA500', ls='dotted', label='MaskPPO (Bandwidth)')
    plt.plot([], c='#E897E8', ls='-.', label='Cost-Greedy')
    plt.plot([], c='#808080', ls='--', label='Bandwidth-Greedy')
    plt.plot([], c='#DAB9AA', ls='dotted', label='Latency-Greedy')
    # plt.plot([], c='#BCCE61', ls='-.', label='Binpack-Greedy')
    # plt.plot([], c='#DAB9AA', ls='--', label='Latency-Greedy')
    # plt.plot([], c='#221F1E', ls='dotted', label='Karmada-Greedy')

    plt.xticks([1, 5, 10, 15, 20, 25, 30], ticks, fontsize=4)
    # plt.xlim(0, 80)
    plt.ylim(0, 24)

    plt.xlabel("Evaluated Strategies")
    plt.ylabel("Deployment Cost (in units)")
    plt.legend(ncols=2)

    plt.savefig('box_plot_cost.pdf', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    width = 0.4
    red_square = dict(markerfacecolor='r', marker='s')

    ticks = ['MaskPPO (Cost)', 'MaskPPO (Latency)', 'MaskPPO (Inequality)', 'MaskPPO (Bandwidth)',
             'Cost-Greedy', 'Bandwidth-Greedy', 'Latency-Greedy']

    data_ppo_cost = [df_cost['avg_dl'].tolist()]
    data_ppo_latency = [df_latency['avg_dl'].tolist()]
    data_ppo_inequality = [df_inequality['avg_dl'].tolist()]
    data_ppo_bandwidth = [df_bandwidth['avg_dl'].tolist()]
    data_cost_greedy = [df_cost_greedy['avg_dl'].tolist()]
    data_bandwidth_greedy = [df_bandwidth_greedy['avg_dl'].tolist()]
    data_latency_greedy = [df_latency_greedy['avg_dl'].tolist()]

    a = plt.boxplot(data_ppo_cost, positions=[1], widths=width, flierprops=red_square)
    b = plt.boxplot(data_ppo_latency, positions=[5], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    c = plt.boxplot(data_ppo_inequality, positions=[10], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    d = plt.boxplot(data_ppo_bandwidth, positions=[15], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)
    e = plt.boxplot(data_cost_greedy, positions=[20], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    f = plt.boxplot(data_bandwidth_greedy, positions=[25], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    g = plt.boxplot(data_latency_greedy, positions=[30], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)

    # h = plt.boxplot(data_binpack_greedy, positions=[35], whiskerprops=dict(ls='-.'), widths=width,
    #                flierprops=red_square)
    # i = plt.boxplot(data_latency_greedy, positions=[40], whiskerprops=dict(ls='--'), widths=width,
    #                flierprops=red_square)
    # j = plt.boxplot(data_karmada_greedy, positions=[45], whiskerprops=dict(ls='dotted'), widths=width,
    #                flierprops=redsquare)

    set_box_color(a, '#77AC30')
    set_box_color(b, '#D95319')
    set_box_color(c, '#3399FF')
    set_box_color(d, '#FFA500')
    set_box_color(e, '#E897E8')
    set_box_color(f, '#808080')
    set_box_color(g, '#DAB9AA')
    # set_box_color(h, '#BCCE61')
    # set_box_color(i, '#DAB9AA')
    # set_box_color(j, '#221F1E')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#77AC30', label='MaskPPO (Cost)')
    plt.plot([], c='#D95319', ls='-.', label='MaskPPO (Latency)')
    plt.plot([], c='#3399FF', ls='--', label='MaskPPO (Inequality)')
    plt.plot([], c='#FFA500', ls='dotted', label='MaskPPO (Bandwidth)')
    plt.plot([], c='#E897E8', ls='-.', label='Cost-Greedy')
    plt.plot([], c='#808080', ls='--', label='Bandwidth-Greedy')
    plt.plot([], c='#DAB9AA', ls='dotted', label='Latency-Greedy')
    # plt.plot([], c='#BCCE61', ls='-.', label='Binpack-Greedy')
    # plt.plot([], c='#DAB9AA', ls='--', label='Latency-Greedy')
    # plt.plot([], c='#221F1E', ls='dotted', label='Karmada-Greedy')

    plt.xticks([1, 5, 10, 15, 20, 25, 30], ticks, fontsize=4)
    # plt.xlim(0, 80)
    plt.ylim(0, 480)

    plt.xlabel("Evaluated Strategies")
    plt.ylabel("Avg. DL (in Mbits/s)")
    plt.legend(ncols=2)

    plt.savefig('box_plot_dl.pdf', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    width = 0.4
    red_square = dict(markerfacecolor='r', marker='s')

    ticks = ['MaskPPO (Cost)', 'MaskPPO (Latency)', 'MaskPPO (Inequality)', 'MaskPPO (Bandwidth)',
             'Cost-Greedy', 'Bandwidth-Greedy', 'Latency-Greedy']

    data_ppo_cost = [df_cost['avg_ul'].tolist()]
    data_ppo_latency = [df_latency['avg_ul'].tolist()]
    data_ppo_inequality = [df_inequality['avg_ul'].tolist()]
    data_ppo_bandwidth = [df_bandwidth['avg_ul'].tolist()]
    data_cost_greedy = [df_cost_greedy['avg_ul'].tolist()]
    data_bandwidth_greedy = [df_bandwidth_greedy['avg_ul'].tolist()]
    data_latency_greedy = [df_latency_greedy['avg_ul'].tolist()]

    a = plt.boxplot(data_ppo_cost, positions=[1], widths=width, flierprops=red_square)
    b = plt.boxplot(data_ppo_latency, positions=[5], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    c = plt.boxplot(data_ppo_inequality, positions=[10], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    d = plt.boxplot(data_ppo_bandwidth, positions=[15], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)
    e = plt.boxplot(data_cost_greedy, positions=[20], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    f = plt.boxplot(data_bandwidth_greedy, positions=[25], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    g = plt.boxplot(data_latency_greedy, positions=[30], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)

    # h = plt.boxplot(data_binpack_greedy, positions=[35], whiskerprops=dict(ls='-.'), widths=width,
    #                flierprops=red_square)
    # i = plt.boxplot(data_latency_greedy, positions=[40], whiskerprops=dict(ls='--'), widths=width,
    #                flierprops=red_square)
    # j = plt.boxplot(data_karmada_greedy, positions=[45], whiskerprops=dict(ls='dotted'), widths=width,
    #                flierprops=redsquare)

    set_box_color(a, '#77AC30')
    set_box_color(b, '#D95319')
    set_box_color(c, '#3399FF')
    set_box_color(d, '#FFA500')
    set_box_color(e, '#E897E8')
    set_box_color(f, '#808080')
    set_box_color(g, '#DAB9AA')
    # set_box_color(h, '#BCCE61')
    # set_box_color(i, '#DAB9AA')
    # set_box_color(j, '#221F1E')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#77AC30', label='MaskPPO (Cost)')
    plt.plot([], c='#D95319', ls='-.', label='MaskPPO (Latency)')
    plt.plot([], c='#3399FF', ls='--', label='MaskPPO (Inequality)')
    plt.plot([], c='#94E827', ls='dotted', label='MaskPPO (Bandwidth)')
    plt.plot([], c='#E897E8', ls='-.', label='Cost-Greedy')
    plt.plot([], c='#808080', ls='--', label='Bandwidth-Greedy')
    plt.plot([], c='#DAB9AA', ls='dotted', label='Latency-Greedy')
    # plt.plot([], c='#BCCE61', ls='-.', label='Binpack-Greedy')
    # plt.plot([], c='#DAB9AA', ls='--', label='Latency-Greedy')
    # plt.plot([], c='#221F1E', ls='dotted', label='Karmada-Greedy')

    plt.xticks([1, 5, 10, 15, 20, 25, 30], ticks, fontsize=4)
    # plt.xlim(0, 80)
    plt.ylim(0, 70)

    plt.xlabel("Evaluated Strategies")
    plt.ylabel("Avg. UL (in Mbits/s)")
    plt.legend(ncols=2)

    plt.savefig('box_plot_ul.pdf', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    width = 0.4
    red_square = dict(markerfacecolor='r', marker='s')

    ticks = ['MaskPPO (Cost)', 'MaskPPO (Latency)', 'MaskPPO (Inequality)', 'MaskPPO (Bandwidth)',
             'Cost-Greedy', 'Bandwidth-Greedy', 'Latency-Greedy']

    data_ppo_cost = [df_cost['avg_total_latency'].tolist()]
    data_ppo_latency = [df_latency['avg_total_latency'].tolist()]
    data_ppo_inequality = [df_inequality['avg_total_latency'].tolist()]
    data_ppo_bandwidth = [df_bandwidth['avg_total_latency'].tolist()]
    data_cost_greedy = [df_cost_greedy['avg_total_latency'].tolist()]
    data_bandwidth_greedy = [df_bandwidth_greedy['avg_total_latency'].tolist()]
    data_latency_greedy = [df_latency_greedy['avg_total_latency'].tolist()]

    a = plt.boxplot(data_ppo_cost, positions=[1], widths=width, flierprops=red_square)
    b = plt.boxplot(data_ppo_latency, positions=[5], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    c = plt.boxplot(data_ppo_inequality, positions=[10], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    d = plt.boxplot(data_ppo_bandwidth, positions=[15], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)
    e = plt.boxplot(data_cost_greedy, positions=[20], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    f = plt.boxplot(data_bandwidth_greedy, positions=[25], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    g = plt.boxplot(data_latency_greedy, positions=[30], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)

    # h = plt.boxplot(data_binpack_greedy, positions=[35], whiskerprops=dict(ls='-.'), widths=width,
    #                flierprops=red_square)
    # i = plt.boxplot(data_latency_greedy, positions=[40], whiskerprops=dict(ls='--'), widths=width,
    #                flierprops=red_square)
    # j = plt.boxplot(data_karmada_greedy, positions=[45], whiskerprops=dict(ls='dotted'), widths=width,
    #                flierprops=redsquare)

    set_box_color(a, '#77AC30')
    set_box_color(b, '#D95319')
    set_box_color(c, '#3399FF')
    set_box_color(d, '#FFA500')
    set_box_color(e, '#E897E8')
    set_box_color(f, '#808080')
    set_box_color(g, '#DAB9AA')
    # set_box_color(h, '#BCCE61')
    # set_box_color(i, '#DAB9AA')
    # set_box_color(j, '#221F1E')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#77AC30', label='MaskPPO (Cost)')
    plt.plot([], c='#D95319', ls='-.', label='MaskPPO (Latency)')
    plt.plot([], c='#3399FF', ls='--', label='MaskPPO (Inequality)')
    plt.plot([], c='#94E827', ls='dotted', label='MaskPPO (Bandwidth)')
    plt.plot([], c='#E897E8', ls='-.', label='Cost-Greedy')
    plt.plot([], c='#808080', ls='--', label='Bandwidth-Greedy')
    plt.plot([], c='#DAB9AA', ls='dotted', label='Latency-Greedy')
    # plt.plot([], c='#BCCE61', ls='-.', label='Binpack-Greedy')
    # plt.plot([], c='#DAB9AA', ls='--', label='Latency-Greedy')
    # plt.plot([], c='#221F1E', ls='dotted', label='Karmada-Greedy')

    plt.xticks([1, 5, 10, 15, 20, 25, 30], ticks, fontsize=4)
    # plt.xlim(0, 80)
    plt.ylim(0, 160)

    plt.xlabel("Evaluated Strategies")
    plt.ylabel("Total Latency (in ms)")
    plt.legend(ncols=2)

    plt.savefig('box_plot_total_latency.pdf', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    width = 0.4
    red_square = dict(markerfacecolor='r', marker='s')

    ticks = ['MaskPPO (Cost)', 'MaskPPO (Latency)', 'MaskPPO (Inequality)', 'MaskPPO (Bandwidth)',
             'Cost-Greedy', 'Bandwidth-Greedy', 'Latency-Greedy']

    data_ppo_cost = [df_cost['avg_rtt'].tolist()]
    data_ppo_latency = [df_latency['avg_rtt'].tolist()]
    data_ppo_inequality = [df_inequality['avg_rtt'].tolist()]
    data_ppo_bandwidth = [df_bandwidth['avg_rtt'].tolist()]
    data_cost_greedy = [df_cost_greedy['avg_rtt'].tolist()]
    data_bandwidth_greedy = [df_bandwidth_greedy['avg_rtt'].tolist()]
    data_latency_greedy = [df_latency_greedy['avg_rtt'].tolist()]

    a = plt.boxplot(data_ppo_cost, positions=[1], widths=width, flierprops=red_square)
    b = plt.boxplot(data_ppo_latency, positions=[5], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    c = plt.boxplot(data_ppo_inequality, positions=[10], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    d = plt.boxplot(data_ppo_bandwidth, positions=[15], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)
    e = plt.boxplot(data_cost_greedy, positions=[20], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    f = plt.boxplot(data_bandwidth_greedy, positions=[25], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    g = plt.boxplot(data_latency_greedy, positions=[30], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)

    # h = plt.boxplot(data_binpack_greedy, positions=[35], whiskerprops=dict(ls='-.'), widths=width,
    #                flierprops=red_square)
    # i = plt.boxplot(data_latency_greedy, positions=[40], whiskerprops=dict(ls='--'), widths=width,
    #                flierprops=red_square)
    # j = plt.boxplot(data_karmada_greedy, positions=[45], whiskerprops=dict(ls='dotted'), widths=width,
    #                flierprops=redsquare)

    set_box_color(a, '#77AC30')
    set_box_color(b, '#D95319')
    set_box_color(c, '#3399FF')
    set_box_color(d, '#FFA500')
    set_box_color(e, '#E897E8')
    set_box_color(f, '#808080')
    set_box_color(g, '#DAB9AA')
    # set_box_color(h, '#BCCE61')
    # set_box_color(i, '#DAB9AA')
    # set_box_color(j, '#221F1E')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#77AC30', label='MaskPPO (Cost)')
    plt.plot([], c='#D95319', ls='-.', label='MaskPPO (Latency)')
    plt.plot([], c='#3399FF', ls='--', label='MaskPPO (Inequality)')
    plt.plot([], c='#FFA500', ls='dotted', label='MaskPPO (Bandwidth)')
    plt.plot([], c='#E897E8', ls='-.', label='Cost-Greedy')
    plt.plot([], c='#808080', ls='--', label='Bandwidth-Greedy')
    plt.plot([], c='#DAB9AA', ls='dotted', label='Latency-Greedy')
    # plt.plot([], c='#BCCE61', ls='-.', label='Binpack-Greedy')
    # plt.plot([], c='#DAB9AA', ls='--', label='Latency-Greedy')
    # plt.plot([], c='#221F1E', ls='dotted', label='Karmada-Greedy')

    plt.xticks([1, 5, 10, 15, 20, 25, 30], ticks, fontsize=4)
    # plt.xlim(0, 80)
    plt.ylim(0, 0.20)

    plt.xlabel("Evaluated Strategies")
    plt.ylabel("RTT (in ms)")
    plt.legend(ncols=2)

    plt.savefig('box_plot_rtt.pdf', dpi=250, bbox_inches='tight')

    # proc latency
    fig = plt.figure()
    width = 0.4
    red_square = dict(markerfacecolor='r', marker='s')

    ticks = ['MaskPPO (Cost)', 'MaskPPO (Latency)', 'MaskPPO (Inequality)', 'MaskPPO (Bandwidth)',
             'Cost-Greedy', 'Bandwidth-Greedy', 'Latency-Greedy']

    data_ppo_cost = [df_cost['avg_processing_latency'].tolist()]
    data_ppo_latency = [df_latency['avg_processing_latency'].tolist()]
    data_ppo_inequality = [df_inequality['avg_processing_latency'].tolist()]
    data_ppo_bandwidth = [df_bandwidth['avg_processing_latency'].tolist()]
    data_cost_greedy = [df_cost_greedy['avg_processing_latency'].tolist()]
    data_bandwidth_greedy = [df_bandwidth_greedy['avg_processing_latency'].tolist()]
    data_latency_greedy = [df_latency_greedy['avg_processing_latency'].tolist()]

    a = plt.boxplot(data_ppo_cost, positions=[1], widths=width, flierprops=red_square)
    b = plt.boxplot(data_ppo_latency, positions=[5], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    c = plt.boxplot(data_ppo_inequality, positions=[10], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    d = plt.boxplot(data_ppo_bandwidth, positions=[15], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)
    e = plt.boxplot(data_cost_greedy, positions=[20], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    f = plt.boxplot(data_bandwidth_greedy, positions=[25], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    g = plt.boxplot(data_latency_greedy, positions=[30], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)

    # h = plt.boxplot(data_binpack_greedy, positions=[35], whiskerprops=dict(ls='-.'), widths=width,
    #                flierprops=red_square)
    # i = plt.boxplot(data_latency_greedy, positions=[40], whiskerprops=dict(ls='--'), widths=width,
    #                flierprops=red_square)
    # j = plt.boxplot(data_karmada_greedy, positions=[45], whiskerprops=dict(ls='dotted'), widths=width,
    #                flierprops=redsquare)

    set_box_color(a, '#77AC30')
    set_box_color(b, '#D95319')
    set_box_color(c, '#3399FF')
    set_box_color(d, '#FFA500')
    set_box_color(e, '#E897E8')
    set_box_color(f, '#808080')
    set_box_color(g, '#DAB9AA')
    # set_box_color(h, '#BCCE61')
    # set_box_color(i, '#DAB9AA')
    # set_box_color(j, '#221F1E')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#77AC30', label='MaskPPO (Cost)')
    plt.plot([], c='#D95319', ls='-.', label='MaskPPO (Latency)')
    plt.plot([], c='#3399FF', ls='--', label='MaskPPO (Inequality)')
    plt.plot([], c='#FFA500', ls='dotted', label='MaskPPO (Bandwidth)')
    plt.plot([], c='#E897E8', ls='-.', label='Cost-Greedy')
    plt.plot([], c='#808080', ls='--', label='Bandwidth-Greedy')
    plt.plot([], c='#DAB9AA', ls='dotted', label='Latency-Greedy')
    # plt.plot([], c='#BCCE61', ls='-.', label='Binpack-Greedy')
    # plt.plot([], c='#DAB9AA', ls='--', label='Latency-Greedy')
    # plt.plot([], c='#221F1E', ls='dotted', label='Karmada-Greedy')

    plt.xticks([1, 5, 10, 15, 20, 25, 30], ticks, fontsize=4)
    # plt.xlim(0, 80)
    plt.ylim(0, 140)

    plt.xlabel("Evaluated Strategies")
    plt.ylabel("Processing Latency (in ms)")
    plt.legend(ncols=2)

    plt.savefig('box_plot_proc_latency.pdf', dpi=250, bbox_inches='tight')

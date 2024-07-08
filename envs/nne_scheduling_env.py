import csv
import glob
import math
import operator
import os
from datetime import datetime, timedelta
import heapq
import time
import random
from statistics import mean
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from envs.utils import DeploymentRequest, get_c2e_deployment_list, save_to_csv, sort_dict_by_value, \
    calculate_gini_coefficient, normalize
import logging

# Actions - for printing purposes
ACTIONS = ["Deploy-Node", "Reject"]

# Reward objectives
# NAIVE Strategy: +1 if agent accepts request, or -1 if rejects it (if resources were available)
NAIVE = 'naive'

# Cost defaults
MAX_COST = 16  # Defined based on the max cost in DEFAULT_CLUSTER_TYPES
MIN_COST = 1  # Defined based on the min cost in DEFAULT_CLUSTER_TYPES

# Multi-objective reward function
MULTI = 'multi'

# Node Types
# Cluster Types
NUM_NODE_TYPES = 5
DEFAULT_NODE_TYPES = [{"type": "edge_tier_1", "cpu": 2.0, "mem": 2.0, "cost": 1, "latency": 1},
                      {"type": "edge_tier_2", "cpu": 2.0, "mem": 4.0, "cost": 2, "latency": 2.5},
                      {"type": "fog_tier_1", "cpu": 2.0, "mem": 8.0, "cost": 4, "latency": 5.0},
                      {"type": "fog_tier_2", "cpu": 4.0, "mem": 16.0, "cost": 8, "latency": 7.5},
                      {"type": "cloud", "cpu": 8.0, "mem": 32.0, "cost": 16, "latency": 10.0}]

# DEFAULT_NODE_TYPES = [{"type": "edge", "cpu": 4.0, "mem": 4.0, "cost": 1},  # pc-engine and celerway
#                     {"type": "fog", "cpu": 8.0, "mem": 16.0, "cost": 4},
#                     {"type": "cloud", "cpu": 12.0, "mem": 32.0, "cost": 8}]

# DEFAULTS for Env configuration
DEFAULT_NUM_EPISODE_STEPS = 100
DEFAULT_NUM_NODES = 4
DEFAULT_ARRIVAL_RATE = 100
DEFAULT_CALL_DURATION = 1
DEFAULT_REWARD_FUNTION = NAIVE
DEFAULT_FILE_NAME_RESULTS = "nne_gym_results"

# Computing metrics: 4 metrics = CPU capacity, memory capacity, CPU allocated, memory allocated
# Dataset metrics: 6 metrics = provider id, interface id, ul_mbps, dl_mbps, jitter, rtt
# Processing latency added = processing_latency

# Other to consider: Latency removed
NUM_METRICS_NODES = 11

# Computing metrics: 3 metrics = cpu_request, memory_request, latency_threshold
# Bandwidth requirements? ul_traffic + dl_traffic?
# sim variables: dt
NUM_METRICS_REQUEST = 4

NUM_PROVIDERS = 3  # Telia, Telenor and Ice
NUM_INTERFACES = 2  # 4G and 5G

PROVIDERS = ["telia", "telenor", "ice"]
INTERFACES = ["4g", "5g"]

# CSV IDs
PROVIDERS_CSV = "network_id"
INTERFACES_CSV = "service_id"

TELIA_CSV = 1
TELENOR_CSV = 2
ICE_CSV = 18
fourG_CSV = 1
fiveG_CSV = 2

TELIA = 0
TELENOR = 1
ICE = 2
four = 1
five = 2

# Defaults
# Adjusted based on min/max values of dataset
MIN_RTT = 0.0  # corresponds to 0.0 ms v1: 0 - 30/v2: 0 - 40
MAX_RTT = 40.0  # corresponds to 30.0 ms

MIN_LATENCY = 1.0  # corresponds to min access latency of node - 1.0
MAX_LATENCY = 10.0  # corresponds to max access latency of node - 10.0

MIN_JITTER = 0.0  # v1: 0 - 513/v2: 0 - 229
MAX_JITTER = 229.0

MIN_UL = 0.0  # v1: 0 - 90/v2: 0 - 93
MAX_UL = 93.0

MIN_DL = 1.0  # v1: 1 - 620/v2: 1 - 516
MAX_DL = 516.0

MIN_PKT_LOSS_RATE = 0.0
MAX_PKT_LOSS_RATE = 1.0

MIN_OBS = 0.0
MAX_OBS = 1000.0

PROCESSING_DELAY = 2.0  # 2.0 ms
MIN_PROC = 0.0
MAX_PROC = 200.0  # 2.0 * 100 steps = 200.0 ms

# Dataframe column names
DF_COLUMN_PKT_LOSS_RATE = "pkt_loss_rate"
DF_COLUMN_RTT_AVG = "rtt_avg"
# DF_COLUMN_RTT_MEDIAN = "rtt_median"
# DF_COLUMN_RTT_STD = "rtt_std"
DF_COLUMN_RTT_Q90 = "rtt_q90"
DF_COLUMN_RSSI = "rssi"
DF_COLUMN_RSRQ = "rsrq"
DF_COLUMN_RSRP = "rsrp"
DF_COLUMN_UL = "speedtest_ul_mbps"
DF_COLUMN_DL = "speedtest_dl_mbps"
DF_COLUMN_LATENCY = "speedtest_latency"
DF_COLUMN_JITTER = "speedtest_jitter"

# Defaults for Weights
LATENCY_WEIGHT = 1.0
GINI_WEIGHT = 0.0
COST_WEIGHT = 0.0
BANDWIDTH_WEIGHT = 0.0

FACTOR = 1.0
SEED = 42
PATH_CSV_FILES = "data/train/v1/nodes/"


class NNESchedulingEnv(gym.Env):
    """ NNE Scheduling env in Kubernetes - an OpenAI gym environment"""
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, num_nodes=DEFAULT_NUM_NODES,
                 arrival_rate_r=DEFAULT_ARRIVAL_RATE,
                 call_duration_r=DEFAULT_CALL_DURATION,
                 episode_length=DEFAULT_NUM_EPISODE_STEPS,
                 reward_function=DEFAULT_REWARD_FUNTION,
                 latency_weight=LATENCY_WEIGHT,
                 gini_weight=GINI_WEIGHT,
                 cost_weight=COST_WEIGHT,
                 bandwidth_weight=BANDWIDTH_WEIGHT,
                 seed=SEED,
                 factor=FACTOR,
                 path_csv_files=PATH_CSV_FILES,
                 file_results_name=DEFAULT_FILE_NAME_RESULTS):

        # Define action and observation space
        super(NNESchedulingEnv, self).__init__()
        self.name = "nne_gym"
        self.__version__ = "0.0.1"
        self.reward_function = reward_function
        self.num_nodes = num_nodes
        self.total_number = num_nodes * NUM_PROVIDERS * NUM_INTERFACES
        self.current_step = 0
        self.default_node_types = DEFAULT_NODE_TYPES

        # Variables for rewards
        self.latency_weight = latency_weight
        self.gini_weight = gini_weight
        self.cost_weight = cost_weight
        self.bandwidth_weight = bandwidth_weight

        self.arrival_rate_r = arrival_rate_r
        self.call_duration_r = call_duration_r
        self.episode_length = episode_length
        self.running_requests: list[DeploymentRequest] = []

        self.avg_rtt = []
        self.avg_ul = []
        self.avg_dl = []
        self.avg_jitter = []
        self.avg_total_latency = []
        self.avg_processing_latency = []
        self.avg_access_latency = []
        self.total_latency = []
        self.avg_deployment_cost = []

        self.avg_load_served_per_provider = np.zeros(NUM_PROVIDERS)
        self.telia_requests = 0
        self.telenor_requests = 0
        self.ice_requests = 0

        # Metrics for different providers/interfaces
        # Telia - 1
        # Telenor - 2
        # Ice - 3
        self.processing_latency = np.zeros(self.total_number)
        self.node_id = np.zeros(self.total_number)
        self.provider_id = np.zeros(self.total_number)
        self.interface_id = np.zeros(self.total_number)
        self.rtt = np.zeros(self.total_number)
        self.ul = np.zeros(self.total_number)
        self.dl = np.zeros(self.total_number)
        # self.latency = np.zeros(self.total_number)
        self.jitter = np.zeros(self.total_number)

        self.seed = seed
        self.np_random, seed = seeding.np_random(self.seed)
        self.factor = factor

        logging.info(
            "[Init] Env: {} | Version {} | Num_Nodes: {} | Total Number: {}".format(self.name, self.__version__,
                                                                                    num_nodes, self.total_number))

        # Defined as a matrix having as rows the nodes and columns their associated metrics
        self.observation_space = spaces.Box(low=MIN_OBS,
                                            high=MAX_OBS,
                                            shape=(self.total_number + 1, NUM_METRICS_NODES + NUM_METRICS_REQUEST),
                                            dtype=np.float32)

        # Action Space
        # deploy the service on node 1 - ID 1, node 1 ID 2,..., n + reject it
        self.num_actions = self.total_number + 1

        # Discrete action space
        self.action_space = spaces.Discrete(self.num_actions)

        # Action and Observation Space
        logging.info("[Init] Action Space: {}".format(self.action_space))
        logging.info("[Init] Observation Space: {}".format(self.observation_space))
        # logging.info("[Init] Observation Space Shape: {}".format(self.observation_space.shape))

        # Setting the experiment based on Cloud2Edge (C2E) deployments
        self.deploymentList = get_c2e_deployment_list()
        self.deployment_request = None

        # New: Resource capacities based on node type
        self.cpu_capacity = np.zeros(self.total_number)
        self.memory_capacity = np.zeros(self.total_number)
        self.node_type = [0] * self.total_number

        logging.info("[Init] Resource Capacities... ")
        j = 0
        for n in range(num_nodes):
            node_type = int(self.np_random.integers(low=0, high=NUM_NODE_TYPES))
            for p in range(NUM_PROVIDERS):
                for i in range(NUM_INTERFACES):
                    self.node_id[j] = n
                    self.provider_id[j] = p
                    self.interface_id[j] = i
                    self.node_type[j] = node_type
                    self.cpu_capacity[j] = DEFAULT_NODE_TYPES[node_type]['cpu']
                    self.memory_capacity[j] = DEFAULT_NODE_TYPES[node_type]['mem']

                    logging.info("[Init] node: {} | provider: {} | interface: {} | Type: {} "
                                 "| cpu: {} | mem: {}".format(n + 1, PROVIDERS[p], INTERFACES[i],
                                                              DEFAULT_NODE_TYPES[node_type]['type'],
                                                              self.cpu_capacity[j],
                                                              self.memory_capacity[j]))
                    j += 1

        # Keeps track of allocated resources
        self.allocated_cpu = np.zeros(self.total_number)
        self.allocated_memory = np.zeros(self.total_number)

        random_cpu = self.np_random.uniform(low=0.0, high=0.2, size=num_nodes)
        random_memory = self.np_random.uniform(low=0.0, high=0.2, size=num_nodes)
        j = 0
        for n in range(num_nodes):
            for p in range(NUM_PROVIDERS):
                for i in range(NUM_INTERFACES):
                    self.allocated_cpu[j] = random_cpu[n]
                    self.allocated_memory[j] = random_memory[n]
                    j += 1

        # Keeps track of Free resources for deployment requests
        self.free_cpu = np.zeros(self.total_number)
        self.free_memory = np.zeros(self.total_number)

        # CSV files for each node
        self.path_csv_files = path_csv_files
        self.node_csv_data = []
        self.df_node = []
        self.action_valid = []

        self.df_node_selected_rows = []
        self.selected_ts = None

        j = 0
        file = ""
        for n in range(num_nodes):
            # Choose a random CSV file for each node
            if os.path.exists(self.path_csv_files):
                file = random.choice(os.listdir(self.path_csv_files))
                logging.info("[Init] FileName: {}".format(file))

            for p in range(NUM_PROVIDERS):
                for i in range(NUM_INTERFACES):
                    self.free_cpu[j] = self.cpu_capacity[j] - self.allocated_cpu[j]
                    self.free_memory[j] = self.memory_capacity[j] - self.allocated_memory[j]

                    # Update files for each node
                    self.node_csv_data.append(self.path_csv_files + file)
                    self.df_node.append(pd.read_csv(self.node_csv_data[j]))

                    # Check if provider and interface exist in dataframe
                    if p == TELIA:
                        id_provider = TELIA_CSV
                    elif p == TELENOR:
                        id_provider = TELENOR_CSV
                    else:
                        id_provider = ICE_CSV

                    if i == fourG_CSV:
                        id_interface = four
                    else:
                        id_interface = five

                    # Select rows based on provider and interface
                    selected_rows = self.df_node[j][(self.df_node[j][PROVIDERS_CSV] == id_provider) &
                                                    (self.df_node[j][INTERFACES_CSV] == id_interface)]

                    # Reset index of the DataFrame
                    # selected_rows.reset_index(drop=True, inplace=True)

                    # print(selected_rows)
                    self.df_node_selected_rows.append(selected_rows)

                    # logging.info("[Init] Node: {} | Provider: {} | Interface: {} | size of rows: {}".format(n + 1,
                    #   PROVIDERS[p],INTERFACES[i], len(selected_rows)))

                    # If len(rows) = 0 then provider or interface do not exist
                    if len(selected_rows) == 0:
                        # logging.info("[Init] Node: {} | Provider: {} | Interface: {}
                        # do not exist in CSV file".format(n + 1, PROVIDERS[p], INTERFACES[i]))
                        self.action_valid.append(False)
                    else:
                        logging.info(
                            "[Init] Node: {} | Provider: {} | Interface: {} "
                            "exists in CSV file".format(n + 1, PROVIDERS[p], INTERFACES[i]))
                        self.action_valid.append(True)

                    j += 1

        # logging.info("[Init] Resources:")
        # logging.info("[Init] CPU Capacity: {}".format(self.cpu_capacity))
        # logging.info("[Init] MEM Capacity: {}".format(self.memory_capacity))
        # logging.info("[Init] CPU allocated: {}".format(self.allocated_cpu))
        # logging.info("[Init] MEM allocated: {}".format(self.allocated_memory))
        # logging.info("[Init] CPU free: {}".format(self.free_cpu))
        # logging.info("[Init] MEM free: {}".format(self.free_memory))

        # Choose a random timestamp to start Episode
        self.get_start_index()

        # Update network
        self.update_network_values()

        # Variables for logging
        self.current_time = 0
        self.penalty = False
        self.accepted_requests = 0
        self.offered_requests = 0
        self.ep_accepted_requests = 0
        self.next_request()

        # Info & episode over
        self.total_reward = 0
        self.episode_over = False
        self.info = {}
        self.block_prob = 0
        self.ep_block_prob = 0
        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results_name = file_results_name
        self.file_results = file_results_name + ".csv"
        self.obs_csv = self.name + "_obs.csv"

    # Reset Function
    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.ep_accepted_requests = 0
        self.penalty = False

        self.block_prob = 0
        self.ep_block_prob = 0
        self.avg_total_latency = []
        self.avg_processing_latency = []
        self.avg_access_latency = []
        self.avg_deployment_cost = []
        self.avg_rtt = []
        self.avg_ul = []
        self.avg_dl = []
        self.avg_jitter = []
        self.total_latency = []

        self.avg_load_served_per_provider = np.zeros(NUM_PROVIDERS)
        self.telia_requests = 0
        self.telenor_requests = 0
        self.ice_requests = 0

        # Reset Deployment Data
        self.deploymentList = get_c2e_deployment_list()

        # Metrics for all interfaces
        self.rtt = np.zeros(self.total_number)
        self.ul = np.zeros(self.total_number)
        self.dl = np.zeros(self.total_number)
        # self.latency = np.zeros(self.total_number)
        self.jitter = np.zeros(self.total_number)
        self.processing_latency = np.zeros(self.total_number)

        # New: Resource capacities based on node type
        self.cpu_capacity = np.zeros(self.total_number)
        self.memory_capacity = np.zeros(self.total_number)
        self.node_type = [0] * self.total_number

        logging.info("[Reset] Resource Capacities... ")
        j = 0
        for n in range(self.num_nodes):
            node_type = int(self.np_random.integers(low=0, high=NUM_NODE_TYPES))
            for p in range(NUM_PROVIDERS):
                for i in range(NUM_INTERFACES):
                    self.node_id[j] = n
                    self.provider_id[j] = p
                    self.interface_id[j] = i
                    self.node_type[j] = node_type
                    self.cpu_capacity[j] = DEFAULT_NODE_TYPES[node_type]['cpu']
                    self.memory_capacity[j] = DEFAULT_NODE_TYPES[node_type]['mem']

                    logging.info("[Reset] node: {} | provider: {} | interface: {} | Type: {} "
                                 "| cpu: {} | mem: {}".format(n + 1, PROVIDERS[p], INTERFACES[i],
                                                              DEFAULT_NODE_TYPES[node_type]['type'],
                                                              self.cpu_capacity[j],
                                                              self.memory_capacity[j]))
                    j += 1

        # Keeps track of allocated resources
        self.allocated_cpu = np.zeros(self.total_number)
        self.allocated_memory = np.zeros(self.total_number)

        random_cpu = self.np_random.uniform(low=0.0, high=0.2, size=self.num_nodes)
        random_memory = self.np_random.uniform(low=0.0, high=0.2, size=self.num_nodes)
        j = 0
        for n in range(self.num_nodes):
            for i in range(NUM_PROVIDERS):
                for k in range(NUM_INTERFACES):
                    self.allocated_cpu[j] = random_cpu[n]
                    self.allocated_memory[j] = random_memory[n]
                    j += 1

        # Keeps track of Free resources for deployment requests
        self.free_cpu = np.zeros(self.total_number)
        self.free_memory = np.zeros(self.total_number)

        # Do not consider CSV part in reset to speedup training

        # files for each node
        self.node_csv_data = []
        self.df_node = []
        self.action_valid = []

        # Rows for each node
        self.df_node_selected_rows = []
        self.selected_ts = None

        j = 0
        file = ""
        for n in range(self.num_nodes):
            # Choose a random CSV file for each node
            if os.path.exists(self.path_csv_files):
                file = random.choice(os.listdir(self.path_csv_files))
                logging.info("[Reset] FileName: {}".format(file))

            for p in range(NUM_PROVIDERS):
                for i in range(NUM_INTERFACES):
                    self.free_cpu[j] = self.cpu_capacity[j] - self.allocated_cpu[j]
                    self.free_memory[j] = self.memory_capacity[j] - self.allocated_memory[j]

                    # Update files for each node
                    self.node_csv_data.append(self.path_csv_files + file)
                    self.df_node.append(pd.read_csv(self.node_csv_data[j]))

                    # Check if provider and interface exist in dataframe
                    if p == TELIA:
                        id_provider = TELIA_CSV
                    elif p == TELENOR:
                        id_provider = TELENOR_CSV
                    else:
                        id_provider = ICE_CSV

                    if i == fourG_CSV:
                        id_interface = four
                    else:
                        id_interface = five

                    # Select rows based on provider and interface
                    selected_rows = self.df_node[j][(self.df_node[j][PROVIDERS_CSV] == id_provider) &
                                                    (self.df_node[j][INTERFACES_CSV] == id_interface)]

                    # Reset index of the DataFrame
                    # selected_rows.reset_index(drop=True, inplace=True)

                    # print(selected_rows)
                    self.df_node_selected_rows.append(selected_rows)

                    # logging.info("[Init] Node: {} | Provider: {} | Interface: {} | size of rows: {}".format(n + 1,
                    #                                                                                        PROVIDERS[
                    #                                                                                            p],
                    #                                                                                        INTERFACES[
                    #                                                                                           i],
                    #                                                                                        len(selected_rows)))

                    # If len(rows) = 0 then provider or interface do not exist
                    if len(selected_rows) == 0:
                        # logging.info("[Init] Node: {} | Provider: {} | Interface: {}
                        # do not exist in CSV file".format(n + 1, PROVIDERS[p], INTERFACES[i]))
                        self.action_valid.append(False)
                    else:
                        logging.info(
                            "[Init] Node: {} | Provider: {} | Interface: {} "
                            "exists in CSV file".format(n + 1, PROVIDERS[p], INTERFACES[i]))

                        self.action_valid.append(True)

                    j += 1

        # Choose a random index to start Episode
        self.get_start_index()

        # Update network
        self.update_network_values()

        # return obs
        return np.array(self.get_state())

    # Step function
    def step(self, action):
        if self.current_step == 1:
            self.time_start = time.time()

        # Execute one time step within the environment
        self.offered_requests += 1
        self.take_action(action)

        # Calculate Reward
        reward = self.get_reward()
        self.total_reward += reward

        # Find correct action move for logging purposes
        move = ""
        if action < self.total_number:
            move = ACTIONS[0] + "-" + str(int(self.node_id[action] + 1)) \
                   + "-Provider-" + str(int(self.provider_id[action] + 1)) \
                   + "-Interface-" + str(int(self.interface_id[action] + 1))

        elif action == self.total_number:
            move = ACTIONS[1]

        # Logging Step and Total Reward
        logging.info('[Step {}] | Action: {} | Reward: {} | Total Reward: {}'.format(self.current_step, move, reward,
                                                                                     self.total_reward))

        # Get next request
        self.next_request()

        # Update RTT values
        self.update_network_values()

        # Update observation
        ob = self.get_state()

        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # self.save_obs_to_csv(self.obs_csv, np.array(ob), date)

        # episode results to save
        self.block_prob = 1 - (self.accepted_requests / self.offered_requests)
        self.ep_block_prob = 1 - (self.ep_accepted_requests / self.current_step)

        if len(self.avg_access_latency) == 0 and len(self.avg_deployment_cost) == 0 \
                and len(self.avg_rtt) == 0 and len(self.avg_dl) == 0 and len(self.avg_ul) == 0 \
                and len(self.avg_jitter) == 0 and len(self.total_latency) == 0 and len(
            self.avg_processing_latency) == 0:
            avg_c = 1
            avg_rtt = 1
            avg_dl = 1
            avg_ul = 1
            avg_jitter = 1
            avg_l = 1
            total_latency = 1
            avg_proc = 1
        else:
            avg_c = mean(self.avg_deployment_cost)
            avg_rtt = mean(self.avg_rtt)
            avg_dl = mean(self.avg_dl)
            avg_ul = mean(self.avg_ul)
            avg_jitter = mean(self.avg_jitter)
            avg_l = mean(self.avg_access_latency)
            total_latency = mean(self.avg_total_latency)
            avg_proc = mean(self.avg_processing_latency)

        self.info = {
            "reward_step": float("{:.2f}".format(reward)),
            "action": float("{:.2f}".format(action)),
            "reward": float("{:.2f}".format(self.total_reward)),
            "ep_block_prob": float("{:.2f}".format(self.ep_block_prob)),
            "ep_accepted_requests": float("{:.2f}".format(self.ep_accepted_requests)),
            'avg_deployment_cost': float("{:.2f}".format(avg_c)),
            'avg_total_latency': float("{:.2f}".format(total_latency)),
            'avg_access_latency': float("{:.2f}".format(avg_l)),
            'avg_processing_latency': float("{:.2f}".format(avg_proc)),
            'avg_rtt': float("{:.2f}".format(avg_rtt)),
            'avg_dl': float("{:.2f}".format(avg_dl)),
            'avg_ul': float("{:.2f}".format(avg_ul)),
            'avg_jitter': float("{:.2f}".format(avg_jitter)),
            'gini': float("{:.2f}".format(calculate_gini_coefficient(self.avg_load_served_per_provider))),
            'telia_requests': float("{:.2f}".format(self.avg_load_served_per_provider[TELIA])),
            'telenor_requests': float("{:.2f}".format(self.avg_load_served_per_provider[TELENOR])),
            'ice_requests': float("{:.2f}".format(self.avg_load_served_per_provider[ICE])),
            'executionTime': float("{:.2f}".format(self.execution_time))
        }

        if self.current_step == self.episode_length:
            self.episode_count += 1
            self.episode_over = True
            self.execution_time = time.time() - self.time_start

            gini = calculate_gini_coefficient(self.avg_load_served_per_provider)

            logging.info("[Step] Episode finished, saving results to csv...")
            save_to_csv(self.file_results, self.episode_count,
                        self.total_reward, self.ep_block_prob,
                        self.ep_accepted_requests,
                        mean(self.avg_deployment_cost),
                        mean(self.avg_total_latency),
                        mean(self.avg_access_latency),
                        mean(self.avg_processing_latency),
                        mean(self.avg_rtt),
                        mean(self.avg_dl),
                        mean(self.avg_ul),
                        mean(self.avg_jitter),
                        gini,
                        self.avg_load_served_per_provider[TELIA],
                        self.avg_load_served_per_provider[TELENOR],
                        self.avg_load_served_per_provider[ICE],
                        self.execution_time)

        # return ob, reward, self.episode_over, self.info
        return np.array(ob), reward, self.episode_over, self.info

    # Reward Function
    def get_reward(self):
        """ Calculate Rewards """
        if self.reward_function == NAIVE:
            if self.penalty:
                if not self.check_if_node_is_really_full():
                    logging.info("[Get Reward] Penalty = True, and resources "
                                 "were available, penalize the agent...")
                    return -1
                else:  # agent should not be penalized
                    logging.info("[Get Reward] Penalty = True, but resources "
                                 "were not available, do not penalize the agent...")
                    return 1
            else:
                return 1
        # Multi-objective
        elif self.reward_function == MULTI:
            if self.penalty:
                if not self.check_if_node_is_really_full():
                    logging.info("[Get Reward] Penalty = True, and resources "
                                 "were available, penalize the agent...")
                    return -1
                else:  # agent should not be penalized
                    logging.info("[Get Reward] Penalty = True, but resources "
                                 "were not available, do not penalize the agent...")
                    return 1
            else:  # Multi-objective reward function: latency + cost + gini + bandwidth
                # Latency
                latency = self.deployment_request.expected_rtt + self.deployment_request.expected_access_latency + self.deployment_request.expected_processing_latency
                logging.info('[Multi Reward] Latency components: RTT: {} | Lat: {} | Processing: {}'.format(
                    self.deployment_request.expected_rtt,
                    self.deployment_request.expected_access_latency,
                    self.deployment_request.expected_processing_latency))
                # Gini
                gini = calculate_gini_coefficient(self.avg_load_served_per_provider)
                # Cost
                cost = self.deployment_request.expected_cost
                # Bandwidth
                bandwidth = self.deployment_request.expected_dl_bandwidth + self.deployment_request.expected_ul_bandwidth

                logging.info(
                    '[Multi Reward] latency: {} | gini: {} | cost: {} | bandwidth: {}'.format(latency, gini, cost,
                                                                                              bandwidth))

                latency = normalize(latency, MIN_RTT + MIN_LATENCY + MIN_PROC, MAX_RTT + MAX_LATENCY + MAX_PROC)
                cost = normalize(cost, MIN_COST, MAX_COST)
                bandwidth = normalize(bandwidth, MIN_DL + MIN_UL, MAX_DL + MAX_UL)

                reward = self.latency_weight * (1 - latency) + self.gini_weight * (1 - gini) + self.cost_weight * (
                        1 - cost) + self.bandwidth_weight * bandwidth

                logging.info(
                    '[Multi Reward] Normalized: latency: {} | gini: {} | cost: {} | bandwidth: {}'.format(latency, gini,
                                                                                                          cost,
                                                                                                          bandwidth))
                logging.info(
                    '[Multi Reward] Applying weights: latency {} | gini part: {} | cost: {} | bandwidth: {} |'.format(
                        self.latency_weight * (1 - latency),
                        self.gini_weight * (1 - gini),
                        self.cost_weight * (1 - cost),
                        self.bandwidth_weight * bandwidth))
                logging.info('[Multi Reward] Final reward: {}'.format(reward))

                return reward
        else:
            logging.info('[Get Reward] Unrecognized reward: {}'.format(self.reward_function))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

    # Apply the action selected by the RL agent
    def take_action(self, action):
        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == self.episode_length:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.episode_over = True

        # Possible Actions: Place all replicas together or split them.
        # Known as NP-hard problem (Bin pack with fragmentation)
        # Any ideas for heuristic? We can later compare with an ILP/MILP model...
        # Check first if "Place all" Action can be performed
        if action < self.total_number:
            if self.check_if_node_is_full_after_full_deployment(action) or not self.action_valid[action]:
                self.penalty = True
                logging.info('[Take Action] Block the selected action since action is invalid or node will be full!')
                # Do not raise error since algorithm might not support action mask
                # raise ValueError("Action mask is not working properly. Full nodes should be always masked.")
            else:
                # accept request
                logging.info("[Take Action] Accept request...")
                self.accepted_requests += 1
                self.ep_accepted_requests += 1
                self.processing_latency[action] += PROCESSING_DELAY

                type_id = self.node_type[action]
                self.avg_load_served_per_provider[int(self.provider_id[action])] += 1
                self.deployment_request.deployed_node = self.node_id[action]
                self.deployment_request.action_id = action
                self.deployment_request.deployed_provider = self.provider_id[action]

                # self.deployment_request.expected_rtt = self.rtt[action]
                # self.deployment_request.expected_ul_bandwidth = self.ul[action]
                # self.deployment_request.expected_dl_bandwidth = self.dl[action]
                # self.deployment_request.expected_jitter = self.jitter[action]

                self.avg_deployment_cost.append(DEFAULT_NODE_TYPES[type_id]['cost'])
                self.avg_total_latency.append(
                    DEFAULT_NODE_TYPES[type_id]['latency'] + self.processing_latency[action] + self.rtt[action])

                self.avg_access_latency.append(DEFAULT_NODE_TYPES[type_id]['latency'])
                self.avg_processing_latency.append(self.processing_latency[action])

                self.avg_rtt.append(self.rtt[action])
                self.avg_ul.append(self.ul[action])
                self.avg_dl.append(self.dl[action])
                self.avg_jitter.append(self.jitter[action])

                self.deployment_request.expected_dl_bandwidth = self.dl[action]
                self.deployment_request.expected_ul_bandwidth = self.ul[action]
                self.deployment_request.expected_rtt = self.rtt[action]
                self.deployment_request.expected_access_latency = DEFAULT_NODE_TYPES[type_id]['latency']
                self.deployment_request.expected_cost = DEFAULT_NODE_TYPES[type_id]['cost']
                self.deployment_request.expected_processing_latency = self.processing_latency[action]

                self.penalty = False

                # Update allocated amounts
                for n in range(len(self.node_id)):
                    if self.node_id[n] == self.node_id[action]:
                        self.allocated_cpu[n] += self.deployment_request.cpu_request
                        self.allocated_memory[n] += self.deployment_request.memory_request

                        # Update free resources
                        self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
                        self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

                        # Update processing latency
                        self.processing_latency[n] += PROCESSING_DELAY

                # Update the request
                self.enqueue_request(self.deployment_request)

        # Reject the request: give the agent a penalty, especially if the request could have been accepted
        elif action == self.total_number:
            self.penalty = True
        else:
            logging.info('[Take Action] Unrecognized Action: {}'.format(action))

    def get_state(self):
        # Get Observation state
        node = np.full(shape=(1, NUM_METRICS_NODES), fill_value=-1)

        observation = np.stack([self.allocated_cpu,
                                self.cpu_capacity,
                                self.allocated_memory,
                                self.memory_capacity,
                                self.provider_id,
                                self.interface_id,
                                self.rtt,
                                # self.latency,
                                self.ul,
                                self.dl,
                                self.jitter,
                                self.processing_latency,
                                ],
                               axis=1)

        # logging.info('[Get State]: node: {}'.format(node))
        # logging.info('[Get State]: node shape: {}'.format(node.shape))
        # logging.info('[Get State]: observation: {}'.format(observation))
        # logging.info('[Get State]: observation shape: {}'.format(observation.shape))

        # Condition the elements in the set with the current node request
        request = np.tile(
            np.array(
                [self.deployment_request.cpu_request,
                 self.deployment_request.memory_request,
                 self.deployment_request.latency_threshold,
                 # self.deployment_request.ul_traffic,
                 # self.deployment_request.dl_traffic,
                 self.dt]
            ),
            (self.total_number + 1, 1),
        )

        # logging.info('[Get State]: request: {}'.format(request))
        # logging.info('[Get State]: request shape: {}'.format(request.shape))

        observation = np.concatenate([observation, node], axis=0)
        # logging.info('[Get State]: concatenation: {}'.format(observation))
        # logging.info('[Get State]: concatenation shape: {}'.format(observation.shape))

        observation = np.concatenate([observation, request], axis=1)
        # logging.info('[Get State]: concatenation: {}'.format(observation))
        # logging.info('[Get State]: concatenation shape: {}'.format(observation.shape))

        '''
        logging.info('[Get State]: cluster: {}'.format(cluster))
        logging.info('[Get State]: cluster shape: {}'.format(cluster.shape))
        logging.info('[Get State]: observation: {}'.format(observation))
        logging.info('[Get State]: observation shape: {}'.format(observation.shape))
        logging.info('[Get State]: request demands: {}'.format(request_demands))
        logging.info('[Get State]: request demands shape: {}'.format(request_demands.shape))
        logging.info('[Get State]: concatenation: {}'.format(observation))
        logging.info('[Get State]: concatenation shape: {}'.format(observation.shape))
        '''

        # logging.info("[Get State] Observation: {}".format(observation))
        return observation

    # Save observation to csv file
    def save_obs_to_csv(self, obs_file, obs, date):
        file = open(obs_file, 'a+', newline='')  # append
        # file = open(file_name, 'w', newline='') # new
        fields = []
        node_obs = {}
        with file:
            fields.append('date')
            for n in range(self.num_nodes):
                fields.append("node_" + str(n + 1) + '_allocated_cpu')
                fields.append("node_" + str(n + 1) + '_cpu_capacity')
                fields.append("node_" + str(n + 1) + '_allocated_memory')
                fields.append("node_" + str(n + 1) + '_memory_capacity')
                fields.append("node_" + str(n + 1) + '_num_replicas')
                fields.append("node_" + str(n + 1) + '_cpu_request')
                fields.append("node_" + str(n + 1) + '_memory_request')
                fields.append("node_" + str(n + 1) + '_dt')

            # logging.info("[Save Obs] fields: {}".format(fields))

            writer = csv.DictWriter(file, fieldnames=fields)
            # writer.writeheader() # write header

            node_obs.update({fields[0]: date})

            for n in range(self.num_nodes):
                i = self.get_iteration_number(n)
                node_obs.update({fields[i + 1]: obs[n][0]})
                node_obs.update({fields[i + 2]: obs[n][1]})
                node_obs.update({fields[i + 3]: obs[n][2]})
                node_obs.update({fields[i + 4]: obs[n][3]})
                node_obs.update({fields[i + 5]: obs[n][4]})
                node_obs.update({fields[i + 6]: obs[n][5]})
                node_obs.update({fields[i + 7]: obs[n][6]})
                node_obs.update({fields[i + 8]: obs[n][7]})
            writer.writerow(node_obs)
        return

    def get_iteration_number(self, n):
        num_fields_per_node = 8
        return num_fields_per_node * n

    def enqueue_request(self, request: DeploymentRequest) -> None:
        heapq.heappush(self.running_requests, (request.departure_time, request))

    # Action masks
    def action_masks(self):
        valid_actions = np.ones(self.total_number + 1, dtype=bool)
        j = 0
        for n in range(self.num_nodes):
            for p in range(NUM_PROVIDERS):
                for i in range(NUM_INTERFACES):
                    if self.check_if_node_is_full_after_full_deployment(n) or not self.action_valid[j]:
                        valid_actions[j] = False
                    else:
                        valid_actions[j] = True
                    j += 1

        # 1 additional action: Reject
        valid_actions[self.total_number] = True
        # logging.info('[Action Mask]: Valid actions {} |'.format(valid_actions))
        return valid_actions

    # Double-check if the selected cluster is full
    def check_if_node_is_full_after_full_deployment(self, action):
        total_cpu = self.deployment_request.cpu_request
        total_memory = self.deployment_request.memory_request

        if (self.allocated_cpu[action] + total_cpu > 0.95 * self.cpu_capacity[action]
                or self.allocated_memory[action] + total_memory > 0.95 * self.memory_capacity[action]):
            logging.info('[Check]: Node is full... Action id: {}'.format(action + 1))
            return True

        return False

    # Remove deployment request
    def dequeue_request(self):
        _, deployment_request = heapq.heappop(self.running_requests)
        logging.info("[Dequeue] Request will be terminated...")

        action = deployment_request.action_id
        total_cpu = self.deployment_request.cpu_request
        total_memory = self.deployment_request.memory_request

        '''
        logging.info("[Dequeue] Before")
        logging.info("[Dequeue] Action ID: {}".format(action))
        logging.info("[Dequeue] CPU allocated: {}".format(self.allocated_cpu))
        logging.info("[Dequeue] CPU free: {}".format(self.free_cpu))
        logging.info("[Dequeue] MEM allocated: {}".format(self.allocated_memory))
        logging.info("[Dequeue] MEM free: {}".format(self.free_memory))
        '''
        # Update allocated amounts
        for n in range(len(self.node_id)):
            if self.node_id[n] == self.node_id[action]:
                self.allocated_cpu[n] -= total_cpu
                self.allocated_memory[n] -= total_memory

                # Update free resources
                self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
                self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

                # Update processing latency
                self.processing_latency[n] -= PROCESSING_DELAY

        '''
        logging.info("[Dequeue] After")
        logging.info("[Dequeue] Action ID: {}".format(action))
        logging.info("[Dequeue] CPU allocated: {}".format(self.allocated_cpu))
        logging.info("[Dequeue] CPU free: {}".format(self.free_cpu))
        logging.info("[Dequeue] MEM allocated: {}".format(self.allocated_memory))
        logging.info("[Dequeue] MEM free: {}".format(self.free_memory))
        logging.info("[Dequeue] Processing Delay: {}".format(self.processing_latency))
        '''

    # Check if all clusters are full
    def check_if_node_is_really_full(self) -> bool:
        is_full = [self.check_if_node_is_full_after_full_deployment(i) for i in range(self.num_nodes)]
        return np.all(is_full)

    # Create a deployment request
    def deployment_generator(self):
        deployment_list = get_c2e_deployment_list()
        n = self.np_random.integers(low=0, high=len(deployment_list))
        d = deployment_list[n - 1]
        return d

    # Select (random) the next deployment request
    def next_request(self) -> None:
        arrival_time = self.current_time + self.np_random.exponential(scale=1 / self.arrival_rate_r)
        departure_time = arrival_time + self.np_random.exponential(scale=self.call_duration_r)
        self.dt = departure_time - arrival_time
        self.current_time = arrival_time

        while True:
            if self.running_requests:
                next_departure_time, _ = self.running_requests[0]
                if next_departure_time < arrival_time:
                    self.dequeue_request()
                    continue
            break

        self.deployment_request = self.deployment_generator()
        self.deployment_request.cpu_request = self.factor * self.deployment_request.cpu_request
        self.deployment_request.memory_request = self.factor * self.deployment_request.memory_request

        logging.info('[Next Request]: Name: {} | CPU: {} | MEM: {}'.format(self.deployment_request.name,
                                                                           self.deployment_request.cpu_request,
                                                                           self.deployment_request.memory_request))

    # Choose random index (ts) from dataframe to start simulation with at least 300 samples left for each node
    def get_start_index(self):
        # check min size and id of the node
        min_size = 1000000
        id = 0
        for n in range(self.num_nodes):
            if len(self.df_node[n]) < min_size:
                min_size = len(self.df_node[n])
                id = n

        # Choose a random index, but making sure 300 consecutive samples exist
        start_index = np.random.randint(0, min_size - 300)

        # Get the timestamp at the random index
        self.selected_ts = self.df_node[id].loc[start_index, 'ts']
        logging.info("Selected TS: {}".format(self.selected_ts))

        j = 0
        for n in range(self.num_nodes):
            for p in range(NUM_PROVIDERS):
                for i in range(NUM_INTERFACES):
                    # If provider and interface exist
                    if self.action_valid[j]:
                        # Select rows starting from ts
                        self.df_node_selected_rows[j] = self.df_node[j][
                            (self.df_node[j]['ts'] >= self.selected_ts)]

                        # Reset index of the final DataFrame
                        self.df_node_selected_rows[j].reset_index(drop=True, inplace=True)

                    j += 1
        return

    def update_network_values(self):
        if self.current_step == 0:
            step = 1
        else:
            step = self.current_step + 1

        j = 0
        for n in range(self.num_nodes):
            for p in range(NUM_PROVIDERS):
                for i in range(NUM_INTERFACES):
                    if self.action_valid[j]:
                        # Update values
                        self.rtt[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_RTT_Q90]
                        self.ul[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_UL]
                        self.dl[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_DL]
                        # self.latency[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_LATENCY]
                        self.jitter[j] = self.df_node_selected_rows[j].at[step, DF_COLUMN_JITTER]

                        logging.info("[update_network_values] Node: {} | Provider: {} | Interface: {} |"
                                     "RTT Q90: {} | UL: {} | DL: {} | Jitter: {}".format(n + 1,
                                                                                         PROVIDERS[p],
                                                                                         INTERFACES[i],
                                                                                         self.rtt[j],
                                                                                         self.ul[j],
                                                                                         self.dl[j],
                                                                                         # self.latency[j],
                                                                                         self.jitter[j])
                                     )
                    else:
                        # Fill all values as -1
                        self.rtt[j] = -1
                        self.ul[j] = -1
                        self.dl[j] = -1
                        # self.latency[j] = -1
                        self.jitter[j] = -1

                    j += 1
        return

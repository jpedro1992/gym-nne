import csv
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


# DeploymentRequest Info
@dataclass
class DeploymentRequest:
    name: str
    cpu_request: float
    cpu_limit: float  # limits can be left out
    memory_request: float
    memory_limit: float
    arrival_time: float
    latency_threshold: int  # Latency threshold that should be respected
    # ul_traffic: int # Expected Uplink traffic
    # dl_traffic: int # Expected downlink traffic
    departure_time: float
    action_id: int = None  # action id
    deployed_provider: int = None

    deployed_node: int = None  # All replicas deployed in one cluster
    expected_cost: int = None  # expected cost after deployment
    expected_dl_bandwidth: int = None  # expected downlink bandwidth after deployment
    expected_ul_bandwidth: int = None  # expected uplink bandwidth after deployment

    expected_rtt: int = None  # expected RTT
    expected_access_latency: int = None  # expected latency based on node type
    expected_processing_latency: int = None  # expected processing latency


# Reverses a dict
def sort_dict_by_value(d, reverse=False):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))


# Normalize function
def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0.0  # Avoid division by zero if min_value equals max_value
    return (value - min_value) / (max_value - min_value)


def get_c2e_deployment_list():
    deployment_list = [
        # 1 adapter-amqp
        DeploymentRequest(name="adapter-amqp",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
        # 2 adapter-http
        DeploymentRequest(name="adapter-http",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
        # 3 adapter-mqtt
        DeploymentRequest(name="adapter-mqtt",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
        # 4 adapter-mqtt
        DeploymentRequest(name="artemis",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.6, memory_limit=0.6,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 5 dispatch-router
        DeploymentRequest(name="dispatch-router",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.06, memory_limit=0.25,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 6 ditto-connectivity
        DeploymentRequest(name="ditto-connectivity",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.7, memory_limit=1.0,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 7 ditto-gateway
        DeploymentRequest(name="ditto-gateway",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 8 ditto-nginx
        DeploymentRequest(name="ditto-nginx",
                          cpu_request=0.05, cpu_limit=0.15,
                          memory_request=0.016, memory_limit=0.032,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 9 ditto-policies
        DeploymentRequest(name="ditto-policies",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=100),

        # 10 ditto-swagger-ui
        DeploymentRequest(name="ditto-swagger-ui",
                          cpu_request=0.05, cpu_limit=0.1,
                          memory_request=0.016, memory_limit=0.032,
                          arrival_time=0, departure_time=0,
                          latency_threshold=400),

        # 11 ditto-things
        DeploymentRequest(name="ditto-things",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 12 ditto-things-search
        DeploymentRequest(name="ditto-things-search",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 13 ditto-mongo-db
        DeploymentRequest(name="ditto-mongo-db",
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),

        # 14 service-auth
        DeploymentRequest(name="service-auth",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.2, memory_limit=0.25,
                          arrival_time=0, departure_time=0,
                          latency_threshold=300),

        # 15 service-command-router
        DeploymentRequest(name="service-command-router",
                          cpu_request=0.15, cpu_limit=1.0,
                          memory_request=0.2, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=300),

        # 16 service-device-registry
        DeploymentRequest(name="service-device-registry",
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.4, memory_limit=0.4,
                          arrival_time=0, departure_time=0,
                          latency_threshold=200),
    ]
    return deployment_list


# TODO: modify function
'''
def save_obs_to_csv(file_name, timestamp, num_pods, desired_replicas, cpu_usage, mem_usage,
                    traffic_in, traffic_out, latency, lstm_1_step, lstm_5_step):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='') # new
    with file:
        fields = ['date', 'num_pods', 'cpu', 'mem', 'desired_replicas',
                  'traffic_in', 'traffic_out', 'latency', 'lstm_1_step', 'lstm_5_step']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()  # write header
        writer.writerow(
            {'date': timestamp,
             'num_pods': int("{}".format(num_pods)),
             'cpu': int("{}".format(cpu_usage)),
             'mem': int("{}".format(mem_usage)),
             'desired_replicas': int("{}".format(desired_replicas)),
             'traffic_in': int("{}".format(traffic_in)),
             'traffic_out': int("{}".format(traffic_out)),
             'latency': float("{:.3f}".format(latency)),
             'lstm_1_step': int("{}".format(lstm_1_step)),
             'lstm_5_step': int("{}".format(lstm_5_step))}
        )
'''


def save_to_csv(file_name, episode, reward, ep_block_prob, ep_accepted_requests, avg_deployment_cost, avg_total_latency,
                avg_access_latency, avg_proc_latency,
                avg_rtt, avg_dl, avg_ul, avg_jitter, gini, telia_requests, telenor_requests, ice_requests,
                execution_time):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='')
    with file:
        fields = ['episode', 'reward', 'ep_block_prob', 'ep_accepted_requests', 'avg_deployment_cost',
                  'avg_total_latency', 'avg_access_latency', 'avg_proc_latency',
                  'avg_rtt', 'avg_dl', 'avg_ul', 'avg_jitter', 'gini', 'telia_requests', 'telenor_requests',
                  'ice_requests', 'execution_time']
        writer = csv.DictWriter(file, fieldnames=fields)
        # writer.writeheader()
        writer.writerow(
            {'episode': episode,
             'reward': float("{:.2f}".format(reward)),
             'ep_block_prob': float("{:.2f}".format(ep_block_prob)),
             'ep_accepted_requests': float("{:.2f}".format(ep_accepted_requests)),
             'avg_deployment_cost': float("{:.2f}".format(avg_deployment_cost)),
             'avg_total_latency': float("{:.2f}".format(avg_total_latency)),
             'avg_access_latency': float("{:.2f}".format(avg_access_latency)),
             'avg_proc_latency': float("{:.2f}".format(avg_proc_latency)),
             'avg_rtt': float("{:.2f}".format(avg_rtt)),
             'avg_dl': float("{:.2f}".format(avg_dl)),
             'avg_ul': float("{:.2f}".format(avg_ul)),
             'avg_jitter': float("{:.2f}".format(avg_jitter)),
             'gini': float("{:.2f}".format(gini)),
             'telia_requests': telia_requests,
             'telenor_requests': telenor_requests,
             'ice_requests': ice_requests,
             'execution_time': float("{:.2f}".format(execution_time))}
        )


# Calculation of Gini Coefficient
# 0 is better - 1 is worse!
def calculate_gini_coefficient(loads):
    n = len(loads)
    total_load = sum(loads)
    mean_load = total_load / n if n != 0 else 0

    if mean_load == 0:
        return 0  # Handle the case where all loads are zero to avoid division by zero

    gini_numerator = sum(abs(loads[i] - loads[j]) for i in range(n) for j in range(n))
    gini_coefficient = gini_numerator / (2 * n ** 2 * mean_load)

    return gini_coefficient

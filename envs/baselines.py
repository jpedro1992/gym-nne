import numpy.typing as npt
import gym
import numpy as np


def latency_greedy_policy(env: gym.Env, action_mask: npt.NDArray, ) -> int:
    """Returns the index of a feasible node that minimizes the latency."""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    # print("[Latency-Greedy] Feasible nodes: {}".format(feasible_nodes))

    if len(feasible_nodes) == 0:
        return len(action_mask) - 1
    return feasible_nodes[np.argmin(env.rtt[feasible_nodes])]


def bandwidth_greedy_policy(env: gym.Env, action_mask: npt.NDArray, ) -> int:
    """Returns the index of a feasible node that maximizes the DL capacity."""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    # print("[Bandwidth-Greedy] Feasible nodes: {}".format(feasible_nodes))

    # Get the endpoint with the highest DL capacity
    if len(feasible_nodes) == 0:
        return len(action_mask) - 1
    return feasible_nodes[np.argmax(env.dl[feasible_nodes])]


def cost_greedy_policy(env: gym.Env, action_mask: npt.NDArray, ) -> int:
    """Returns the index of a feasible node that minimizes the deployment cost."""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    # print("[Cost-Greedy] Feasible nodes: {}".format(feasible_nodes))

    # Get the endpoint with the lowest deployment cost
    if len(feasible_nodes) == 0:
        return len(action_mask) - 1

    # mean deployment cost based on node type
    deployment_costs = [env.default_node_types[env.node_type[node]]['cost'] for node in feasible_nodes]
    # print("[Cost-Greedy] Deployment Cost of feasible nodes: {}".format(deployment_costs))
    # deployment_costs = [cost if cost is not None else float('inf') for cost in deployment_costs]  # replace None with infinity

    return feasible_nodes[np.argmin(deployment_costs)]

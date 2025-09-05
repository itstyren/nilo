from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import random
import numpy as np
import networkx as nx
from typing import Callable
import gymnasium


def make_env(raw_env):
    def env(args, **kwargs):
        env = raw_env(args, **kwargs)
        return env

    return env


def parallel_wrapper_fn(env_fn: Callable) -> Callable:
    def par_fn(args, **kwargs):
        env = env_fn(args, **kwargs)
        env = aec_to_parallel_wrapper(env)
        return env

    return par_fn


def assign_group_indices(agent_list, group_num):
    agent_num = len(agent_list)

    if group_num <= 0:

        for agent in agent_list:
            agent.group_idx = 0

    elif group_num >= agent_num:

        for i, agent in enumerate(agent_list):
            agent.group_idx = i

    else:

        indices = list(range(agent_num))
        random.shuffle(
            indices
        )
        group_size = (
            agent_num // group_num
        )
        remainder = (
            agent_num % group_num
        )


        start = 0
        for group_idx in range(group_num):
            end = (
                start + group_size + (1 if group_idx < remainder else 0)
            )
            for i in indices[start:end]:
                agent_list[i].group_idx = group_idx
            start = end

    return agent_list


def gen_random_donor_recipient_pairs(agents_list, num_recipients=1, in_group_prob=0.8):
    num_agents = len(agents_list)

    if num_recipients >= num_agents:
        raise ValueError(
            "The number of neighbours must be less than the number of agents."
        )
    if not (0 <= in_group_prob <= 1):
        raise ValueError("in_group_prob must be between 0 and 1.")


    group_indices = np.array([agent.group_idx for agent in agents_list])
    unique_groups = np.unique(group_indices)


    for agent in agents_list:
        agent.recipients = set()


    if len(unique_groups) == 1:
        agent_indices = np.arange(num_agents)
        for agent_index in range(num_agents):
            possible_recipients = np.setdiff1d(
                agent_indices, [agent_index]
            )
            recipients = np.random.choice(
                possible_recipients, size=num_recipients, replace=False
            )
            agents_list[agent_index].recipients = list(recipients)


    else:

        group_to_agents = {
            group: np.where(group_indices == group)[0] for group in unique_groups
        }


        for agent_index in range(num_agents):
            current_recipients = set()

            while len(current_recipients) < num_recipients:
                if np.random.rand() < in_group_prob:

                    possible_recipients = group_to_agents[
                        agents_list[agent_index].group_idx
                    ]
                else:

                    possible_recipients = np.setdiff1d(
                        np.arange(num_agents),
                        group_to_agents[agents_list[agent_index].group_idx],
                    )


                possible_recipients = possible_recipients[
                    possible_recipients != agent_index
                ]
                possible_recipients = np.setdiff1d(
                    possible_recipients, list(current_recipients)
                )

                if len(possible_recipients) > 0:
                    recipient_idx = np.random.choice(possible_recipients)
                    current_recipients.add(recipient_idx)

            agents_list[agent_index].recipients = list(current_recipients)
            print(f"Agent {agent_index} connected to: {current_recipients}")

    return agents_list


def sample_repu_for_agents(space, agent_num, rng):
    if isinstance(space, gymnasium.spaces.Discrete):
        return np.array(
            [float(rng.integers(low=0, high=space.n)) for _ in range(agent_num)]
        )

    elif isinstance(space, gymnasium.spaces.Box):
        low = space.low[0]
        high = space.high[0]
        return np.around(
            [rng.uniform(low=low, high=high) for _ in range(agent_num)], decimals=2
        )

    else:
        raise NotImplementedError(
            f"Sampling not implemented for space type: {type(space)}"
        )


def gen_connected_interaction_pairs(
    agents_list, num_recipients=1, in_group_prob=0.8, call_times=0,seed=None
):


    _self_loops = 0


    np.random.seed(seed + call_times)

    try:
        num_agents = len(agents_list)
        if num_recipients > num_agents:
            raise ValueError(
                "The number of recipients must not exceed the number of agents."
            )
        if not (0 <= in_group_prob <= 1):
            raise ValueError("in_group_prob must be between 0 and 1.")


        group_indices = np.array([agent.group_idx for agent in agents_list])
        unique_groups = np.unique(group_indices)


        group_to_agents = {
            group: np.where(group_indices == group)[0].tolist()
            for group in unique_groups
        }


        remaining_slots = {i: num_recipients for i in range(num_agents)}


        connections = {
            i: [] for i in range(num_agents)
        }


        for i, agent in enumerate(agents_list):
            while len(connections[i]) < num_recipients:

                if np.random.rand() < in_group_prob:

                    possible_recipients = set(group_to_agents[agent.group_idx])
                else:

                    possible_recipients = set(range(num_agents)) - set(
                        group_to_agents[agent.group_idx]
                    )


                possible_recipients -= {i} | set(connections[i])


                possible_recipients = {
                    x for x in possible_recipients if remaining_slots[x] > 0
                }


                if not possible_recipients:

                    recipient = i
                    _self_loops += 1
                else:
                    recipient = int(np.random.choice(list(possible_recipients)))


                connections[i].append(
                    recipient
                )
                if recipient != i:
                    connections[recipient].append(i)


                remaining_slots[i] -= 1
                if recipient != i and remaining_slots[recipient] > 0:
                    remaining_slots[recipient] -= 1


        for agent_index, agent in enumerate(agents_list):
            agent.recipients = connections[agent_index]


        return _self_loops
    finally:

        np.random.seed(seed)


def categorize_interactions(agents_list, num_groups):

    observable_agents = {group_id: set() for group_id in range(num_groups)}

    for agent in agents_list:

        observable_agents[agent.group_idx].add(agent.index)


        for recipient_index in agent.recipients:
            recipient_agent = next(
                (a for a in agents_list if a.index == recipient_index), None
            )
            if recipient_agent:

                observable_agents[recipient_agent.group_idx].add(agent.index)


    return {
        group_id: sorted(observable)
        for group_id, observable in observable_agents.items()
    }


def display_agent_connections(agents_list):
    total_in_group = 0
    total_out_group = 0


    for agent in agents_list:
        in_group_count = sum(
            1
            for recipient_id in agent.recipients
            if agents_list[recipient_id].group_idx == agent.group_idx
        )
        out_group_count = len(agent.recipients) - in_group_count


        total_in_group += in_group_count
        total_out_group += out_group_count


    print("=" * 40)
    print("Overall Connection Summary:")
    print(
        f"  Total in-group connections: {total_in_group} {total_in_group/(total_in_group + total_out_group)*100:.2f}%"
    )
    print(
        f"  Total out-group connections: {total_out_group} {total_out_group/(total_in_group + total_out_group)*100:.2f}%"
    )
    print(f"  Total connections: {total_in_group + total_out_group}")


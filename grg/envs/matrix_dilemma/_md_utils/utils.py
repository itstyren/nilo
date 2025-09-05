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
    """
    Assigns a group index to each agent in the agent_list based on the group_num.

    Parameters:
    - agent_list (list): List of agents (can be any data type, typically a list of dictionaries or objects).
    - group_num (int): Number of groups to divide the agents into.

    Returns:
    - list: Updated agent_list where each agent has a 'group_idx' attribute.
    """
    agent_num = len(agent_list)

    if group_num <= 0:
        # If group_num is 0 or negative, all agents are in group 0
        for agent in agent_list:
            agent.group_idx = 0

    elif group_num >= agent_num:
        # If group_num is greater than or equal to agent_num, each agent is isolated in its own group
        for i, agent in enumerate(agent_list):
            agent.group_idx = i  # Each agent gets its own unique group index

    else:
        # Otherwise, divide agents among group_num groups
        indices = list(range(agent_num))
        random.shuffle(
            indices
        )  # Shuffle indices to ensure randomness in group assignments
        group_size = (
            agent_num // group_num
        )  # Number of agents in each group (except for remaining agents)
        remainder = (
            agent_num % group_num
        )  # Extra agents that don't fit into evenly sized groups

        # Assign agents to groups
        start = 0
        for group_idx in range(group_num):
            end = (
                start + group_size + (1 if group_idx < remainder else 0)
            )  # Distribute remaining agents among groups
            for i in indices[start:end]:
                agent_list[i].group_idx = group_idx
            start = end

    return agent_list


def gen_random_donor_recipient_pairs(agents_list, num_recipients=1, in_group_prob=0.8):
    """
    Efficiently generate donor-recipient pairs for each agent.

    Parameters:
    - agents_list (list): List of agents, each agent should have 'id' and 'group_idx' attributes.
    - num_connection (int): Number of recipient connections each agent should have.
    - in_group_prob (float): Probability (0 to 1) that a donor connects to a recipient within the same group.

    Returns:
    - agents_list (list): Each agent will have a 'recipients' attribute containing the list of connected agents.
    """
    num_agents = len(agents_list)

    if num_recipients >= num_agents:
        raise ValueError(
            "The number of neighbours must be less than the number of agents."
        )
    if not (0 <= in_group_prob <= 1):
        raise ValueError("in_group_prob must be between 0 and 1.")

    # Extract agent group indices
    group_indices = np.array([agent.group_idx for agent in agents_list])
    unique_groups = np.unique(group_indices)

    # Create recipient list for each agent
    for agent in agents_list:
        agent.recipients = set()  # Use set to avoid duplicates

    # ** If only one group, use simpler logic **
    if len(unique_groups) == 1:
        agent_indices = np.arange(num_agents)
        for agent_index in range(num_agents):
            possible_recipients = np.setdiff1d(
                agent_indices, [agent_index]
            )  # Exclude self
            recipients = np.random.choice(
                possible_recipients, size=num_recipients, replace=False
            )
            agents_list[agent_index].recipients = list(recipients)

    # ** Multiple group case (original logic retained) **
    else:
        # Pre-group agents for efficient lookup
        group_to_agents = {
            group: np.where(group_indices == group)[0] for group in unique_groups
        }

        # ** First Pass: Randomly sample connections **
        for agent_index in range(num_agents):
            current_recipients = set()

            while len(current_recipients) < num_recipients:
                if np.random.rand() < in_group_prob:
                    # Sample from the same group
                    possible_recipients = group_to_agents[
                        agents_list[agent_index].group_idx
                    ]
                else:
                    # Sample from different groups
                    possible_recipients = np.setdiff1d(
                        np.arange(num_agents),
                        group_to_agents[agents_list[agent_index].group_idx],
                    )

                # Exclude self-connections and agents already connected
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


# def gen_connected_interaction_pairs(agents_list, num_recipients=1, in_group_prob=0.8):
#     """
#     Efficiently generate donor-recipient pairs for each agent, ensuring they are connected to each other.

#     Parameters:
#     - agents_list (list): List of agents, each agent should have 'id' and 'group_idx' attributes.
#     - num_recipients (int): Number of recipient connections each agent should have.
#     - in_group_prob (float): Probability (0 to 1) that a donor connects to a recipient within the same group.

#     Returns:
#     - agents_list (list): Each agent will have a 'recipients' attribute containing the list of connected agents.
#     """
#     num_agents = len(agents_list)
#     if num_recipients >= num_agents:
#         raise ValueError("The number of recipients must be less than the number of agents.")
#     if not (0 <= in_group_prob <= 1):
#         raise ValueError("in_group_prob must be between 0 and 1.")

#     # Extract agent group indices
#     group_indices = np.array([agent.group_idx for agent in agents_list])
#     unique_groups = np.unique(group_indices)

#     # Initialize NetworkX graph
#     G = nx.Graph()
#     G.add_nodes_from(range(num_agents))

#     # Group agents for efficient lookup
#     group_to_agents = {group: np.where(group_indices == group)[0] for group in unique_groups}

#     # Track connections
#     connections = {i: set() for i in range(num_agents)}

#     for i, agent in enumerate(agents_list):

#         possible_recipients = []

#         # Step 1: Choose group based on in_group probability
#         if np.random.rand() < in_group_prob:
#             # In-group selection
#             possible_recipients = group_to_agents[agent.group_idx].tolist()
#         else:
#             # Out-group selection
#             possible_recipients = np.setdiff1d(np.arange(num_agents), group_to_agents[agent.group_idx]).tolist()

#         # Remove self from possible recipients
#         possible_recipients = [x for x in possible_recipients if x != i]

#         # Shuffle recipients to ensure randomness
#         np.random.shuffle(possible_recipients)

#         # Step 2: Select unique recipients, ensuring undirected connectivity
#         while len(connections[i]) < num_recipients:
#             if not possible_recipients:
#                 raise ValueError("Not enough available agents to make the required connections.")

#             recipient = possible_recipients.pop()

#             if len(connections[recipient]) < num_recipients:
#                 # Create bidirectional connection
#                 connections[i].add(recipient)
#                 connections[recipient].add(i)

#     # Step 3: Assign the connections to agents
#     for agent_index, agent in enumerate(agents_list):
#         agent.recipients = list(connections[agent_index])
#         print(f"Agent {agent_index} connected to: {list(connections[agent_index])}")

#     return agents_list


# def gen_connected_interaction_pairs(agents_list, num_recipients=1, in_group_prob=0.8,call_times=0):
#     """
#     Generate donor-recipient pairs for each agent, ensuring exactly `num_recipients` connections for each.
#     Includes a selective fallback mechanism to guarantee connectivity.

#     Parameters:
#     - agents_list (list): List of agents, each agent should have 'id' and 'group_idx' attributes.
#     - num_recipients (int): Number of recipient connections each agent should have.
#     - in_group_prob (float): Probability (0 to 1) that a donor connects to a recipient within the same group.

#     Returns:
#     - agents_list (list): Each agent will have a 'recipients' attribute containing the list of connected agents.
#     """
#     # Save the current state of the random number generator
#     rng_state = np.random.get_state()

#     # Extract current seed and increment it temporarily
#     seed = rng_state[1][0]  # The first element in the RNG state tuple is the seed
#     np.random.seed(seed + call_times)

#     try:
#         num_agents = len(agents_list)
#         if num_recipients >= num_agents:
#             raise ValueError("The number of recipients must be less than the number of agents.")
#         if not (0 <= in_group_prob <= 1):
#             raise ValueError("in_group_prob must be between 0 and 1.")

#         # Extract agent group indices
#         group_indices = np.array([agent.group_idx for agent in agents_list])
#         unique_groups = np.unique(group_indices)

#         # Group agents for efficient lookup
#         group_to_agents = {group: np.where(group_indices == group)[0].tolist() for group in unique_groups}

#         # Keep track of remaining connection slots for each agent
#         remaining_slots = {i: num_recipients for i in range(num_agents)}

#         # Initialize connections dictionary
#         connections = {i: set() for i in range(num_agents)}

#         # Helper function for fallback
#         def fallback_random_recipient(agent_idx):
#             """Select a random recipient from all agents, ignoring connection limits."""
#             possible_recipients = [x for x in range(num_agents) if x != agent_idx]
#             recipient = np.random.choice(possible_recipients)
#             return recipient

#         # Generate connections
#         for i, agent in enumerate(agents_list):
#             while len(connections[i]) < num_recipients:
#                 # Step 1: Choose group based on in_group probability
#                 if np.random.rand() < in_group_prob:
#                     # In-group selection
#                     possible_recipients = set(group_to_agents[agent.group_idx])
#                 else:
#                     # Out-group selection
#                     possible_recipients = set(range(num_agents)) - set(group_to_agents[agent.group_idx])

#                 # Remove self and already connected agents
#                 possible_recipients -= {i} | connections[i]

#                 # Filter by remaining slots
#                 possible_recipients = {x for x in possible_recipients if remaining_slots[x] > 0}

#                 # If no valid recipients, use fallback
#                 if not possible_recipients:
#                     print(f"Fallback triggered for agent {i}.")
#                     recipient = fallback_random_recipient(i)
#                 else:
#                     recipient = np.random.choice(list(possible_recipients))

#                 # Add bidirectional connection
#                 connections[i].add(int(recipient))
#                 connections[recipient].add(i)

#                 # Update remaining slots, allowing recipient to exceed limit in fallback cases
#                 remaining_slots[i] -= 1
#                 if remaining_slots[recipient] > 0:  # Only decrement if within limit
#                     remaining_slots[recipient] -= 1

#         # Assign connections to agents
#         for agent_index, agent in enumerate(agents_list):
#             agent.recipients = list(connections[agent_index])
#             print(f"Agent {agent_index} connected to: {agent.recipients}")

#         return agents_list
#     finally:
#         # Restore the original random state
#         np.random.set_state(rng_state)


def gen_connected_interaction_pairs(
    agents_list, num_recipients=1, in_group_prob=0.8, call_times=0,seed=None
):
    """
    Generate donor-recipient pairs for each agent, ensuring exactly `num_recipients` connections for each.
    Allows self-loops to count multiple times if no valid recipients are available.

    Parameters:
    - agents_list (list): List of agents, each agent should have 'id' and 'group_idx' attributes.
    - num_recipients (int): Number of recipient connections each agent should have.
    - in_group_prob (float): Probability (0 to 1) that a donor connects to a recipient within the same group.

    Returns:
    - agents_list (list): Each agent will have a 'recipients' attribute containing the list of connected agents.
    """
    # Save the current state of the random number generator
    # rng_state = np_rng.get_state()
    _self_loops = 0

    # Extract current seed and increment it temporarily
    # seed = rng_state[1][0]  # The first element in the RNG state tuple is the seed
    np.random.seed(seed + call_times)

    try:
        num_agents = len(agents_list)
        if num_recipients > num_agents:
            raise ValueError(
                "The number of recipients must not exceed the number of agents."
            )
        if not (0 <= in_group_prob <= 1):
            raise ValueError("in_group_prob must be between 0 and 1.")

        # Extract agent group indices
        group_indices = np.array([agent.group_idx for agent in agents_list])
        unique_groups = np.unique(group_indices)

        # Group agents for efficient lookup
        group_to_agents = {
            group: np.where(group_indices == group)[0].tolist()
            for group in unique_groups
        }

        # Keep track of remaining connection slots for each agent
        remaining_slots = {i: num_recipients for i in range(num_agents)}

        # Initialize connections dictionary
        connections = {
            i: [] for i in range(num_agents)
        }  # Allow duplicates for multiple self-loops

        # Generate connections
        for i, agent in enumerate(agents_list):
            while len(connections[i]) < num_recipients:
                # Step 1: Choose group based on in_group probability
                if np.random.rand() < in_group_prob:
                    # In-group selection
                    possible_recipients = set(group_to_agents[agent.group_idx])
                else:
                    # Out-group selection
                    possible_recipients = set(range(num_agents)) - set(
                        group_to_agents[agent.group_idx]
                    )

                # Remove self and already connected agents
                possible_recipients -= {i} | set(connections[i])

                # Filter by remaining slots
                possible_recipients = {
                    x for x in possible_recipients if remaining_slots[x] > 0
                }

                # If no valid recipients, allow self-loop
                if not possible_recipients:
                    # print(f"Adding self-loop for agent {i}.")
                    recipient = i  # Self-loop
                    _self_loops += 1
                else:
                    recipient = int(np.random.choice(list(possible_recipients)))

                # Add connection (self-loop or bidirectional)
                connections[i].append(
                    recipient
                )  # Duplicate entries allowed for self-loops
                if recipient != i:
                    connections[recipient].append(i)

                # Update remaining slots
                remaining_slots[i] -= 1
                if recipient != i and remaining_slots[recipient] > 0:
                    remaining_slots[recipient] -= 1

        # Assign connections to agents
        for agent_index, agent in enumerate(agents_list):
            agent.recipients = connections[agent_index]
            # print(f"Agent {agent_index} connected to: {agent.recipients}")

        return _self_loops
    finally:
        # Restore the original random state
        np.random.seed(seed)


def categorize_interactions(agents_list, num_groups):
    """
    Categorize interactions and find observable agent indices for each group.

    Args:
        agents_list (list of Agent): List of agents.
        num_groups (int): Number of groups.

    Returns:
        dict: A dictionary where keys are group numbers, and values are lists
              of observable agent indices for each group.
    """
    # Create a dictionary to store observable indices for each group
    observable_agents = {group_id: set() for group_id in range(num_groups)}

    for agent in agents_list:
        # Add the agent itself to its own group's observable set
        observable_agents[agent.group_idx].add(agent.index)

        # Iterate through the agent's recipients
        for recipient_index in agent.recipients:
            recipient_agent = next(
                (a for a in agents_list if a.index == recipient_index), None
            )
            if recipient_agent:
                # Add the agent to the observable set of the recipient's group
                observable_agents[recipient_agent.group_idx].add(agent.index)

    # Convert sets to sorted lists for consistent output
    return {
        group_id: sorted(observable)
        for group_id, observable in observable_agents.items()
    }


def display_agent_connections(agents_list):
    """
    Display information about the recipients for each agent, including the number of in-group and out-group recipients.

    Parameters:
    - agents_list (list): List of agents, where each agent has 'id', 'group_idx', and 'recipients' attributes.
    """
    total_in_group = 0
    total_out_group = 0

    # print("\nAgent Connection Details:\n")
    for agent in agents_list:
        in_group_count = sum(
            1
            for recipient_id in agent.recipients
            if agents_list[recipient_id].group_idx == agent.group_idx
        )
        out_group_count = len(agent.recipients) - in_group_count

        # Update total counts
        total_in_group += in_group_count
        total_out_group += out_group_count

    # Print overall connection summary
    print("=" * 40)
    print("Overall Connection Summary:")
    print(
        f"  Total in-group connections: {total_in_group} {total_in_group/(total_in_group + total_out_group)*100:.2f}%"
    )
    print(
        f"  Total out-group connections: {total_out_group} {total_out_group/(total_in_group + total_out_group)*100:.2f}%"
    )
    print(f"  Total connections: {total_in_group + total_out_group}")
    # print('='*40)





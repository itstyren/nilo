import numpy as np
from .._md_utils.core import World, Agent
from .._md_utils.matrix_env import MatrixEnv
from .._md_utils import utils
from .._md_utils.scenario import BaseScenario


class raw_env(MatrixEnv):
    """
    A Doner game environment has gym API for soical dilemma
    """

    def __init__(self, args, max_cycles=1, render_mode=None,seed=None):
        scenario = Scenario()
        world = scenario.make_world(args,seed=seed)
        MatrixEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            max_cycles=max_cycles,
            render_mode=render_mode,
            args=args,
        )
        self.metadata["name"] = "donation_v0"


env = utils.make_env(raw_env)
parallel_env = utils.parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()

    def make_world(self, args,seed=None):
        agent_num = args.env_dim**2

        # Payoff matrix for the dilemma
        dilemma_R = 1
        dilemma_P = 0

        dilemma_S = args.dilemma_S

        dilemma_parameter = np.array(
            [[dilemma_R, dilemma_S], [args.dilemma_T, dilemma_P]]
        )

        world = World(np.array(args.initial_ratio), dilemma_parameter)

        world.agents = [Agent(args) for _ in range(agent_num)]

        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.index = i

        self.baseline_type = args.baseline_type

        world.initialize_network(args,seed)

        return world

    def reset_world(self, world, args, action_space,np_rng,_rng_seed=None):
        """
        Randomly initialize the world and agnets strategy
        0 for coop and 1 for defect
        """
        # Generate dilemma_action directly using np.random.choice
        # the size of the action is (agent_num, num_recipients)
        agent_num = len(world.agents)

        dilemma_action = np_rng.choice(
            [1, 0],
            size=(agent_num, world.agents[0].num_recipients),
            p=world.initial_ratio.ravel(),
        )
        # Assign actions to agents efficiently
        for agent, dilemma_action in zip(world.agents, dilemma_action):
            agent.action.s = dilemma_action


        # print(repu_sample)
        repu_sample=utils.sample_repu_for_agents(action_space['reputation'],agent_num,np_rng)
        # print(repu_sample)
        for agent in world.agents:
            agent.reputation_view = repu_sample.copy()


        # reset the group index and get new recipient for each agent
        utils.assign_group_indices(world.agents, args.group_num)
        world.get_new_recipient(_rng_seed)
        # print("Recipient: ",world.agents[0].recipients)

        # print(f"Dilemma Action: {dilemma_action[0]}")
        # print(f"Reputation View: {world.agents[0].reputation_view[0]}")
        # print(f"Group Index: {world.agents[0].group_idx}")
        # print(f"Recipients: {world.agents[0].recipients}")

    def observation(
        self, central_agent, world, all_agent_actions, agent_recipients_idx
    ):
        """
        Get the observation for a particular agent

        :param all_agent_actions: the action of all agents
        :param agent_recipients_idx: the index of the all agent and their recipients

        The reputaion observation should be a list contain each combination
        """
        _nun_recipients = world.agents[central_agent.index].num_recipients
        repu_obs = []
        dilemma_obs = []


        # iterate all agents
        # dilemma_action: central agent dilemma strategy, a_r: [arrary list] central agent and its recipient repu
        for current_agent_idx, (obs_agent_act, obs_agent_recip_repu_full) in enumerate(
            zip(all_agent_actions, central_agent.reputation_view[agent_recipients_idx])
        ):
            obs_agent_self_repu = obs_agent_recip_repu_full[0]  # first element
            obs_agent_recip_repu = obs_agent_recip_repu_full[1:]  # remaining elements

            no_match = 0
            # interacted_idx=[]
            # iterate current agent' recipients, get idx of the recipients
            for recipients_idx in world.agents[current_agent_idx].recipients:
                # interacted_idx.append(recipients_idx)
                # check if current obs agent's recipient is in the same group with central agent
                if central_agent.group_idx != world.agents[recipients_idx].group_idx:
                    no_match += 1

            # interaction_obs.append(interacted_idx)
            r_obs = []
            # fully observable for same group agent
            # also append for those fully different interaction that can't be obsereved, only for better coding purpose (we dont update those)
            if (
                central_agent.group_idx == world.agents[current_agent_idx].group_idx
                or no_match == _nun_recipients
            ):  
                for i in range(_nun_recipients):
                    # fisrt is the current agent action, then the central agent reputation, then it's recipient reputation
                    current_recipient_idx = world.agents[current_agent_idx].recipients[i]
                    # print(np.where(
                    #     np.array(world.agents[current_recipient_idx].recipients) == current_agent_idx
                    # )[0])
                    index_inside_recipient=np.where(
                        np.array(world.agents[current_recipient_idx].recipients) == current_agent_idx
                    )[0][0]

                    agent_one_hot = np.eye(len(all_agent_actions))[current_agent_idx]
                    # Ensure all components are 1D arrays and concatenate
                    components = [
                        obs_agent_self_repu,
                        obs_agent_recip_repu[i],
                        obs_agent_act[i],
                        all_agent_actions[current_recipient_idx][index_inside_recipient],
                        agent_one_hot
                    ]
                    r_obs.append(np.concatenate([np.atleast_1d(x) for x in components]))
        
            # different group only know the interaction happened in the same group
            else:
                for i, recipient_idx in enumerate(
                    world.agents[current_agent_idx].recipients
                ):
                    if central_agent.group_idx == world.agents[recipient_idx].group_idx:
                        r_obs.append(
                            [
                                # 332,
                                # central_agent.action.s[i],
                                obs_agent_self_repu,
                                # agent_recipients_idx[i+1],
                                obs_agent_recip_repu[i],
                                obs_agent_act[i],   #  current agent actions
                                world.agents[current_agent_idx].recipients[i],
                            ]
                        )

                        # r_obs.append([obs_agent_act[i], obs_agent_recip_repu[i + 1]])

            repu_obs.append(r_obs)

        repu_obs = self.normalize_list(repu_obs, _nun_recipients)



        # Precompute self reputation
        central_self_repu = central_agent.reputation_view[central_agent.index]
        # Dilemma observation
        for recipient_idx in central_agent.recipients:
            if self.baseline_type == "NL":
                dilemma_obs.append(
                    world.agents[recipient_idx].action.s
                )  # recipient action
            else:
                recipient_repu = central_agent.reputation_view[recipient_idx]
                components = [
                    np.atleast_1d(central_self_repu),
                    np.atleast_1d(recipient_repu),
                ]
                dilemma_obs.append(np.concatenate(components))



        obs = {
            "repu_obs": np.array(repu_obs, dtype=np.float32),
            "dilemma_obs": np.array(dilemma_obs, dtype=np.float32),
        }

        return obs

    def calculate_reward(self, agent, world):
        """
        Calculate the reward for the agent
        """
        # Calculate the reward for the agent
        agent_rewards = []
        recipient_actions = []
        for i, recipient_idx in enumerate(agent.recipients):
            find_idx = int(
                np.where(
                    np.array(world.agents[recipient_idx].recipients) == agent.index
                )[0][0]
            )

            # self_act=1 if agent.reputation_view[recipient_idx] ==0 else 1
            # recipient_act=1 if world.agents[recipient_idx].reputation_view[agent.index] ==0 else 1
            # print(f"recipient action {world.agents[recipient_idx].action.s[find_idx]},repu view: {agent.reputation_view[recipient_idx]}")
            recipient_act = world.agents[recipient_idx].action.s[find_idx]
            agent_rewards.append(
                float(
                    world.payoff_matrix[
                        agent.action.s[i],
                        # self_act,
                        # world.agents[recipient_idx].action.s[find_idx],
                        recipient_act,
                    ]
                )
            )
            recipient_actions.append(recipient_act)
        return agent_rewards, recipient_actions

    def normalize_list(self, input_list, repeat_count):
        """
        Repeats each sublist in input_list based on repeat_count while ignoring empty sublists.
        """
        normalized_list = []

        for sublist in input_list:
            if sublist:  # Skip empty sublists
                repeat_factor = repeat_count // len(sublist)
                normalized_list.append(sublist * repeat_factor)

        return normalized_list

    def update_repu_view(self, agent, updatd_index_list, repu_action):
        """
        Update the reputation view of the agent
        """
        # only update actual observable agents
        for update_idx in updatd_index_list:
            agent.reputation_view[update_idx] = repu_action[update_idx]
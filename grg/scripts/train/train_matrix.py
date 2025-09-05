import random
import numpy as np
import setproctitle
import torch
from grg.config.config import get_config, update_config
import sys
from pathlib import Path
import os
import wandb
import datetime
import socket
from grg.utils import tools as utils
import supersuit as ss
from grg.envs.matrix_dilemma.md_wrappers import DummyVecEnv,SubprocVecEnv


def make_run_env(all_args, raw_env, env_type=0):
    """
    make train or eval env
    :param evn_type: wether env for training (0) or eval (1). Setting differnt seed
    """

    def get_env_fn(rank):
        def init_env():
            seed=all_args.seed * (1 + 4999 * env_type) + rank * 1000
            env = raw_env(
                all_args,
                max_cycles=all_args.episode_length,
                render_mode="train" if env_type == 0 else "eval",
                seed=seed,
            )
            env._seed(seed)
            return env

        return init_env

    rollout_threads = (
        all_args.n_rollout_threads if env_type == 0 else all_args.n_eval_rollout_threads
    )
    if rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(rollout_threads)])



device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)

    parser = get_config()
    # parse command-line arguments and pre-set argument from config.py
    parsed_args = parser.parse_known_args(sys.argv[1:])[0]
    all_args = update_config(parsed_args)

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = (
        Path(
            os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[
                0
            ]
            + "/results"
        )
        / all_args.project_name
        / all_args.scenario_name
        / all_args.substrate_name  # sepecify which substrate to run on
        / all_args.algorithm_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    running_substrate = utils.extract_scenarios(all_args.substrate_name)
    
    if all_args.use_wandb:
        job_name = f"{running_substrate}_{all_args.algorithm_name}"
        run_name = f"G{all_args.group_num}N{all_args.num_recipients}_dim{all_args.env_dim}_{all_args.norm_type}[{all_args.seed}]"
        run = wandb.init(
            config=all_args,
            project=all_args.project_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=run_name,
            dir=str(run_dir),
            job_type=job_name,
            group=all_args.scenario_name,
            sync_tensorboard=True,)
        wandb.define_metric('Steps')
        wandb.define_metric("*", step_metric="Steps",step_sync=True)
    else:
        # Generate a run name based on the current timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        curr_run = f"run_{current_time}"

        # Create the full path for the new run directory
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # Set the process title to the run name
    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.substrate_name)
        + "-"
        + str(all_args.exp_name)
        + str(all_args.env_dim)
        + "@"
        + str(all_args.user_name)
    )          


    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    from grg.envs.matrix_dilemma import pairwise_v1 as PairwiseEnv

    envs = make_run_env(all_args, PairwiseEnv.raw_env, env_type=0)

    config={
        "envs": envs,
        "all_args": all_args,
        "num_agents": all_args.env_dim**2,
        "device": device,
        "run_dir": run_dir,
    }

    from grg.runner.rl.on_policy.pairwise_runner import PairwiseRunner

    runner=PairwiseRunner(config)
    runner.run()



    envs.close()
    
    if all_args.use_wandb:
        wandb.finish()    

    # env =PairwiseEnv.env(all_args)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)

    # print(env.observation_space())

    # from pettingzoo.test import api_test

    # api_test(env, num_cycles=1000, verbose_progress=True)

    
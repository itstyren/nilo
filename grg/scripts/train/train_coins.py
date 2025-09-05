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
from grg.envs.coin_game.cg_wrappers import DummyVecEnv,SubprocVecEnv

def make_run_env(all_args, raw_env, env_type=0):

    def get_env_fn(rank):
        def init_env():
            seed=all_args.seed * (1 + 4999 * env_type) + rank * 1000
            env = raw_env(
                all_args,
                max_cycles=all_args.episode_length,
                render_mode="human" if not all_args.save_gifs else "rgb_array",
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
        / all_args.substrate_name
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
        wandb.define_metric("render_video", step_metric="Episodes",step_sync=True)
        wandb.define_metric('Steps')
        wandb.define_metric("*", step_metric="Steps",step_sync=True)
    else:

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        curr_run = f"run_{current_time}"


        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))


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


    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    from grg.envs.coin_game import coinRepu_v1 as CoinRepuEnv    

    envs = make_run_env(all_args, CoinRepuEnv.raw_env, env_type=0)

    eval_env = make_run_env(all_args, CoinRepuEnv.raw_env, env_type=1) if all_args.use_render else None

    config={
        "envs": envs,
        "eval_env": eval_env,
        "all_args": all_args,
        "num_agents": all_args.env_dim,
        "device": device,
        "run_dir": run_dir,
    }


    from grg.runner.rl.on_policy.coingame_runner import CoinGamenner

    runner=CoinGamenner(config)
    runner.run()


    envs.close()

    if all_args.use_render:
        eval_env.close()
    if all_args.use_wandb:
        wandb.finish()    
import os
import statistics

import gymnasium as gym
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.monitor import Monitor

# Create environment
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
total_timesteps = 100_000

env = gym.make("LunarLander-v3", render_mode="rgb_array",max_episode_steps=350)
env = Monitor(env, log_dir)

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch_size" : self.model.batch_size,
            "buffer_size": self.model.buffer_size,
            "learning_starts": self.model.learning_starts,
            "target_update_interval": self.model.target_update_interval,
            "exploration_fraction" : self.model.exploration_fraction,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0,
            "rollout/exploration_rate" : 0.0,
            "time/fps": 0,
            "time/episodes": 0,
            "train/loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:     # Log Q-values
        obs_tensor = torch.tensor(self.locals["replay_buffer"].observations[-1], device=self.model.device).float()
        with torch.no_grad():
            q_values = model.q_net(obs_tensor)
            self.logger.record("q_values/mean",q_values)
        return True

mean_reward_list = []
std_reward_list = []

tb_log_name = "DQN_no_max_ep_len"
current_run_folder = "./max_ep_len_runs/no_max_ep_len/"

runs= [1,2,3,4,5,6,7,8,10,11] #episode 9 überall rauslassen wegen crash mit replay_buffer 5e4 und max_ep_len_350

for i in runs:

    current_run= str(i)

    # Train the agent and display a progress bar
    model = DQN(policy="MlpPolicy",
            env=env,
            learning_rate=0.00063,      # default ist 1e-4, das target net wird also auf die geschätzte Fehlerrate etwas stärker angepasst
            batch_size=128,             # mini-batch size für replay Buffer sampling
            buffer_size=50_000,         # replay buffer size
            learning_starts=0,          # direkt lernen, keine trainigsdaten sammelphase am anfang
            gamma=0.99,                 # zukünftige Rewards werden bevorzugt
            tau=1,                      # default. Mit 1 machen wir "hard updates" des Target Networks
            target_update_interval=250, # update target Network nach 250 timesteps
            train_freq=4,               # train every 4 steps, default
            gradient_steps=-1,          # -1 as many gradient steps as steps done in the env during rollout, also 4 wie bei train_freq
            exploration_initial_eps=1,  # auch default
            exploration_fraction=0.12,  # nach (total_timesteps*exp_fraction) erreichen wir epsilon_min=0.1
            exploration_final_eps=0.1,  # default wäre 0.01, so noch höhere exploration am Ende erlaubt
            policy_kwargs=dict(net_arch=[256,256],                  # Hidden Layers im NN
                               optimizer_class=torch.optim.AdamW,   # AdamW anscheinend besser? Adam Optimizer scheint
                                                                    # laut internet die weights nicht ganz korrekt zu berechnet
                               activation_fn=torch.nn.ReLU),        # ist default, aber mal explizit reingeschrieben
            verbose=1,
            stats_window_size=10,       # amount of episodes to use for the stats window mean_scores
            device='cpu',               # wir machen keine Bildverarbeitung, ist irgendwann aufgefallen das CPU ein wenig schneller ist
            seed=int(current_run),      # seeding for reproducibility
            tensorboard_log="./dqn_lunar_tensorboard_no_max_ep_len")

    model.learn(total_timesteps=total_timesteps,
            tb_log_name=tb_log_name+"_"+current_run,
            callback=HParamCallback(),
            progress_bar=True)

    #save model
    if os.path.exists(current_run_folder):
        model.save(current_run_folder+tb_log_name+"_"+current_run)
        model.save_replay_buffer(current_run_folder+tb_log_name+"_"+current_run+"_replay_buffer")
    else:
        model.save(tb_log_name+current_run)
        model.save_replay_buffer(tb_log_name+"_"+current_run+"_replay_buffer")

    #evaluate Policy over 100 episodes and save mean/std reward to txt file
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    print(f"mean_reward: {mean_reward}\n std_reward: {std_reward}")
    mean_reward_list.append(mean_reward)
    std_reward_list.append(std_reward)

    if os.path.exists(current_run_folder+"eval.txt"):
        with open(current_run_folder+"eval.txt","a") as f:
            f.write(f"\nRun: {current_run} with 100 episodes evaluated\n"
                    f"mean_reward: {round(mean_reward, 2)}\n"
                    f"std_reward: {round(std_reward, 2)}\n"
                    f"episodes: \n")

#Durschnitt der Ergebnisse errechnen
#if os.path.exists(current_run_folder + "runs.txt"):
#    with open(current_run_folder + "runs.txt", "a") as f:
#        f.write(f"\nDurchschnitt über alle Läufe\n"
#        f"mean_reward: {statistics.mean(mean_reward_list)}\n"
#        f"std_reward: {statistics.mean(std_reward_list)}")
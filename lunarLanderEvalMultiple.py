import os
import gymnasium as gym
import matplotlib.pyplot as plt
import torch as th

from stable_baselines3 import DQN
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

# Create environment
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = Monitor(env, log_dir)

runs= [1,2,3,4,5,6,7,8,10,11] #episode 9 überall rauslassen wegen crash mit replay_buffer 5e4 und max_ep_len_350

current_run_folder="./max_ep_len_runs_replay_buffer/1e4/"
for i in runs:

    run_to_load= str(i)

    # load model
    model = DQN.load(current_run_folder+"DQN_max_ep_len_350_replay_buffer_1e4_"+run_to_load+".zip", env=env)

    #evaluate policy
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    print(f"mean_reward: {mean_reward}\n std_reward: {std_reward}")

    if os.path.exists(current_run_folder+"eval.txt"):
        with open(current_run_folder+"eval.txt","a") as f:
            f.write(f"\nRun: {run_to_load} with 100 episodes evaluated\n"
                    f"mean_reward: {round(mean_reward,2)}\n"
                    f"std_reward: {round(std_reward,2)}\n")


#results with trained agent for viewing
#vec_env = model.get_env()
#obs = vec_env.reset()
#for i in range(4000):
#    action, _states = model.predict(obs, deterministic=True)
#    obs, reward, done, info = vec_env.step(action)
#    with th.no_grad():
#        obs_tensor, _ = model.q_net.obs_to_tensor(obs)
#        q_values = model.q_net(obs_tensor)
#    if done:
#        print(f"info: {info}\nq_values: {q_values}")
#    vec_env.render("human")


#plot to show how many episodes succeeded
#total_timesteps = 100_000
#plot_results([log_dir], total_timesteps, results_plotter.X_EPISODES, "DQN Lunar")
#plt.show()

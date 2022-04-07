import torch
import gym
import numpy as np
import os
import gym

from agents import PPO, DDPG, SAC, TD3, Trainer, Config


config = Config()
config.seed = 1
config.environment = gym.make("MountainCarContinuous-v0")
config.num_episodes_to_run = 150
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {
    "Policy_Gradient_Agents": {
            "learning_rate": 0.05,
            "linear_hidden_units": [30, 15],
            "final_layer_activation": "TANH",
            "learning_iterations_per_round": 10,
            "discount_rate": 0.9,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 10,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.2,
            "epsilon_decay_rate_denominator": 1,
            "clip_rewards": False
        },

    "Actor_Critic_Agents": {
            "Actor": {
                "learning_rate": 0.003,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.02,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

        "min_steps_before_learning": 1000, #for SAC only
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": True,
        "do_evaluation_iterations": True,
        "clip_rewards": False

    }

}


class MyTrainer(Trainer):
	...

	
class RobustnessTrainer(MyTrainer):
	...


if __name__ == "__main__":
    AGENTS = [SAC] # [SAC]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()

# SAC, ,







# # Configuration
# gamma = 0.95
# trainEpisodeCount = 1500
# testEpisodeCount = 10
# stepCount = 10000
# prefix = 'stlTool/openAI/mountaincar'
# modelSaveName = f'{prefix}/model.pickle'
# saveTrainFile = f'{prefix}/data.py'

# # RL agent class
# class MyNetwork(torch.nn.Module):

# 	def __init__(self):
# 		super().__init__()
# 		self.model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
# 		self.savedActions = []
# 		self.rewards = []

# 	def forward(self, x):
# 		action = self.model(x)
# 		return action

# env = gym.make('MountainCarContinuous-v0')
# env.reset()
# criterion = torch.nn.MSELoss()
# model = MyNetwork()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# eps = np.finfo(np.float32).eps.item()


# def selectAction(model: MyNetwork, state):
# 	state = torch.from_numpy(state).float().unsqueeze(0)
# 	action = model(state)
# 	return action.detach().numpy()


# def noop(*args, **kwargs):
# 	pass
# def run(episodeCount, env, model, actionFn, training=True):
# 	rewards = []
# 	with open(saveTrainFile, 'w') as f:
# 		if not training:
# 			f.write = noop
# 		f.write("trainingData = ([\n")
# 		for episodeIndex in range(episodeCount):
# 			state = env.reset()
# 			maxObs = float('-inf')
# 			reward = 0
# 			f.write("\t[\n")
# 			for step in range(1, stepCount + 1):
# 				action = actionFn(model, state)
# 				if not training:
# 					env.render()
# 				state, stepReward, done, note = env.step(action); state = state.flatten().transpose()
# 				if state[0] > maxObs:
# 					maxObs = state[0]
# 				f.write(f"\t\t({[x for x in state]}, {stepReward}, {done}, {note}),\n")
# 				reward += stepReward
# 				if done:
# 					break
# 			f.write(f"\t],\n")
# 			if training:
# 				loss = criterion(torch.DoubleTensor([abs(maxObs)]), torch.DoubleTensor([0.45]))
# 				# print(f"Maximum achieved pos = {maxObs}, loss = {loss}")
# 				loss.requires_grad = True
# 				optimizer.zero_grad()
# 				loss.backward()
# 				optimizer.step()
# 			if episodeIndex % 50 == 0 and training:
# 				print(f"Finished episode {episodeIndex}")
# 			rewards.append(reward)
# 		f.write("])\n")
# 	return rewards


# if not os.path.exists(modelSaveName):
# 	print(f"No previously trained model found. Beginning training...")
# 	run(trainEpisodeCount, env, model, selectAction)
# 	print(f"Finished training!")
# else:
# 	print(f"Found pre-existing model! Loading...")
# 	model = torch.load(modelSaveName)

# print(f"Testing the model...")
# rewards = run(testEpisodeCount, env, model, selectAction, training=False)

# print(f"Model had the following scores in test: {rewards}")
# if os.path.exists(modelSaveName):
# 	wantToDel = True if input("Do you want to delete this model's weights?\n") == 'y' else False
# 	if wantToDel:
# 		os.remove(modelSaveName)
# if not os.path.exists(modelSaveName):
# 	wantToSave = True if input("Do you want to save the model weights?\n") == 'y' else False
# 	if wantToSave:
# 		print(f"Storing model weights...")
# 		torch.save(model, modelSaveName)

# env.close()
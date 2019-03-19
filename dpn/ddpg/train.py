
"""
DDPG (Actor-Critic) RL Example for Unity ML-Agents Environments using PyTorch
Includes examples of the following DDPG training algorithms:

The example uses a modified version of the Unity ML-Agents Reacher Example Environment.
The environment includes In this environment, a double-jointed arm can move to target locations. 
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of your agent is to maintain its position at the target location for as many 
time steps as possible.

Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""

###################################
# Import Required Packages
import torch
import random, torch
import numpy as np
from collections import deque
from drl.dpn.ddpg.ddpg_agent import Agent
from unityagents import UnityEnvironment
from drl.dpn.ddpg.reacher_env import ReacherEnv
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

"""
###################################
STEP 1: Set the Training Parameters
======
        num_episodes (int): maximum number of training episodes
        episode_scores (float): list to record the scores obtained from each episode
        scores_average_window (int): the window size employed for calculating the average score (e.g. 100)
        solved_score (float): the average score required for the environment to be considered solved
    """
num_episodes=500
episode_scores = []
scores_average_window = 100      
solved_score = 30     

"""
###################################
STEP 2: Start the Unity Environment
# Use the corresponding call depending on your operating system 
"""
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--os", default="linux", help="os")
parser.add_argument("--graph", action="store_true")
parser.add_argument("--udacity", action="store_true")
args = parser.parse_args()

env = ReacherEnv(os=args.os, display=args.graph)

agent = Agent(state_size=env.obs_dim, action_size=env.act_dim, num_agents=env.num_agents)


"""
###################################
STEP 6: Run the DDPG Training Sequence
The DDPG Training Process involves the agent learning from repeated episodes of behaviour 
to map states to actions the maximize rewards received via environmental interaction.

The agent training process involves the following:
(1) Reset the environment at the beginning of each episode.
(2) Obtain (observe) current state, s, of the environment at time t
(3) Perform an action, a(t), in the environment given s(t)
(4) Observe the result of the action in terms of the reward received and 
	the state of the environment at time t+1 (i.e., s(t+1))
(5) Update agent memory and learn from experience (i.e, agent.step)
(6) Update episode score (total reward received) and set s(t) -> s(t+1).
(7) If episode is done, break and repeat from (1), otherwise repeat from (3).

Below we also exit the training process early if the environment is solved. 
That is, if the average score for the previous 100 episodes is greater than solved_score.
"""

# loop from num_episodes
for i_episode in range(1, num_episodes+1):

    states, r, dones = env.reset()
    scores = np.zeros(env.num_agents)
    agent.reset()

    agent_scores = np.zeros(env.num_agents)

    # Run the episode training loop;
    # At each loop step take an action as a function of the current state observations
    # Based on the resultant environmental state (next_state) and reward received update the Agents Actor and Critic networks
    # If environment episode is done, exit loop...
    # Otherwise repeat until done == true 
    while True:
        # determine actions for the unity agents from current sate
        actions = agent.act(states)

        next_states, rewards, dones = env.step(actions)

        #Send (S, A, R, S') info to the training agent for replay buffer (memory) and network updates
        agent.step(states, actions, rewards, next_states, dones)

        # set new states to current states for determining next actions
        states = next_states

        # Update episode score for each unity agent
        agent_scores += rewards

        # If any unity agent indicates that the episode is done, 
        # then exit episode loop, to begin new episode
        if np.any(dones):
            break

    # Add episode score to Scores and...
    # Calculate mean score over last 100 episodes 
    # Mean score is calculated over current episodes until i_episode > 100
    episode_scores.append(np.mean(agent_scores))
    average_score = np.mean(episode_scores[i_episode-min(i_episode,scores_average_window):i_episode+1])

    #Print current and average score
    print('\nEpisode {}\tEpisode Score: {:.3f}\tAverage Score: {:.3f}'.format(i_episode, episode_scores[i_episode-1], average_score), end="")
    
    # Save trained  Actor and Critic network weights after each episode
    an_filename = "ddpgActor_Model.pth"
    torch.save(agent.actor_local.state_dict(), an_filename)
    cn_filename = "ddpgCritic_Model.pth"
    torch.save(agent.critic_local.state_dict(), cn_filename)

    # Check to see if the task is solved (i.e,. avearge_score > solved_score over 100 episodes). 
    # If yes, save the network weights and scores and end training.
    if i_episode > 100 and average_score >= solved_score:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode, average_score))

        # Save the recorded Scores data
        scores_filename = "ddpgAgent_Scores.csv"
        np.savetxt(scores_filename, episode_scores, delimiter=",")
        break


"""
###################################
STEP 7: Everything is Finished -> Close the Environment.
"""
env.close()

# END :) #############


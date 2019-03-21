import torch
import random
import numpy as np
from drl.dpn.ddpg.reacher_env import ReacherEnv
from .ddpg_agent import Agent

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

BATCH_SIZE=128

num_episodes=500
episode_scores = []
scores_average_window = 100      
solved_score = 30

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--os", default="linux", help="os")
parser.add_argument("--display", action="store_true")
parser.add_argument("--good", action="store_true")
args = parser.parse_args()

"""
###################################
STEP 2: Start the Unity Environment
# Use the corresponding call depending on your operating system 
"""

env = ReacherEnv(os=args.os, display=args.display)

print('\nNumber of Agents: ', env.num_agents)


"""
###################################
STEP 5: Create a DDPG Agent from the Agent Class in ddpg_agent.py
A DDPG agent initialized with the following parameters.
    ======
    state_size (int): dimension of each state (required)
    action_size (int): dimension of each action (required)
    num_agents (int): number of agents in the unity environment
    seed (int): random seed for initializing training point (default = 0)

Here we initialize an agent using the Unity environments state and action size and number of Agents
determined above.
"""
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

	# reset the training agent for new episode
    agent.reset()
    # set the initial episode score to zero.
    agent_scores = np.zeros(env.num_agents)
    while True:
        # determine actions for the unity agents from current sate
        actions = agent.act(states)
        # send the actions to the unity agents in the environment and receive resultant environment information
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


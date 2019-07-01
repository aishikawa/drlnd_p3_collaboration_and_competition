from collections import deque
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from unityagents import UnityEnvironment

from ddpg_agent import Agent


def plot_scores(scores, moving_socres, filename='score_history.png'):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores)), moving_socres)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    plt.savefig(filename)


def ddpg(env, brain_name, agent, num_agents, n_episodes=1000, print_every=1):
    scores_deque = deque(maxlen=100)
    episode_scores = []
    moving_scores = []
    noise_coefficient = 0.995
    for i_episode in range(1, n_episodes+1):
        s_time = time.time()
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)
        while True:
            actions = agent.act(states, noise_coefficient)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)

            states = next_states
            scores += rewards
            if np.any(dones):
                break
        noise_coefficient *= 0.995
        max_score = np.max(scores)
        scores_deque.append(max_score)
        episode_scores.append(max_score)
        moving_scores.append(np.mean(scores_deque))
        elapsed_time = time.time() - s_time
        if i_episode % print_every == 0:
            print(
                'Episode {}\tScore: {:.2f}\tMoving Average Score: {:.2f}\tTime: {:.2f}'
                    .format(i_episode, max_score, np.mean(scores_deque), elapsed_time),
                flush=True
            )
        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tMoving Average Score: {:.2f}'.format(i_episode - 100, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return episode_scores, moving_scores


def main():
    env = UnityEnvironment(file_name='Tennis_Linux_NoVis/Tennis.x86_64')

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    print('brain_name', brain_name)
    print('brain', brain)

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents: ', num_agents)

    action_size = brain.vector_action_space_size
    print('Size of each action: ', action_size)

    states = env_info.vector_observations
    state_size = states.shape[1]
    print(f'There are {states.shape[0]} agents. Each observes a state with length: {state_size}')
    print('The state for the first agent looks like:', states[0])

    agent = Agent(state_size, action_size, random_seed=1, n_agents=num_agents)

    scores, moving_scores = ddpg(env, brain_name, agent, num_agents, n_episodes=5000, print_every=1)
    plot_scores(scores, moving_scores)

    env.close()


if __name__ == '__main__':
    main()
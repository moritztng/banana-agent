from unityagents import UnityEnvironment
import numpy as np
import torch, time, argparse
import torch.nn as nn
from qnetwork import QNetwork

seed = 42

parser = argparse.ArgumentParser(description='Testing The Trained Model In Unity')
parser.add_argument('--episodes', required=True, type=int, help='Number of Episodes the Agent should play')
parser.add_argument('--actions_per_second', default=15, type=int, help='Actions per Second')

if  __name__ == "__main__":
    args = vars(parser.parse_args())

    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    q_policy = QNetwork(len(env_info.vector_observations[0]), brain.vector_action_space_size, seed).to(device)
    q_policy.load_state_dict(torch.load('checkpoint.pth'))

    for i in range(args['episodes']):
        env_info = env.reset(train_mode=True)[brain_name]
        state = torch.tensor(env_info.vector_observations[0]).float().to(device)
        score = 0
        while True:
            action_values = q_policy(state.unsqueeze(0))
            action = action_values.max(1)[1]
            env_info = env.step(action.item())[brain_name]

            reward = env_info.rewards[0]
            score += reward
            done = float(env_info.local_done[0])
            next_state = env_info.vector_observations[0]
            state = torch.tensor(next_state).float().to(device)
            if done:
                break
            time.sleep(1/args['actions_per_second'])
        print('Episode: {}, Score: {}'.format(i+1, score))

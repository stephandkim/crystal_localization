from src import *
from utils import *
from datetime import datetime
import gym
import random
import argparse
import torch
import json
import os
from torch.utils.tensorboard import SummaryWriter

def get_args():
    
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    parser = argparse.ArgumentParser(description='Training a DQN agent for localizing xtals.')
    parser.add_argument('--id', help='Identifier of this run', type=str, default=time_now)
    parser.add_argument('--image_name', help='Name of image', type=str, default='img_map0.npy')
    
    parser.add_argument('--num_episodes', help='Number of episodes', type=int, default=1000)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=128)
    
    parser.add_argument('--gamma', help='Gamma', type=float, default=0.9)
    parser.add_argument('--eps_start', help='Exploration start', type=float, default=0.9)
    parser.add_argument('--eps_end', help='Exploration end', type=float, default=0.05)
    parser.add_argument('--eps_decay', help='Exploration decay', type=float, default=10)
    parser.add_argument('--memory_size', help='Replay memory size', type=int, default=100000)
    
    parser.add_argument('--turn_max', help='Max turns', type=int, default=20)
    parser.add_argument('--target_update', help='Target update frequency(episode)', type=int, default=1)
    parser.add_argument('--model_save_freq', help='Model save frequency', type=int, default=100)
    parser.add_argument('--replay_save_freq', help='Replay save frequency for debugging', type=int, default=5)
    
    parser.add_argument('--SHIFT_STEPSIZE', help='Shift step size', type=int, default=5)
    parser.add_argument('--ZOOM_STEPSIZE', help='Zoom step size', type=int, default=5)
    
    parser.add_argument('--LAMBDA_X', help='lambda x', type=int, default=1)
    parser.add_argument('--LAMBDA_P', help='lambda p', type=int, default=0.1)
    parser.add_argument('--LAMBDA_S', help='lambda s', type=int, default=1)
    
    parser.add_argument('--l', help='inital l size', type=int, default=300)
    parser.add_argument('--h', help='inital h size', type=int, default=300)
    parser.add_argument('--M', help='M', type=int, default=3)
    
    args = parser.parse_args()
    param_save_path = 'data/'+args.id
    if not os.path.exists(param_save_path):
        os.mkdir(param_save_path)
    with open(param_save_path+'/parameters.json', 'w') as outfile:
        json.dump(vars(args), outfile, indent=4)
        outfile.close()
        
# #     Loading parameters from json file
#     with open(args.load_json, 'rt') as f:
#         t_args = argparse.Namespace()
#         t_args.__dict__.update(json.load(f))
#         args = parser.parse_args(namespace=t_args)

    return args

def train(args):
    path = 'data/'+args.id
    writer = SummaryWriter(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(device, M=args.M, writer=writer)

    agent.TARGET_UPDATE = args.target_update
    agent.BATCH_SIZE = args.batch_size
    agent.GAMMA = args.gamma
    agent.EPS_START = args.eps_start
    agent.EPS_END = args.eps_end
    agent.EPS_DECAY = args.eps_decay
    
    constants.LAMBDA_X = args.LAMBDA_X
    constants.LAMBDA_P = args.LAMBDA_P
    constants.LAMBDA_S = args.LAMBDA_S
    
    img_map = np.load('images/'+args.image_name)
    env = Environment(args.h, args.l, img_map.copy(), Square(1,args.M))
    env.turn_max = args.turn_max

    num_episodes = args.num_episodes

    # For recording computation time
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)

    for episode in range(num_episodes):
        env.termination = False
        env.reset()
        env.state.to(agent.device)

        agent.steps_done = 0

        for t in count():
#             start.record()

            action = agent.get_action(env.state)
            agent.policy_net.train()
            env.check_termination(t)

            state_next, reward = env.step(action)
            state_next = state_next.to(agent.device)

            env.action_tracker.append(action)
            env.total_reward_tracker.append(env.total_reward)

            agent.replay.push(
                SARS(
                    env.state.to(agent.device),
                    action,
                    torch.tensor([reward], dtype=torch.float).to(agent.device),
                    state_next
                )
            )

            env.state = state_next
            agent.optimize()

            if agent.total_count % args.model_save_freq == 0:
                torch.save(agent.policy_net.state_dict(), path+'/policy_'+str(agent.total_count))
                torch.save(agent.target_net.state_dict(), path+'/target_'+str(agent.total_count))
                print('Episode: {}, total_count: {}, models saved'.format(episode, agent.total_count))

            if episode % args.replay_save_freq == 0:
                if t % 5 == 0:
                    env.save_replay(path, episode, action.item())

            agent.total_count += 1
            if env.termination:
                break

#             end.record()
#             torch.cuda.synchronize()
#             print('time: {}, inference: {}, reward: {}'.format(start.elapsed_time(end),
#                                                                agent.inference,
#                                                                env.total_reward
#                                                               ))

        if episode % agent.TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                print('Episode: {}, total_count: {}, weights updated'.format(episode, agent.total_count))


if __name__ == '__main__':
    args = get_args()
    train(args)

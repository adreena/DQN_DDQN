from __future__ import print_function
import numpy as np
from tqdm import tqdm
import gym
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from collections import deque, namedtuple
from PIL import Image
from tensorboardX import SummaryWriter 
import os, sys, csv
import matplotlib.image as mpimg
import argparse
from config_reader import ConfigReader

print('Torch Version:',torch.__version__)

from dqn_ddqn_agent import Agent


def preprocess_state(image):
    img = np.reshape(image, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    img = Image.fromarray(img)
    resized_screen = img.resize((84, 110), Image.BILINEAR)
    resized_screen = np.array(resized_screen)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84])
    return x_t.astype(np.uint8)/255.


def evaluate(agent, csv_writer,iter_idx, num_episodes, stack_size):
    avg_score = []
    iter_record = [iter_idx]
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        score = 0.
        crop_pixels=8
        
        # current state & 2,3-previous states 
        state_frames = deque(maxlen=stack_size)

        step = 0
        while not done and step < max_steps:
            _state = preprocess_state(state)
            state = torch.from_numpy(_state).float()

            # if it's the first frame, copy the same state multiple time in the stack
            if len(state_frames) < stack_size:
                for i in range(stack_size):
                    state_frames.append(state)
            else:
                state_frames.append(state)

            state_stack = torch.stack(list(state_frames)).unsqueeze(dim=0)
            action = agent.act(state_stack, epsilon=0.05)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
        
            score += reward
            step +=1
        avg_score.append(score)
    iter_record.extend(avg_score)
    csv_writer.writerow(iter_record)
    return  np.mean(avg_score)

def run(experiment_name, num_iterations, learning_rate, buffer_size, batch_size, gamma,epsilon, epsilod_decay, epsilon_min, stack_size, device, is_ddqn, evaluation_rate, evaluation_num_episodes, log_directory, validation_directory):
    scores = []
    
    episodic_accum = 0
    epsidoic_rewards = []
    iteration_rewards = []
    episode=1
    print('???', is_ddqn)
    agent = Agent(env=env, state_space=state_space, action_space=action_space, learning_rate=learning_rate,\
                     buffer_size=buffer_size, batch_size=batch_size, gamma=gamma,\
                     device=device, in_channels=stack_size, is_ddqn = is_ddqn)
    
    #initializing log directory for tensorboard
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    tb_writer = SummaryWriter('{}/{}'.format(log_directory,experiment_name))
    
    # initializing validation directiry for csv records of evaluations
    if not os.path.exists(validation_directory):
        os.makedirs(validation_directory)
        
    frame_count = 0
    epoch_plot_count=0
    stop = False
    with open('{}/{}.csv'.format(validation_directory, experiment_name), 'w') as validation_file:
        # initializing csv file and header
        csv_writer = csv.writer(validation_file)
        csv_headers = ['iter_idx']
        for ep_id in range(evaluation_num_episodes):
            csv_headers.append('episode_{}'.format(ep_id))
        csv_writer.writerow(csv_headers)
        
        while agent.num_train_updates < num_iterations+1 and not stop:
            state = env.reset()
            done = False

            # current state & 3-previous states 
            state_frames = deque(maxlen=stack_size)

            episode_reward = []
            prev_iteration = None
            while not done:
                frame_count+=1

                _state = preprocess_state(state)
                state = torch.from_numpy(_state).float()

                # if it's the first frame, copy the same state multiple time in the stack
                if len(state_frames) < stack_size:
                    for i in range(stack_size):
                        state_frames.append(state)
                else:
                    state_frames.append(state)

                state_stack = torch.stack(list(state_frames)).unsqueeze(dim=0)
                action = agent.act(state_stack, epsilon)
                prev_action = action

                next_state, reward, done, info = env.step(action)
                _next_state = preprocess_state(next_state)
                _next_state = torch.from_numpy(_next_state).float()
                agent.step(state_frames.__copy__(), action, reward, _next_state, done)
                state = next_state

                episodic_accum += reward
                iteration_rewards.append(reward)

                if agent.num_train_updates > 0:
                    # evaluate every 1M steps and decay epsilon (based on paper)
                    if agent.num_train_updates % evaluation_rate == 0 and prev_iteration != agent.num_train_updates:
                        epsilon = max( epsilon_min, epsilon*epsilod_decay)
                        avg_eval_score = evaluate(csv_writer=csv_writer,
                                                  iter_idx=agent.num_train_updates,
                                                  agent=agent, 
                                                  stack_size=stack_size, 
                                                  num_episodes=evaluation_num_episodes)
                        tb_writer.add_scalar('Iteration Evaluation Score', avg_eval_score, epoch_plot_count)
                        epoch_plot_count+=1
                        prev_iteration = agent.num_train_updates

                if agent.num_train_updates > num_iterations:
                    stop= True

            episode += 1
            epsidoic_rewards.append(episodic_accum)
            episodic_accum = 0.


            if episode % 100 == 0 and len(epsidoic_rewards)>20 :
                tb_writer.add_scalar('Episode Accum score', np.mean(epsidoic_rewards[-20:]), episode)
                print('iter:{}\tepisode_num:{}\tframe:{}\tepisode_score:{}\tepsilon:{}\tmemory_size:{}'.format(\
                      agent.num_train_updates, frame_count, episode, np.mean(epsidoic_rewards[-20:]), epsilon,len(agent.memory)))
                torch.save(agent.QNetwork_local.state_dict(), '{}_checkpoint.pth'.format(experiment_name)) 
    validation_file.close() 
    return epsidoic_rewards


if __name__ == '__main__':

    # read environment
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", "-e", help="atari environment name (example: SpaceInvaders-v0)", required=True)
    parser.add_argument("--name", "-n", help="experiment name", required=True)
    args = parser.parse_args()

    # load default configuration
    try:
        configuration = ConfigReader(config_dir='./configs/default.ini', environment_name = args.environment)
    except Exception as err:
        print(err)
        exit(0)

    
    
    if not os.path.exists(configuration.log_directory):
        os.makedirs(log_directroy)
    print('* Find tensorboard logs under "{}/{}"'.format(configuration.log_directory,args.name))

    env = gym.make(args.environment)
    state_space = env.observation_space
    action_space = env.action_space
    print('* Starting the experiment on enviroment:{}'.format(args.environment))

    device = torch.device(configuration.device if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    if str(device) != configuration.device:
        print(">>> {} Not Available !".format(configuration.device))
    print('* Device is set to {}'.format(device))

    run(experiment_name = args.name, 
        num_iterations = configuration.iterations, 
        learning_rate = configuration.learning_rate, 
        buffer_size = configuration.replay_memory_size, 
        batch_size = configuration.batch_size,
        gamma = configuration.gamma ,
        epsilon = configuration.epsilon, 
        epsilod_decay = configuration.epsilon_decay, 
        epsilon_min = configuration.epsilon_min, 
        stack_size = configuration.stack_size,
        device = device, 
        is_ddqn = configuration.is_ddqn, 
        evaluation_rate = configuration.evaluation_rate,
        log_directory = configuration.log_directory, 
        validation_directory=configuration.validation_directory,
        evaluation_num_episodes=configuration.evaluation_num_episodes)

    print('Finished!')

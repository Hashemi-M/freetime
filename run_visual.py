import ffmpeg
import sys
sys.path.append(r'C:\ffmpeg\bin\ffmpeg.exe')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import hydra
from omegaconf import OmegaConf
import numpy as np
from matplotlib.animation import writers
from lib.visualize import visualize
from lib.gym_windy_gridworld import WindyGridworld
import time


@hydra.main(config_path='configs', config_name='config')
def main(config): 
    
    print(OmegaConf.to_object(config))



    env = WindyGridworld(
        height=config.env.height, 
        width=config.env.width, 
        rewards=list(config.env.rewards), 
        wind=config.env.wind, 
        start=config.env.start,
        #start=(19,5),
        allowed_actions=list(config.env.allowed_actions), 
        reward_terminates_episode=config.env.reward_terminates_episode
    )
    env.reset()



    steps = 100
    i=0
    terminal_state = 1

    frames = []

    while i<steps:
        if terminal_state or i==0:
            canvas = visualize(env,1)
            frames += [canvas]
            #plt.imshow(canvas,  interpolation='nearest')
            #plt.show()
        action = np.random.choice(3)
        new_state, reward, terminal_state, info = env.step(action)
        canvas = visualize(env,1)
        frames += [canvas]
        #plt.imshow(canvas,  interpolation='nearest')
        #plt.show()

        if terminal_state:
            env.reset()

        
        
        i += 1
    
    fig, frame= generate_video(frames, "GridWorld Game")
    ani = animation.ArtistAnimation(fig, frame, interval=100, blit=True,
                                repeat_delay=1000)
    plt.show()

    Writer = writers['ffmpeg']
    writer = Writer(fps=15, metadata={'artist': 'Me'}, bitrate=1800)

    ani.save("grid.mp4",writer=writer)

    '''
    env.reset()
    canvas = visualize(env)
    plt.imshow(canvas,  interpolation='nearest')
    plt.show()
    seq = [0,0,1,1,0,0,0,2,2,2,2,2,2,1,1,1,1,2,1,1]
    for action in seq:

        new_state, reward, terminal_state, info = env.step(action)
        canvas = visualize(env)
        plt.imshow(canvas,  interpolation='nearest')
        plt.show()

        if terminal_state:
            env.reset()'''

    


def generate_video(img, title):
    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(len(img)):
        plt.imshow(img[i],animated=True)
        s = str(i*1000)
        frames.append([plt.imshow(img[i],animated=True)])
        plt.title(title)

    return fig, frames
    

@hydra.main(config_path='configs', config_name='config')
def play(config):
    
    print(OmegaConf.to_object(config))

    env = WindyGridworld(
        height=config.env.height, 
        width=config.env.width, 
        rewards=list(config.env.rewards), 
        wind=config.env.wind, 
        start=config.env.start,
        #start=(19,5),
        allowed_actions=list(config.env.allowed_actions), 
        reward_terminates_episode=config.env.reward_terminates_episode
    )
    env.reset()
    env.action_space
    frame = visualize(env,1)
    plt.ion()
 

    
    action = 'none'

    while action != 'q':
        plt.imshow(frame,  interpolation='nearest')
        action = input('Select action: W(up), A(left), D(right) or Q(quit) ')
        if action in {'q','w','d','a'}:
            if action == 'q':
                pass
            else:
                if action == 'w':
                    new_state, reward, terminal_state, info = env.step(2)
                elif action == 'a':
                    new_state, reward, terminal_state, info = env.step(0)
                elif action == 'd':
                    new_state, reward, terminal_state, info = env.step(1)

                if terminal_state:
                    c=0
                    pass

                frame = visualize(env,1)
                plt.imshow(frame)
                

                if terminal_state:
                    time.sleep(4)
                    env.reset()
                    frame = visualize(env,1)
                    plt.imshow(frame)

                
        else:
            print("invalid action!")
                
            
        



if __name__ == '__main__':
    #main()
    play()
    



            


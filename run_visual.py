

import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import numpy as np

@hydra.main(config_path='configs', config_name='config')
def main(config): 
    
    print(OmegaConf.to_object(config))

    from lib.visualize import visualize
    from lib.gym_windy_gridworld import WindyGridworld

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
    while i<steps:
        if terminal_state or i==0:
            canvas = visualize(env)
            plt.imshow(canvas,  interpolation='nearest')
            plt.show()
        action = np.random.choice(3)
        new_state, reward, terminal_state, info = env.step(action)
        canvas = visualize(env)
        plt.imshow(canvas,  interpolation='nearest')
        plt.show()

        if terminal_state:
            env.reset()

        
        
        i += 1

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
            env.reset()




    

    

if __name__ == '__main__':
    main()
    



            



import matplotlib.pyplot as plt
from lib.algorithms import  freetime_no_reward
from .utils import make_trajectory_map
import numpy as np
import os
dirname = os.path.dirname(__file__)


def plot_Q(Q, title, vmin, vmax):
    
    plt.figure()
    plt.imshow(Q.max(axis=-1), vmin=vmin, vmax=vmax, cmap='jet')
    plt.title(title)
    plt.colorbar()
    filename = os.path.join(dirname, f'plots\{title}.png')     
    plt.savefig(fname = filename, bbox_inches = 'tight')


def plot_F(F, title, vmin, vmax, action='max'): 
    plt.figure()
    if action == 'max':
        plt.imshow(F.max(axis=-1), vmin=vmin, vmax=vmax)
    elif action == 'min':
        plt.imshow(F.min(axis=-1), vmin=vmin, vmax=vmax)
    else:
        plt.imshow(F[..., action], vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar
    filename = os.path.join(dirname, f'plots\{title}.png')     
    plt.savefig(fname = filename, bbox_inches = 'tight')
  
    
def plot_errorbars(values, label):
    values = np.stack(values, axis=0)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    plt.plot(mean, label=label)
    plt.fill_between(np.arange(mean.shape[0]), mean+std, mean-std, alpha=0.5)
    plt.xlabel('number of timesteps')
    plt.ylabel('cumulative rewards')
    
    
def run(config):
    
    from .gym_windy_gridworld import WindyGridworld
    from .algorithms import Q_learn, Q_learn_freetime, build_q_table, build_f_table
    
    INITIALIZATIONS = config.initializations
    
    print('Configuring environment...')
    env = WindyGridworld(
        height=config.env.height, 
        width=config.env.width, 
        rewards=list(config.env.rewards), 
        wind=config.env.wind, 
        start=config.env.start, 
        allowed_actions=list(config.env.allowed_actions), 
        reward_terminates_episode=config.env.reward_terminates_episode
    )
    
    max_r = np.array(config.env.rewards)[:,0].max()

    # BASELINE EXPERIMENT
    
    print('Running baseline...')
    
    results_baseline = {
        initialization: [] for initialization in INITIALIZATIONS
    }
    
    Q_tables = {}
    
    for initialization in INITIALIZATIONS:
        
        for run in range(config.num_runs):
            Q = build_q_table(
                (env.height, env.width),                
                env.action_space.n, 
                initialization = initialization,
                seed = config.random_initialization_seed, # type: ignore
                max_reward = max_r
            )
            
            Q, rewards = Q_learn(
                env, 
                Q, 
                config.baseline.num_steps, 
                config.baseline.epsilon, 
                config.baseline.discount, 
                config.baseline.alpha
            )
            
            results_baseline[initialization].append(rewards)
            Q_tables[initialization] = Q


    if config.baseline.show_trajectory:
        for initialization, Q in Q_tables.items():
            make_trajectory_map(Q, env, title=f'Init {initialization} baseline trajectory', 
                                num_plots=config.trajectory_maps.num_plots)
    
    if config.baseline.show_rewards:
        plt.figure()
        for title, rewards in results_baseline.items():
            plot_errorbars(rewards, label=title)
            plt.legend()
            plt.title('baseline results')
    
    if config.baseline.show_q: 
        for title, Q in Q_tables.items():
            plot_Q(Q, f'{title} init q-table baseline', config.q_plots.vmin, config.q_plots.vmax)
 
    print('Running freetime')
    F_INITIALIZATIONS = config.f_initializations
    
    results_freetime = {
        initialization: {f_initialization: [] for f_initialization in F_INITIALIZATIONS} for initialization in INITIALIZATIONS
    }
    Q_tables = {
        initialization: {f_initialization: [] for f_initialization in F_INITIALIZATIONS} for initialization in INITIALIZATIONS
    }
    F_tables = {
        initialization: {f_initialization: [] for f_initialization in F_INITIALIZATIONS} for initialization in INITIALIZATIONS
    }
    
    # For each init of the Q table
    for initialization in INITIALIZATIONS:
        # For each init of F table
        for f_initialization in F_INITIALIZATIONS:

            for run in range(config.num_runs):
                # Build Q table
                Q = build_q_table(
                    (env.height, env.width),                
                    env.action_space.n, 
                    initialization = initialization, 
                    seed = config.random_initialization_seed, # type: ignore
                    max_reward = max_r
                )
                # Build F table
                F = build_f_table(
                    Q,
                    init = f_initialization
                )
                
                Q, F, rewards, _ = Q_learn_freetime(
                    env, 
                    Q,
                    F,
                    config.freetime.num_steps, 
                    config.freetime.epsilon, 
                    config.freetime.discount, 
                    config.freetime.alpha, 
                    config.freetime.alpha_f, 
                    config.freetime.tolerance
                )
                
                results_freetime[initialization][f_initialization].append(rewards)
                Q_tables[initialization][f_initialization] = Q
                F_tables[initialization][f_initialization] = F
        
    '''
    if config.freetime.show_trajectory: 
        for initialization, Q in Q_tables.items():
            make_trajectory_map(Q, env, title=f'Init {initialization} freetime trajectory', 
                                num_plots=config.trajectory_maps.num_plots)'''
    '''
    if config.freetime.show_rewards: 
        plt.figure()
        for title, rewards in results_freetime.items():
            plot_errorbars(rewards, label=title)
            plt.legend()
            plt.title('freetime results_freetime')'''
        
    if config.freetime.show_q:
        for title, f_init in Q_tables.items():
            for init, Q in f_init.items():
                plot_Q(Q, f'{title} init q-table with freetime-{init}', config.q_plots.vmin, config.q_plots.vmax)

        
    if config.freetime.show_f:
        for action in config.freetime.show_f_actions:
            for title, f_init in F_tables.items():
                for init, F in f_init.items():
                    plot_F(F, f'{title} init F-table action {action} with freetime-{init}', config.f_plots.vmin, config.f_plots.vmax, 
                       action)
    if config.plot_freetime_vs_baseline_same_table:
        for initialization in INITIALIZATIONS:
            plt.figure()
            plot_errorbars(results_baseline[initialization], label='baseline')
            for f_init in F_INITIALIZATIONS:

                plot_errorbars(results_freetime[initialization][f_init], label= f'freetime-{f_init}')
                
            plt.title(f'{initialization} initialization')
            plt.legend()
            filename = os.path.join(dirname, f'plots\{initialization}_scores.png')

            plt.savefig(fname = filename, bbox_inches = 'tight')
        
       
    plt.show()
    
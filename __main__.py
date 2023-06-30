from grid import Grid
from sarsa import sarsa
from plots import *
from update import *
import numpy as np
import sys


if __name__ == "__main__":
    sim_speed = 0.05

    # Test run
    showRender=True
    episodes=5
    renderEveryNth=5
    printEveryNth=1
    do_plot_rewards=True

    # Full run
    """
    showRender=False
    episodes=2000
    renderEveryNth=10000
    printEveryNth=100
    do_plot_rewards=True
    """

    """
    

    if(len(sys.argv)>1):
        episodes = int(sys.argv[1])
    if(len(sys.argv)>2):
        showRender = sys.argv[2] in ['true','True','T','t']
    if(len(sys.argv)>3):
        datafile = sys.argv[3]
    """

    # Start and End Point
    start_port = [0,0]
    end_port = [9,9]

    #10x10 Environment
    islands = np.array([[7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7],[7,8],[2,8],[3,8],[4,8],[5,8],[6,8]])
    rough_waters = np.array([[0,7],[1,7],[2,7],[0,9],[1,9],[2,9],[0,8],[1,8]])

    # Empty list for experiment results
    experiments = []

    # SARSA
    env = Grid(start_port,end_port,islands,rough_waters)
    algo = sarsa(actions=list(range(env.n_actions)))
    data={}
    env.after(10, env_update(env, algo, data, episodes))
    env.mainloop()
    experiments.append((env,algo, data))


    print("All experiments complete")

    for env, RL, data in experiments:
        print("{} : max reward = {} medLast100={} varLast100={}".format(RL.display_name, np.max(data['global_reward']),np.median(data['global_reward'][-100:]), np.var(data['global_reward'][-100:])))
    
    plot_rewards(experiments)
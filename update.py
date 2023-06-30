import numpy as np

def debug(debuglevel, msg, DEBUG=1, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg) 
        else:
            print(msg) 

def env_update(env, RL, data, episodes=50, sim_speed = 0.05, showRender=True, renderEveryNth=5, printEveryNth=1):
    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward

    for episode in range(episodes): 
        t=0
        # initial state
        if episode == 0:
            state = env.reset(value = 0)
        else:
            state = env.reset()
       
        debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))

        # RL choose action based on state
        action = RL.choose_action(str(state))
        while True:
            # fresh env
            #if(t<5000 and (showRender or (episode % renderEveryNth)==0)):
            if(showRender or (episode % renderEveryNth)==0):
                env.render(sim_speed)


            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2,'reward_{}=  total return_t ={} Mean50={}'.format(reward, global_reward[episode],np.mean(global_reward[-50:])))
            

            # RL learn from this transition
            # and determine next state and action
            state, action =  RL.learn(str(state), action, reward, str(state_))


            # break while loop when end of this episode
            if done:
                break
            else:
                t=t+1

        debug(1,"({}) Episode {}: Length={}  Total return = {} ".format(RL.display_name,episode, t,  global_reward[episode],global_reward[episode]),printNow=(episode%printEveryNth==0))
        if(episode>=100):
            debug(1,"    Median100={} Variance100={}".format(np.median(global_reward[episode-100:episode]),np.var(global_reward[episode-100:episode])),printNow=(episode%printEveryNth==0))
    # end of game
    print('game over -- Algorithm {} completed'.format(RL.display_name))
    env.destroy()
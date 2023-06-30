import matplotlib.pyplot as plt

def plot_rewards(experiments):
    color_list=['orange','green','red','black','magenta']
    label_list=[]
    for i, (env, RL, data) in enumerate(experiments):
        x_values=range(len(data['global_reward']))
        label_list.append(RL.display_name)
        y_values=data['global_reward']
        plt.plot(x_values, y_values, c=color_list[i],label=label_list[-1])
        plt.legend(label_list)
    plt.title("Reward Progress", fontsize=24)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Return", fontsize=18)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)
    #plt.axis([0, 1100, 0, 1100000])
    plt.savefig('plots/sarsa.png')
    plt.show()
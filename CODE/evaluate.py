import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves():
    try:
        data = np.load('eval_results.npz')
    except FileNotFoundError:
        print("Error: 'eval_results.npz' not found. Please run train.py first.")
        return

    episodes = data['episodes']
    ego_rewards = data['ego_rewards']
    adv_rewards = data['adv_rewards']

    plt.figure(figsize=(10, 6))
    
    plt.plot(episodes, ego_rewards, label='Ego Agent (QRM-SG)', color='black', linestyle='--', linewidth=2)
    plt.plot(episodes, adv_rewards, label='Adv Agent (QRM-SG)', color='gray', linestyle='--', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward of Agent')
    plt.title('Case Study I: Evaluation Performance')
    plt.ylim([-0.05, 1.05])
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()
    a = np.array([1,2,3,4])
    b = np.array([5,6,7,8])
    print(a*b)
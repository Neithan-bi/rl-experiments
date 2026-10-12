import gymnasium as gym
import numpy as np
import pandas as pd
from collections import defaultdict

# ==========================================
# COMMON INITIALIZATION
# ==========================================
env = gym.make('Blackjack-v1')


# ==========================================
# PART 1: FIRST-VISIT MC PREDICTION
# ==========================================
def sample_policy(state):
    """Stick (0) if score is >= 20, else Hit (1)"""
    player_sum, _, _ = state
    return 0 if player_sum >= 20 else 1

def generate_prediction_episode(policy, env):
    episode = []
    state, info = env.reset()
    done = False
    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state
    return episode

def first_visit_mc_prediction(env, policy, num_episodes=500000):
    V = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    for _ in range(num_episodes):
        episode = generate_prediction_episode(policy, env)
        states_in_episode = [step[0] for step in episode]
        
        G = 0
        for t in reversed(range(len(episode))):
            state, _, reward = episode[t]
            G += reward
            
            # First-visit check for the state
            if state not in states_in_episode[:t]:
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]
    return V


# ==========================================
# PART 2: MC ON-POLICY CONTROL
# ==========================================
def epsilon_greedy_policy(state, Q, epsilon, n_actions):
    """Explores randomly with probability epsilon, otherwise exploits max Q-value"""
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    return np.argmax(Q[state])

def generate_control_episode(Q, env, epsilon, n_actions):
    episode = []
    state, info = env.reset()
    done = False
    while not done:
        action = epsilon_greedy_policy(state, Q, epsilon, n_actions)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state
    return episode

def mc_on_policy_control(env, num_episodes=50000, epsilon=0.15):
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))
    
    for _ in range(num_episodes):
        episode = generate_control_episode(Q, env, epsilon, n_actions)
        state_action_in_episode = [(step[0], step[1]) for step in episode]
        
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G += reward
            
            # First-visit check for the state-action pair
            if (state, action) not in state_action_in_episode[:t]:
                returns_sum[state][action] += G
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]
    return Q


# ==========================================
# EXECUTION & OUTPUT
# ==========================================
if __name__ == "__main__":
    # --- Execute Prediction ---
    print("--- 1. Running First-Visit MC Prediction (500,000 episodes) ---")
    value_table = first_visit_mc_prediction(env, sample_policy, num_episodes=500000)
    
    print("\nSample Values from the Prediction Value Table:")
    for state, val in list(value_table.items())[:5]:
        print(f"State {state} -> Value: {val:.4f}")
        
    # --- Execute Control ---
    print("\n--- 2. Running On-Policy MC Control (50,000 episodes) ---")
    Q_table = mc_on_policy_control(env, num_episodes=50000, epsilon=0.15)
    
    # Store and Display the Q-values in a DataFrame
    df_data = []
    for state, actions in Q_table.items():
        df_data.append({
            "Player Sum": state[0],
            "Dealer Card": state[1],
            "Usable Ace": state[2],
            "Q(Stick)": actions[0],
            "Q(Hit)": actions[1],
            "Optimal Action": "Stick" if np.argmax(actions) == 0 else "Hit"
        })
        
    df = pd.DataFrame(df_data)
    print("\nFirst few rows of the analyzed Q-values DataFrame:")
    print(df.head(10))

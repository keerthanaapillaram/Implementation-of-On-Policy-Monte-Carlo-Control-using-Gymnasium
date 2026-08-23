# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

#### Software
```
Python 3.x
Jupyter Notebook / Google Colab / VS Code
Gymnasium
NumPy
Matplotlib
```

#### Installation
```
pip install gymnasium numpy matplotlib
```


## Environment Description

The experiment uses the FrozenLake-v1 environment from Gymnasium.

The default 4 × 4 FrozenLake environment is:
```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```
Where:

**S**  = Starting state  
**F**  = Frozen surface  
**H**  = Hole  
**G**  = Goal

The environment contains:

* **16 states**
* **4 actions**

Actions

| Action Number | Direction |
| ------ | ------- |
| 0      | Left    |
| 1      | Down    |
| 2      | Right   |
| 3      | Up      |

The agent receives a reward of 1 when it reaches the goal. Other transitions generally provide a reward of 0.

In this experiment:

is_slippery=False

is used so that the environment is deterministic and the effect of the learned policy can be clearly observed.


## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

1. Initialize the FrozenLake-v1 environment with the required states and actions.
2. Initialize the Q-table Q(s,a) with zeros for all state-action pairs.
3. Set the hyperparameters: number of episodes, learning rate α, discount factor γ, and exploration rate ε.
4. Reset the environment to obtain the initial state.
5. Use an epsilon-greedy policy to select an action based on the current Q-values.
6. Execute the selected action and observe the next state and reward.
7. Store the state, action, and reward as part of the current episode.
8. Continue selecting actions and collecting experiences until the episode terminates.
9. Initialize the return G to zero after completing the episode.
10. Traverse the stored episode in reverse order to calculate the return for each state-action pair.
11. Update Q(s,a) using the Monte Carlo incremental update rule.
12. Gradually decrease ε to reduce exploration and increase exploitation during training.
13. Repeat the episode generation and Q-value update process for the specified number of episodes.
14. Derive the state-value function V(s) and the final greedy policy from the learned Q-table.
15. Evaluate the learned policy, calculate the average reward, and plot the learning curve to analyze the performance.


## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of States:", n_states)
print("Number of Actions:", n_actions)


# Parameters
num_episodes = 5000
alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9997

print("\nnum_episodes =", num_episodes)


# Initialize Q-table
Q = np.zeros((n_states, n_actions))


# Epsilon-greedy action selection
def choose_action(state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()

    max_q = np.max(Q[state])
    best_actions = np.flatnonzero(Q[state] == max_q)

    return np.random.choice(best_actions)


# Training
episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()

    episode_data = []

    terminated = False
    truncated = False

    total_reward = 0

    # Generate complete episode
    while not (terminated or truncated):

        action = choose_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode_data.append((state, action, reward))

        total_reward += reward

        state = next_state

    # Calculate Monte Carlo returns
    G = 0

    for state, action, reward in reversed(episode_data):

        G = reward + gamma * G

        Q[state, action] += alpha * (G - Q[state, action])

    # Epsilon decay
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    episode_rewards.append(total_reward)


# Display Q-table
np.set_printoptions(precision=3, suppress=True)

print("\nFinal Q-table:")
print(Q)


# Estimated State-Value Function
V = np.max(Q, axis=1)

print("\nEstimated State-Value Function:")
print(V.reshape(4, 4))


# Learned Policy
policy = np.argmax(Q, axis=1)

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

learned_policy = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nLearned Policy:")
print(learned_policy)


# Average reward over last 1000 episodes
average_reward = np.mean(episode_rewards[-1000:])

print(
    "\nAverage reward over last 1000 episodes:",
    round(average_reward, 2)
)


# Learning Curve
window = 100

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(10, 5))

plt.plot(
    moving_average,
    label="Average Reward"
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("On-Policy Monte Carlo Control Learning Curve")

plt.grid(True)
plt.legend()

plt.show()


# Evaluate learned policy
evaluation_episodes = 1000
successes = 0

for episode in range(evaluation_episodes):

    state, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Greedy action
        action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state

        if reward == 1:
            successes += 1


success_rate = successes / evaluation_episodes

print(
    "\nEvaluation Success Rate:",
    round(success_rate, 3)
)

print(
    "Evaluation Success Percentage:",
    round(success_rate * 100, 2),
    "%"
)


# Close environment
env.close()



```

## Output

### num_episodes =5000


Final Q-table:

<img width="694" height="346" alt="image" src="https://github.com/user-attachments/assets/f538a523-d8f8-4387-b303-c6618d27f061" />


Estimated State-Value Function:

<img width="636" height="108" alt="image" src="https://github.com/user-attachments/assets/c761d493-88dc-4489-baa4-a41c5421713a" />


Learned Policy:

<img width="598" height="108" alt="image" src="https://github.com/user-attachments/assets/f01743c6-ef5f-4a4b-a10a-9a2fa626acf8" />


Average reward over last 1000 episodes: 

<img width="455" height="42" alt="image" src="https://github.com/user-attachments/assets/d48f8760-65c3-4a32-a8ba-004ce67b56fd" />

Manto Carlo Learning curve :

<img width="1218" height="532" alt="image" src="https://github.com/user-attachments/assets/38d77d52-039f-4bf8-9730-88af9210df64" />


### num_episodes =20000


Final Q-table:

<img width="705" height="346" alt="image" src="https://github.com/user-attachments/assets/52218010-084c-47ca-8d34-59ebcef31652" />

Estimated State-Value Function:

<img width="664" height="117" alt="image" src="https://github.com/user-attachments/assets/f6ce3769-90aa-4ad6-9b3d-68eb08d84d42" />


Learned Policy:

<img width="519" height="109" alt="image" src="https://github.com/user-attachments/assets/52b09051-7824-4d99-a105-c6ec5b804bb2" />

Average reward over last 1000 episodes: 

<img width="538" height="41" alt="image" src="https://github.com/user-attachments/assets/8d1f7837-67fe-43ab-834c-61af2e8883be" />


Manto Carlo Learning curve :

<img width="1002" height="467" alt="image" src="https://github.com/user-attachments/assets/c69cf78e-3fb9-46fa-b791-9315bb11429e" />


## Result

The On-Policy Monte Carlo Control method was successfully implemented in the FrozenLake-v1 environment. The agent learned the action-value function Q(s,a) from complete episodes using an epsilon-greedy policy. The average reward over the last 1,000 episodes was 0.69 for 5,000 training episodes and 0.95 for 20,000 training episodes, indicating successful policy learning.

## Inference

The experimental comparison shows that increasing the training episodes led to better agent performance. The average reward improved from 0.69 to 0.95 when the training duration increased from 5,000 to 20,000 episodes. This indicates that additional experience helped the agent refine its Q-values and make more effective decisions. Therefore, 20,000 episodes provided better learning performance and a more reliable policy than 5,000 episodes.

---


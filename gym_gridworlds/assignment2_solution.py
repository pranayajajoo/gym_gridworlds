import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated





# computing v
def v_compute(policy,R,P,V_initial,gamma):
    V_calculated = np.sum(policy * (R + gamma * np.matmul(P,V_initial)*(1-T)), axis = 1)
    return V_calculated

# policy evaluation with state value function
def bellman_v(policy, gamma_test, V_test, error_threshold=0):
    iteration_count = 0 
    bellman_error_log = []
    while True:
        V_calculated = v_compute(policy, R, P, V_test, gamma_test)
        error = np.sum(np.absolute(V_test - V_calculated))
        bellman_error_log.append(error)
        V_test = V_calculated
        # print(iteration_count, error, V_test)
        iteration_count += 1
        if error <= error_threshold: break
    return bellman_error_log, V_calculated



# computing q
def q_compute(policy,R,P,Q_initial,gamma):
    Q_calculated = R + gamma * np.matmul(P, (np.sum((policy * Q_initial), axis = 1)).reshape(-1,1)).reshape(P.shape[0],P.shape[1]) * (1-T)
    return Q_calculated

# policy evaluation with state, action value function
def bellman_q(policy, gamma_test, Q_test, error_threshold=0):
    iteration_count = 0 
    bellman_error_log = []
    while True:    
        Q_calculated = q_compute(policy, R, P, Q_test, gamma_test)
        error = np.sum(np.absolute(Q_test - Q_calculated))
        bellman_error_log.append(error)
        Q_test = Q_calculated
        # print(iteration_count, error, Q)
        iteration_count += 1
        if error <= error_threshold: break
    return bellman_error_log, Q_calculated




#defining optimal policy
optimal_policy = policy = np.array([[0., 1., 0., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 0., 0., 1.],
                                    [0., 1., 0., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 0., 1., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 0., 1., 0.]])



# graphs
gammas = [0.01, 0.5, 0.99]
for init_value in [-10, 0, 10]:
    fig, axs = plt.subplots(2, len(gammas))
    fig.suptitle(f"$V_0$: {init_value}")
    for i, gamma in enumerate(gammas):
        bellman_error_log_v, V_calculated = bellman_v(policy = optimal_policy, gamma_test=gamma, V_test=np.ones(n_states)*init_value)
        V_grid = V_calculated.reshape(3, 3)
        axs[0][i].imshow(V_grid)
        for x in range(V_grid.shape[0]):
            for y in range(V_grid.shape[1]):
                axs[0][i].text(y, x, f'{V_grid[x, y]:.2f}', ha='center', va='center', color='red')
        
        axs[1][i].plot(bellman_error_log_v)
        axs[0][i].set_title(f'$\gamma$ = {gamma}')
    fig.text(0.5, 0.02, 'Bellman error for state value function', ha='center', va='center')

    fig, axs = plt.subplots(n_actions + 1, len(gammas))
    fig.suptitle(f"$Q_0$: {init_value}")
    for i, gamma in enumerate(gammas):
        bellman_error_log_q, Q_calculated = bellman_q(policy = optimal_policy, gamma_test=gamma, Q_test=np.ones((n_states,n_actions))*init_value)
        for a in range(n_actions):
            Q_grid = np.transpose(Q_calculated)[a].reshape(3,3)
            axs[a][i].imshow(Q_grid)
            for x in range(Q_grid.shape[0]):
                for y in range(Q_grid.shape[1]):
                    axs[a][i].text(y, x, f'{Q_grid[x, y]:.2f}', ha='center', va='center', color='red', fontsize = 4)
        axs[-1][i].plot(bellman_error_log_q)
        axs[0][i].set_title(f'$\gamma$ = {gamma}')
    fig.text(0.5, 0.02, 'Bellman error for action value function', ha='center', va='center')

    plt.show()


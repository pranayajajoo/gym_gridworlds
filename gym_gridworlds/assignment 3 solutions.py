import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import copy

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




# defining random policy
start_policy = np.ones((n_states,n_actions))*(1/n_actions)
optimal_policy = policy = np.array([[0., 1., 0., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 0., 0., 1.],
                                    [0., 1., 0., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 0., 1., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 0., 1., 0.]])



### POLICY ITERATION ###



# computing v
def v_compute(policy,R,P,V_initial,gamma):
    V_calculated = np.sum(policy * (R + gamma * np.matmul(P,V_initial)*(1-T)), axis = 1)
    return V_calculated

# policy evaluation with state value function
def v_policy_evaluation(policy, gamma_test, V_test, bellman_error_log, iteration_evaluation, error_threshold):
    while True:
        V_calculated = v_compute(policy, R, P, V_test, gamma_test)
        error = np.sum(np.absolute(V_test - V_calculated))
        bellman_error_log.append(error)
        V_test = V_calculated
        # print(iteration_count, error, V_test)
        iteration_evaluation += 1
        if error <= error_threshold: break
    return bellman_error_log, V_calculated, iteration_evaluation


def v_policy_improvement(V, policy, T, gamma):
    start_policy = copy.deepcopy(policy)
    
    q_sa = R + gamma * (np.matmul(P,V) * (1-T))
    # max_action = np.argmax(q_sa, -1) # does not work if there is more than one optimal action
    max_action = np.argmax(q_sa, -1)

    policy = np.zeros((n_states, n_actions))
    for i in range(len(max_action)):
        policy[i,max_action[i]] = 1

    is_stable = np.allclose(start_policy, policy)
    return policy, is_stable


def v_policy_iteration(V, policy, T, gamma, error_threshold):
    bellman_error_log = []
    iteration_evaluation = 0
    while True:
        bellman_error_log, V_calculated, iteration_evaluation = v_policy_evaluation(policy, gamma, V, bellman_error_log, iteration_evaluation, error_threshold)
        updated_policy, is_stable = v_policy_improvement(V_calculated, policy, T, gamma)
        
        V = V_calculated
        policy = updated_policy
    
        if is_stable == True: break
    
    assert np.allclose(optimal_policy, policy)
    return policy, iteration_evaluation, bellman_error_log


# computing q
def q_compute(policy,R,P,Q_initial,gamma):
    Q_calculated = R + gamma * np.matmul(P, np.sum((policy * Q_initial), axis = 1).reshape(-1,1)).reshape(P.shape[0],P.shape[1]) * (1-T)
    return Q_calculated

# policy evaluation with action value function
def q_policy_evaluation(policy, gamma_test, Q_test, bellman_error_log, iteration_evaluation, error_threshold):
    while True:    
        Q_calculated = q_compute(policy, R, P, Q_test, gamma_test)
        error = np.sum(np.absolute(Q_test - Q_calculated))
        bellman_error_log.append(error)
        Q_test = Q_calculated
        # print(iteration_count, error, Q)
        iteration_evaluation += 1
        if error <= error_threshold: break
    return bellman_error_log, Q_calculated, iteration_evaluation


def q_policy_improvement(Q, policy, T, gamma):
    start_policy = copy.deepcopy(policy)

    q_sa = R + (gamma * np.matmul(P,np.sum((Q*policy),axis = 1)) * (1-T))

    max_action = np.argmax(q_sa, -1)

    policy = np.zeros((n_states, n_actions))
    for i in range(len(max_action)):
        policy[i,max_action[i]] = 1

    is_stable = np.allclose(start_policy, policy)
    return policy, is_stable


def q_policy_iteration(Q, policy, T, gamma, error_threshold):
    bellman_error_log = []
    iteration_evaluation = 0
    while True:
        bellman_error_log, Q_calculated, iteration_evaluation = q_policy_evaluation(policy, gamma, Q, bellman_error_log, iteration_evaluation, error_threshold)
        updated_policy, is_stable = q_policy_improvement(Q, policy, T, gamma)
        
        Q = Q_calculated
        policy = updated_policy
        

        if is_stable == True: break

    assert np.allclose(optimal_policy, policy)
    return policy, iteration_evaluation, bellman_error_log




### VALUE ITERATION ###


# value iteration with v
def v_value_iteration(V, T, gamma, error_threshold):
    iteration_count = 0
    bellman_error_log = []
    while True:
        V_start = copy.deepcopy(V)
        V = np.max((R + gamma * np.matmul(P,V_start)*(1-T)), axis = 1)

        error = np.max(np.abs(V_start - V))
        bellman_error_log.append(error)
        V_start = V
        # print(iteration_count, error, V)
        iteration_count += 1        
        if error <= error_threshold: break
        
    q_sa = R + gamma * (np.matmul(P,V) * (1-T))
    max_action = np.argmax(q_sa, -1)
    policy = np.identity(n_actions)[max_action]
    
    assert np.allclose(optimal_policy, policy)
    return policy, iteration_count, bellman_error_log


# value iteration with q
def q_value_iteration(Q, V, T, gamma, error_threshold):
    bellman_error_log = []
    iteration_count = 0

    while True:
        Q_start = copy.deepcopy(Q)        

        Q = R + gamma * np.matmul(P, V) * (1-T)

        V = np.max(Q, axis = 1)

        error = np.max(np.abs(Q_start - Q))
        bellman_error_log.append(error)
        Q_start = Q
        iteration_count += 1        
        if error <= error_threshold: break
        
    q_sa = R + gamma * (np.matmul(P,V) * (1-T))
    max_action = np.argmax(q_sa, -1)
    policy = np.identity(n_actions)[max_action]

    assert np.allclose(optimal_policy, policy)
    return policy, iteration_count, bellman_error_log



### GENERALIZED POLICY ITERATION ###


def gpi_v_compute(policy,R,P,V_initial,gamma, bellman_error_log, iteration_evaluation):
    V_calculated = np.sum(policy * (R + gamma * np.matmul(P,V_initial)*(1-T)), axis = 1)
    error = np.sum(np.absolute(V_initial - V_calculated))
    bellman_error_log.append(error)
    iteration_evaluation += 1
    return V_calculated, bellman_error_log, iteration_evaluation

def v_generalized_policy_iteration(V, policy, T, gamma, error_threshold):
    bellman_error_log = []
    iteration_evaluation = 0
    V_old = copy.deepcopy(V)
    while True:
        for i in range(5):
            V, bellman_error_log, iteration_evaluation = gpi_v_compute(policy,R,P,V,gamma, bellman_error_log, iteration_evaluation)
        updated_policy, is_stable = v_policy_improvement(V, policy, T, gamma)
        
        v_stable = np.allclose(V_old, V)
        pi_stable = np.allclose(policy, updated_policy)
        
        if (v_stable == True and pi_stable == True): break

        policy = updated_policy
        V_old = V
    
    assert np.allclose(optimal_policy, policy)
    return policy, iteration_evaluation, bellman_error_log


def gpi_q_compute(policy,R,P,Q_initial,gamma, bellman_error_log, iteration_evaluation):
    Q_calculated = R + gamma * np.matmul(P, np.sum((policy * Q_initial), axis = 1).reshape(-1,1)).reshape(P.shape[0],P.shape[1]) * (1-T)
    error = np.sum(np.absolute(Q_initial - Q_calculated))
    bellman_error_log.append(error)
    iteration_evaluation += 1
    return Q_calculated, bellman_error_log, iteration_evaluation

def q_generalized_policy_iteration(Q, policy, T, gamma, error_threshold):
    bellman_error_log = []
    iteration_evaluation = 0
    Q_old = copy.deepcopy(Q)
    while True:
        for i in range(5):
            Q, bellman_error_log, iteration_evaluation = gpi_q_compute(policy, R, P, Q, gamma, bellman_error_log, iteration_evaluation)
        updated_policy, is_stable = q_policy_improvement(Q, policy, T, gamma)
        
        q_stable = np.allclose(Q_old, Q)
        pi_stable = np.allclose(policy, updated_policy)
        
        if (q_stable == True and pi_stable == True): break

        policy = updated_policy
        Q_old = Q
    
    assert np.allclose(optimal_policy, policy)
    return policy, iteration_evaluation, bellman_error_log




### GRAPHS ###

init = [-100, -10, -5, 0, 5, 10, 100]
v_totaliterations_pi = []
v_totaliterations_vi = []
v_totaliterations_gpi = []


for initialization in [-100, -10, -5, 0, 5, 10, 100]:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
    # for j in [v_policy_iteration, v_value_iteration, v_generalized_policy_iteration]:
    Q = np.ones((n_states, n_actions)) * initialization
    V = np.ones((n_states)) * initialization
    gamma = 0.99
    error_threshold = 1e-4

    policy, iteration_evaluation, bellman_error_log = v_policy_iteration(V, start_policy, T, gamma, error_threshold)
    v_totaliterations_pi.append(len(bellman_error_log))
    ax1.plot(bellman_error_log, color='blue')
    ax1.set_title(f'PI. V = {initialization}. #iterations = {len(bellman_error_log)}', fontsize = 15)
    
    policy, iteration_count, bellman_error_log= v_value_iteration(V, T, gamma, error_threshold)
    v_totaliterations_vi.append(len(bellman_error_log))
    ax2.plot(bellman_error_log, color='green')
    ax2.set_title(f'VI. V = {initialization}. #iterations = {len(bellman_error_log)}', fontsize = 15)

    policy, iteration_evaluation, bellman_error_log = v_generalized_policy_iteration(V, start_policy, T, gamma, error_threshold)
    v_totaliterations_gpi.append(len(bellman_error_log))
    ax3.plot(bellman_error_log, color='red')
    ax3.set_title(f'GPI. V = {initialization}. #iterations = {len(bellman_error_log)}', fontsize = 15)

    fig.savefig(f"V_{initialization}.png")


init = [-100, -10, -5, 0, 5, 10, 100]
q_totaliterations_pi = []
q_totaliterations_vi = []
q_totaliterations_gpi = []

for initialization in [-100, -10, -5, 0, 5, 10, 100]:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
    # for j in [v_policy_iteration, v_value_iteration, v_generalized_policy_iteration]:
    Q = np.ones((n_states, n_actions)) * initialization
    V = np.ones((n_states)) * initialization
    gamma = 0.99
    error_threshold = 1e-4

    policy, iteration_evaluation, bellman_error_log = q_policy_iteration(Q, start_policy, T, gamma, error_threshold)
    q_totaliterations_pi.append(len(bellman_error_log))
    ax1.plot(bellman_error_log, color='blue')
    ax1.set_title(f'PI. Q = {initialization}. #iterations = {len(bellman_error_log)}', fontsize = 15)
    
    policy, iteration_count, bellman_error_log = q_value_iteration(Q, V, T, gamma, error_threshold)
    q_totaliterations_vi.append(len(bellman_error_log))
    ax2.plot(bellman_error_log, color='green')
    ax2.set_title(f'VI. Q = {initialization}. #iterations = {len(bellman_error_log)}', fontsize = 15)

    policy, iteration_evaluation, bellman_error_log = q_generalized_policy_iteration(Q, start_policy, T, gamma, error_threshold)
    q_totaliterations_gpi.append(len(bellman_error_log))
    ax3.plot(bellman_error_log, color='red')
    ax3.set_title(f'GPI. Q = {initialization}. #iterations = {len(bellman_error_log)}', fontsize = 15)

    fig.savefig(f"Q_{initialization}.png")
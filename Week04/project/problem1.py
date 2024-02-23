import numpy as np

# Setting parameters
sigma = 0.2  # Standard deviation of returns
P_t_minus_1 = 100  # Price at time t-1
n_simulations = 10000  # Number of simulations

# Generating returns following a normal distribution
r = np.random.normal(0, sigma, n_simulations)

# Simulating the three types of returns
# Classical Brownian Motion
P_t_classical = P_t_minus_1 + r

# Arithmetic Return System
P_t_arithmetic = P_t_minus_1 * (1 + r)

# Log Return or Geometric Brownian Motion
P_t_geometric = P_t_minus_1 * np.exp(r)

# Calculating mean and standard deviation for each type
mean_classical = np.mean(P_t_classical)
std_classical = np.std(P_t_classical)

mean_arithmetic = np.mean(P_t_arithmetic)
std_arithmetic = np.std(P_t_arithmetic)

mean_geometric = np.mean(P_t_geometric)
std_geometric = np.std(P_t_geometric)

# Printing the results
print("Mean and standard deviation for Classical Brownian Motion:", mean_classical, std_classical)
print("Mean and standard deviation for Arithmetic Return System:", mean_arithmetic, std_arithmetic)
print("Mean and standard deviation for Log Return or Geometric Brownian Motion:", mean_geometric, std_geometric)

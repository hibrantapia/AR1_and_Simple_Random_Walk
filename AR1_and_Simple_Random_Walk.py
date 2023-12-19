import numpy as np
import matplotlib.pyplot as plt

M = 10000
N = 51  # length of each embodiment
a = 0.5
r = 1

X = np.zeros((M, N))
Z = np.random.normal(0, r, (M, N))

# Simulation of our X_n
for t in range(1, N):
    X[:, t] = a * X[:, t - 1] + Z[:, t]

fig = plt.figure(figsize = (10, 6), facecolor = 'black')
ax = fig.add_subplot(111)

neon_colors = ['#ff00ff', '#00ffff', '#ff99cc', '#ccff66', '#ff6699', '#99ffcc', '#ffff00', '#ffcc00', '#cc99ff', '#ff0000']

for i in range(10):

    ax.plot(X[i, :], label = f"Path {i + 1}", color = neon_colors[i])

ax.set_title("Sample Roads of the AR(1)")
ax.set_xlabel("Time (n)")
ax.set_ylabel("$X_n$")
ax.legend()
ax.grid(True)
ax.set_facecolor('black')
plt.show()

steps = [1, 10, 20, 30, 40, 50] # Selected time steps to display histograms

plt.figure(figsize = (12, 8))
plt.style.use('dark_background')

neon_colors = ['#ff00ff', '#00ffcc', '#ffcc00', '#ff0066', '#cc00ff', '#00ccff']

for i, step in enumerate(steps):
    plt.subplot(2, 3, i + 1)
    plt.hist(X[:, step], bins = 50, density = True, color = neon_colors[i], edgecolor = 'black')
    plt.title(f"Histogram over time, step {step}")
    plt.xlabel("$X_n$")
    plt.ylabel("Density")
    plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)

plt.tight_layout()
plt.show()

E_X_t = X.mean(axis = 0)
E_X_t

plt.figure(figsize = (15, 5))
plt.plot(E_X_t, color = 'cyan')
plt.axhline(0, color = 'red', linestyle = '--')
plt.title("$E[X_t]$")
plt.xlabel("Time ($t$)")
plt.ylabel("$E[X_t]$")
plt.grid(True)
plt.show()

Var_X_t = X.var(axis = 0)
Var_X_t

plt.figure(figsize = (15, 5))
plt.plot(Var_X_t, color = 'cyan')
plt.axhline(1.33333, color = 'red', linestyle='--', label = "Theoretical: 1.33333")
plt.title("$Var[X_t]$")
plt.xlabel("Time ($t$)")
plt.ylabel("$Var[X_t]$")
plt.legend()
plt.grid(True)
plt.show()

Cov_Xt_y_Xtprima = np.zeros((N, N)) # 51x51

for t in range(N):
    for t_prima in range(N):
        Cov_Xt_y_Xtprima[t, t_prima] = np.mean(X[:, t] * X[:, t_prima]) - np.mean(X[:, t]) * np.mean(X[:, t_prima])

Cov_Xt_y_Xtprima

plt.imshow(Cov_Xt_y_Xtprima, cmap = 'viridis')
plt.colorbar()
plt.title("$Cov[X_t, X_t']$")
plt.xlabel("Time ($t'$)")
plt.ylabel("Time ($t$)")
plt.tight_layout()
plt.show()

M = 10000
N = 101  # length of each realization, I increased it to 100 so that the mess that is made is more noticeable hehe
a = 1
r = 1

X = np.zeros((M, N))
Z = np.random.normal(0, r, (M, N))

# Simulation of our X_n
for t in range(1, N):
    X[:, t] = a * X[:, t - 1] + Z[:, t]

fig = plt.figure(figsize = (10, 6), facecolor = 'black')
ax = fig.add_subplot(111)

neon_colors = ['#ff00ff', '#00ffff', '#ff99cc', '#ccff66', '#ff6699', '#99ffcc', '#ffff00', '#ffcc00', '#cc99ff', '#ff0000']

for i in range(10):

    ax.plot(X[i, :], label = f"Path {i + 1}", color = neon_colors[i])

ax.set_title("AR(1) Sample Paths? or Random Walk?")
ax.set_xlabel("Time (n)")
ax.set_ylabel("$X_n$")
ax.legend()
ax.grid(True)
ax.set_facecolor('black')
plt.show()


steps = [1, 10, 20, 30, 40, 100] # Selected time steps to display histograms

plt.figure(figsize = (12, 8))
plt.style.use('dark_background')

neon_colors = ['#ff00ff', '#00ffcc', '#ffcc00', '#ff0066', '#cc00ff', '#00ccff']

for i, step in enumerate(steps):
    plt.subplot(2, 3, i + 1)
    plt.hist(X[:, step], bins = 50, density = True, color = neon_colors[i], edgecolor = 'black')
    plt.title(f"Histogram over time, step {step}")
    plt.xlabel("$X_n$")
    plt.ylabel("Density")
    plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)

plt.tight_layout()
plt.show()


E_X_t = X.mean(axis = 0)
E_X_t

plt.figure(figsize = (15, 5))
plt.plot(E_X_t, color = 'cyan')
plt.axhline(0, color='red', linestyle = '--')
plt.title("$E[X_t]$")
plt.xlabel("Time ($t$)")
plt.ylabel("$E[X_t]$")
plt.grid(True)
plt.show()


Var_X_t = X.var(axis = 0)
Var_X_t

plt.figure(figsize = (15, 5))
plt.plot(Var_X_t, color = 'cyan')
plt.title("$Var[X_t]$")
plt.xlabel("Time (t)")
plt.ylabel("$Var[X_t]$")
plt.grid(True)
plt.show()


Cov_Xt_y_Xtprima = np.zeros((N, N)) # 101x101

for t in range(N):
    for t_prima in range(N):
        Cov_Xt_y_Xtprima[t, t_prima] = np.mean(X[:, t] * X[:, t_prima]) - np.mean(X[:, t]) * np.mean(X[:, t_prima])

Cov_Xt_y_Xtprima

plt.imshow(Cov_Xt_y_Xtprima, cmap = 'viridis')
plt.colorbar()
plt.title("$Cov[X_t, X_t']$")
plt.xlabel("Time (t')")
plt.ylabel("Time (t)")
plt.tight_layout()
plt.show()


def random_walk(n):
    # Generates n values ​​of Z following a binomial distribution and converts them to -1 or 1
    z = 2 * np.random.binomial(1, 0.5, n) - 1
    return np.cumsum(z)


steps = [10, 50, 100, 200, 250]
walks = [random_walk(n) for n in steps]

neon_colors = ['#ff00ff', '#00ffff', '#ff99cc', '#ccff66', '#ff0000']
plt.figure(figsize = (15, 6))
for i, walk in enumerate(walks):
    plt.plot(walk, label = f'n = {steps[i]}', color = neon_colors[i])

plt.title('Random Walks with Different Steps')
plt.xlabel('Steps')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()


from scipy.stats import binom

fig, ax = plt.subplots(figsize = (15, 6))

for i, n in enumerate(steps):
    x = np.arange(0, n + 1)
    y = binom.pmf(x, n, 0.5)
    ax.plot(x, y, label = f'n = {n}', color = neon_colors[i])

ax.set_title('Binomial Distribution for different values ​​of N')
ax.set_xlabel('Number of Successes')
ax.set_ylabel('Probability')
ax.legend()
plt.tight_layout()
plt.show()



steps = [300, 400, 500, 600, 700, 800, 900, 1000]
walks = [random_walk(n) for n in steps]

neon_colors = ['#ff00ff', '#00ffff', '#ff99cc', '#ccff66', '#ff6699', '#99ffcc', '#ffff00', '#ff0000']

plt.figure(figsize = (15, 6))
for i, walk in enumerate(walks):
    plt.plot(walk, label = f'n = {steps[i]}', color = neon_colors[i])

plt.title('Random Walks with Different Steps')
plt.xlabel('Steps')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()


fig, ax = plt.subplots(figsize = (15, 6))

for i, n in enumerate(steps):
    x = np.arange(0, n + 1)
    y = binom.pmf(x, n, 0.5)
    ax.plot(x, y, label = f'n = {n}', color = neon_colors[i])

ax.set_title('Binomial Distribution for Different Values of N')
ax.set_xlabel('Number of Successes')
ax.set_ylabel('Probability')
ax.legend()
plt.tight_layout()
plt.show()


steps = [1000000]
walks = [random_walk(n) for n in steps]

plt.figure(figsize = (15, 6))
for i, walk in enumerate(walks):
    plt.plot(walk, label = f'n = {steps[i]}', color = '#ff00ff')

plt.title('Random Walk with N = 1,000,000')
plt.xlabel('Steps')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()



from scipy.stats import norm

n = 1000000
p = 0.5
mu = n * p
sigma = np.sqrt(n * p * (1-p))

x_binom = np.arange(496000, 504000)
y_binom = binom.pmf(x_binom, n, p)


x_norm = np.linspace(496000, 504000, 1000)
y_norm = norm.pdf(x_norm, mu, sigma)

fig, ax = plt.subplots(figsize = (15, 6))

ax.bar(x_binom, y_binom, label = 'Empirical Binomial Distribution', color = '#ffff00', width = 50)
ax.plot(x_norm, y_norm, label = 'Theoretical Normal Distribution', color = '#ff0000', linestyle = '--')

ax.set_title('Empirical Binomial Distribution vs. Theoretical Normal Distribution where $N = 1,000,000$')
ax.set_xlabel('Number of Successes')
ax.set_ylabel('Probability')
ax.legend()
ax.set_xlim(496000, 504000)
plt.tight_layout()
plt.show()


def random_walk(n):
    # Generates n values ​​of Z following a binomial distribution and converts them to -1 or 1
    z = 2 * np.random.binomial(1, 0.5, n) - 1
    return np.cumsum(z)

n_steps = 1001
K = 10000

all_walks = [random_walk(n_steps) for _ in range(K)]

sample_indices = np.random.randint(0, K, 20)
sample_walks = [all_walks[i] for i in sample_indices]


neon_colors = ['#ff00ff', '#00ffff', '#ff99cc', '#ccff66', '#ff6699', '#99ffcc', '#ffff00', '#ff0000', '#cc00cc', '#00ccff', '#ff9966', '#66ff99', '#cc0099', '#99ccff', '#ffcc00', '#cc0000', '#00ffcc', '#ccff33', '#ff33cc', '#33ccff']

plt.figure(figsize = (15, 6))

for i, walk in enumerate(sample_walks):
    plt.plot(walk, alpha = 0.7, color = neon_colors[i], label = f'Walk {i + 1}')

plt.title('20 Random Walks Sampled from 10,000 Walks')
plt.xlabel('Steps')
plt.ylabel('Position')
plt.grid(True)
plt.legend()
plt.show()

steps2 = [50, 100, 300, 500, 700, 1000]

# We collect positions at times of interest for all walks
positions_at_steps = {step: [walk[step] for walk in all_walks] for step in steps2}

neon_colors = ['#ff00ff', '#00ffff', '#ff99cc', '#ccff66', '#ff6699', '#99ffcc', '#ffff00']

p = 0.5

plt.figure(figsize = (12, 8))

for i, step in enumerate(steps2, 1):
    plt.subplot(3, 2, i)
    plt.hist(positions_at_steps[step], bins = 30, density = True, color = neon_colors[i], edgecolor = 'black', alpha = 0.6, label = "Walks")

    # Theoretical Binomial Distribution
    x_binom = 2 * np.arange(0, step + 1) - step
    y_binom = binom.pmf(np.arange(0, step + 1), step, p)
    plt.plot(x_binom, y_binom, color = 'blue', label = 'Binomial', linewidth = 2)

    # Theoretical Normal Distribution
    mu = 0
    sigma = np.sqrt(step)
    x_norm = np.linspace(-step, step, 1000)
    y_norm = norm.pdf(x_norm, mu, sigma)
    plt.plot(x_norm, y_norm, color = 'red', label = 'Normal', linewidth = 2)
    plt.xlim(mu - 3.5*sigma, mu + 3.5*sigma) # Adjust the x-axis range to include a little more than the 3 standard deviations
    
    plt.title(f'Histogram with {step} steps')
    plt.xlabel('Position')
    plt.ylabel('Density')
    plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)
    plt.legend()

plt.tight_layout()
plt.show()


thresholds = [5, 20]
times_to_threshold = {x: [] for x in thresholds}

for walk in all_walks:
    for x in thresholds:
        times_to_threshold[x].append(np.where(walk >= x)[0][0] if (walk >= x).any() else np.nan)


neon_colors_for_thresholds = {'5': '#ff00ff', '20': '#00ffff'}

plt.figure(figsize = (12, 6))

for idx, x in enumerate(thresholds, 1):
    plt.subplot(1, 2, idx)

    valid_times = [time for time in times_to_threshold[x] if not np.isnan(time)]
    plt.hist(valid_times, bins=30, density = True, color = neon_colors_for_thresholds[str(x)], alpha = 0.7, label = f'Times for $x^*={x}$')

    plt.title(f'Empirical PMF for $x^*={x}$')
    plt.xlabel('Time to reach threshold')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)

plt.tight_layout()
plt.show()


thresholds = [-5, -20]
times_to_threshold = {x: [] for x in thresholds}

# We go from >= x to <= x
for walk in all_walks:
    for x in thresholds:
        times_to_threshold[x].append(np.where(walk <= x)[0][0] if (walk <= x).any() else np.nan)

neon_colors_for_thresholds = {-5: '#ff0000', -20: '#ccff66'}

plt.figure(figsize=(12, 6))
for idx, x in enumerate(thresholds, 1):
    plt.subplot(1, 2, idx)

    valid_times = [time for time in times_to_threshold[x] if not np.isnan(time)]
    plt.hist(valid_times, bins=30, density=True, color=neon_colors_for_thresholds[x], alpha=0.7, label = f'Times for $x^*={x}$')

    plt.title(f'Empirical PMF for $x^*={x}$')
    plt.xlabel('Time to reach threshold')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
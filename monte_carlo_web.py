import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Function to generate GBM data
def monte_carlo_sim(current_prices, volatilities, mu, num_days, num_simulations=1000, seed=42):
    N = num_days  # Number of time steps
    T_days = 1 / 365  # Time horizon in years
    dt = T_days  # Time step
    S0 = current_prices
    sigma = volatilities  # Volatility
    np.random.seed(seed)

    # Initialize the stock price matrix
    S = np.zeros((num_simulations, N + 1))
    S[:, 0] = S0

    Days_annualized = np.insert(np.cumsum(np.full(num_days, dt)), 0, 0)
    diff_Days_annualized = np.diff(Days_annualized[:N+1])
    d_t = np.pad(diff_Days_annualized, (1, 0), 'constant', constant_values=0)

    # Generate random numbers for the Brownian motion
    mean_gbm = 0
    std_gbm = np.sqrt(d_t)
    dW = np.random.normal(mean_gbm, std_gbm, size=(num_simulations, N + 1))

    # Cumulative sum of random numbers
    W_t = np.cumsum(dW, axis=1)  # Cumulative sum of Brownian motion

    # Calculate each step based on previous step
    for t in range(1, N + 1):
        dW_increment = W_t[:, t] - W_t[:, t-1]  # Brownian increment
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW_increment)

    return S

# Streamlit app
st.title('Geometric Brownian Motion Simulation by AIM BJP TEAM')

# Sidebar inputs
st.sidebar.header('Settings')

product_name = st.sidebar.text_input('Product Name', 'VaR95')
current_price = st.sidebar.number_input('Current Price', min_value=0.0, value=100.0)
volatility = st.sidebar.number_input('Volatility (σ)', min_value=0.0, value=0.08, format="%.4f")
mu = st.sidebar.number_input('Drift Rate (μ)', min_value=0.0, value=0.08, format="%.4f")
N = st.sidebar.number_input('Number of Days', min_value=1, value=14, step=1)
M = st.sidebar.number_input('Number of Simulations', min_value=1, value=1000, step=1)
subset_size = st.sidebar.number_input('Subset Size for Plotting', min_value=1, value=1000, step=1)
VaR_confidence = st.sidebar.number_input('Confidence Level (%)', min_value=0.0, max_value=100.0, value=95.0, step=0.1)

# Generate simulated data
simulated_data = monte_carlo_sim(current_price, volatility, mu, N, M)

# Calculate VaR
VaR_percentile = 100 - VaR_confidence
VaR_value = np.percentile(simulated_data[:, -1], VaR_percentile)

# Display results
st.header(f'Simulation Results for {product_name}')
st.write('Initial Price:', current_price)
st.write('Volatility:', volatility)
st.write(f'value at {VaR_confidence}: ', VaR_value)

# Plot simulated paths
fig, ax = plt.subplots(figsize=(10, 6))
t = np.linspace(0, N, N + 1) / 365  # Time grid in years
for i in range(subset_size):  # Plot a subset for clarity
    ax.plot(t, simulated_data[i])
ax.set_xlabel("Years $(t)$")
ax.set_ylabel("Price")
ax.set_title(
    f"Realizations of Geometric Brownian Motion\n$S_0 = {current_price}, \mu = {mu}, \sigma = {volatility}$")
st.pyplot(fig)

# Plot the final distribution of stock prices with VaR line
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(simulated_data[:, -1], bins=50, edgecolor='black')
ax.axvline(VaR_value, color='r', linestyle='dashed', linewidth=2)
ax.set_xlabel(f"Price after {N} Days")
ax.set_ylabel("Frequency")
ax.set_title(f"Distribution of Prices after {N} Days")
ax.legend([f'{VaR_confidence}% : {VaR_value:.2f}'])
st.pyplot(fig)

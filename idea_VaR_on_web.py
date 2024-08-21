import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Function to generate GBM data
def generate_data(M, S0, mu, sigma, N):
    T_days = N / 365  # Time horizon in years (N days)
    dt = T_days  # Time step

    # Generate random numbers for the Brownian motion (noise)
    np.random.seed(42)
    noise = np.random.normal(0, np.sqrt(dt), size=(M, N))

    # Compute the exponential component of the GBM formula
    St = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * noise)

    # Add an initial column of ones for the starting price
    ones = np.ones((M, 1))

    # Concatenate the ones array to the beginning of St along the second axis (columns)
    St = np.hstack([ones, St])

    # Multiply through by S0 and return the cumulative product of simulation 
    St = S0 * St.cumprod(axis=1)
    return St

# Streamlit app
st.title('Geometric Brownian Motion Simulation by AIM BJP TEAM')

# Sidebar inputs
st.sidebar.header('Settings')

product_name = st.sidebar.text_input('Product Name', 'VaR95')
current_price = st.sidebar.number_input('Current Price', min_value=0.0, value=100.0)
volatility = st.sidebar.number_input('Volatility (σ)', min_value=0.0, value=0.03, format="%.4f")
mu = st.sidebar.number_input('Drift Rate (μ)', min_value=0.0, value=0.08, format="%.4f")
N = st.sidebar.number_input('Number of Days', min_value=1, value=14, step=1)
M = st.sidebar.number_input('Number of Simulations', min_value=1, value=1000, step=1)
subset_size = st.sidebar.number_input('Subset Size for Plotting', min_value=1, value=10, step=1)
VaR_confidence = st.sidebar.number_input('Confidence Level (%)', min_value=0.0, max_value=100.0, value=95.0, step=0.1)

# Generate simulated data
simulated_data = generate_data(M, current_price, mu, volatility, N)

# Calculate VaR
VaR_percentile = 100 - VaR_confidence
VaR_value = np.percentile(simulated_data[:, -1], VaR_percentile)

# Plot simulated paths
st.header(f'Simulation Results for {product_name}')
st.write('Initial Price:', current_price)
st.write('Volatility:', volatility)
st.write(f'({VaR_confidence}% Confidence Level):', VaR_value)

# Plot simulated paths
fig, ax = plt.subplots(figsize=(10, 6))
t = np.linspace(0, N, N + 1) / 365  # Time grid in years (N+1 points including start)
for i in range(subset_size):  # Plot a subset for clarity
    ax.plot(t, simulated_data[i])
ax.set_xlabel("Years $(t)$")
ax.set_ylabel("Price")
ax.set_title(
    f"Realizations of Geometric Brownian Motion\n$S_0 = {current_price}, \mu = {mu}, \sigma = {volatility}$"
)
st.pyplot(fig)

# Plot the final distribution of stock prices with VaR line
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(simulated_data[:, -1], bins=50, edgecolor='black')
ax.axvline(VaR_value, color='r', linestyle='dashed', linewidth=2)
ax.set_xlabel("Price after {0} Days".format(N))
ax.set_ylabel("Frequency")
ax.set_title(f"Distribution of Prices after {N} Days")
ax.legend([f'{VaR_confidence}%): {VaR_value:.2f}'])
st.pyplot(fig)

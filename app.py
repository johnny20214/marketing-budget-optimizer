# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
# Make sure the 'advertising.csv' file is in the same directory as your script/notebook
try:
    df = pd.read_csv('E:\\ZS\\sim project\\advertising.csv')
except FileNotFoundError:
    print("Error: 'advertising.csv' not found. Please download it and place it in the correct folder.")
    # Exit or handle the error as you see fit
    exit()

# --- Initial Inspection ---

# 1. View the first 5 rows to understand the structure
print("--- First 5 Rows ---")
print(df.head())

# 2. Get a concise summary of the dataframe
# This checks for data types and non-null values
print("\n--- Data Info ---")
df.info()

# 3. Get descriptive statistics for numerical columns
# This gives you mean, std dev, min, max, etc.
print("\n--- Descriptive Statistics ---")
print(df.describe())

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# --- Exploratory Data Analysis ---

# 1. Visualize the relationship between features and the response using scatterplots
# This creates a grid of plots, one for each advertising channel vs. sales
print("--- Scatter Plots of Advertising Channels vs. Sales ---")
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

sns.scatterplot(data=df, x='TV', y='Sales', ax=axs[0])
axs[0].set_title('TV Spend vs. Sales')

sns.scatterplot(data=df, x='Radio', y='Sales', ax=axs[1])
axs[1].set_title('Radio Spend vs. Sales')

sns.scatterplot(data=df, x='Newspaper', y='Sales', ax=axs[2])
axs[2].set_title('Newspaper Spend vs. Sales')

plt.tight_layout()
plt.show()


# 2. Visualize the correlation between all variables using a heatmap
print("\n--- Correlation Matrix Heatmap ---")
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of All Variables')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- Step 2: Impact Modeling ---

# 1. Prepare the data for modeling
# X represents the features (spending on channels)
X = df[['TV', 'Radio', 'Newspaper']]
# y represents the target (Sales)
y = df['Sales']

# 2. Build and train the linear regression model
# We create an instance of the model
model = LinearRegression()

# We 'fit' the model to our data, which is the training process
model.fit(X, y)

# 3. Interpret the model's results
print("--- Model Coefficients ---")
print(f"Intercept: {model.intercept_:.4f}")
print("\nCoefficients:")
# Create a small DataFrame to see the coefficients next to their channel names
coeffs = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coeffs)

print("\n--- Interpretation ---")
print(f"For a $1000 increase in TV ad spend, sales are predicted to increase by {model.coef_[0]*1000:.2f} units, holding other factors constant.")
print(f"For a $1000 increase in Radio ad spend, sales are predicted to increase by {model.coef_[1]*1000:.2f} units, holding other factors constant.")
print(f"For a $1000 increase in Newspaper ad spend, sales are predicted to increase by {model.coef_[2]*1000:.2f} units, holding other factors constant.")

# --- Model Validation ---

# 1. Use the trained model to make predictions on our existing data
y_pred = model.predict(X)

# 2. Calculate the R-squared value
from sklearn.metrics import r2_score
r_squared = r2_score(y, y_pred)

print(f"\n--- Model Performance ---")
print(f"R-squared (R²): {r_squared:.4f}")

# 3. Interpret the R-squared value
print(f"\n--- Interpretation of R-squared ---")
print(f"Our model explains approximately {r_squared:.1%} of the variance in Sales.")
print("This indicates a very strong fit, meaning the model's predictions are highly representative of the actual sales data.")

import scipy.optimize as optimize

# --- Step 3: Budget Optimization ---

# 1. Define the total marketing budget we want to allocate
# Let's assume a hypothetical budget of $100,000
TOTAL_BUDGET = 100

# Note: Since our model was trained on the raw numbers (e.g., TV=230.1),
# the budget should be on the same scale. The dataset values are in thousands of dollars,
# so TOTAL_BUDGET = 100 represents $100,000.

# 2. Define the objective function we want to maximize (our model's prediction)
# The optimizer is a minimizer, so we return the NEGATIVE of the sales.
# 'x' will be a list of spends: [tv_spend, radio_spend, newspaper_spend]
def objective_function(x):
    # We use our trained 'model' to predict sales
    sales_prediction = model.predict([x])
    # Return the negative of the prediction
    return -sales_prediction[0]

# 3. Define the constraints for the optimization
# Constraint 1: The sum of all spends must be equal to the total budget.
cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] - TOTAL_BUDGET})

# Constraint 2: The spend on each channel cannot be negative.
# This is defined by 'bounds'.
bnds = ((0, None), (0, None), (0, None)) # (min, max) for each spend: TV, Radio, Newspaper

# 4. Set an initial guess for the allocation
# An equal split is a reasonable starting point.
initial_guess = [TOTAL_BUDGET / 3, TOTAL_BUDGET / 3, TOTAL_BUDGET / 3]

# 5. Run the optimizer
# We use the 'SLSQP' method which is good for constrained optimization
solution = optimize.minimize(objective_function, initial_guess, method='SLSQP', bounds=bnds, constraints=cons)


# 6. Display the results
if solution.success:
    optimal_spends = solution.x
    max_sales = -solution.fun

    print("--- Budget Optimization Results ---")
    print(f"Total Budget to Allocate: ${TOTAL_BUDGET * 1000:,.2f}")
    print("\nOptimal Spend Allocation:")
    print(f"  TV:        ${optimal_spends[0] * 1000:,.2f}")
    print(f"  Radio:     ${optimal_spends[1] * 1000:,.2f}")
    print(f"  Newspaper: ${optimal_spends[2] * 1000:,.2f}")
    print("\nProjected Maximum Sales:")
    print(f"  {max_sales * 1000:,.0f} units") # Sales are often whole units
else:
    print("Optimization failed:", solution.message)


    # ===============================================
# --- Step 4: Streamlit Simulation Tool ---
# ===============================================
import streamlit as st

# --- App UI ---
st.title("Marketing Budget Optimization Simulator")
st.write("""
This tool uses a linear regression model (R² = 0.903) to predict sales based on advertising spend.
Use the sidebar to set a total budget, and the tool will recommend an optimal allocation.
You can also manually set spending to simulate different scenarios.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Budget Configuration")
total_budget_input = st.sidebar.number_input("Enter Total Marketing Budget ($)", min_value=1000, max_value=1000000, value=100000, step=1000)

# The model and optimizer expect budget in 'thousands' scale
total_budget = total_budget_input / 1000

# --- Optimization Section ---
# --- Optimization Section ---
st.header("1. Optimal Budget Allocation")

st.markdown("""
Based on our model, we calculate the optimal spend to maximize sales. 
You can apply business constraints below to reflect real-world strategy.
""")

# --- Add new sliders for business constraints ---
min_tv_pct = st.slider("Min TV % of Budget", 0, 100, 10)
min_radio_pct = st.slider("Min Radio % of Budget", 0, 100, 10)
min_newspaper_pct = st.slider("Min Newspaper % of Budget", 0, 100, 0) # Default to 0 as it's ineffective

# --- Check if constraints are possible ---
if (min_tv_pct + min_radio_pct + min_newspaper_pct) > 100:
    st.error("Minimum allocation percentages cannot exceed 100%. Please adjust.")
else:
    # --- Dynamic Optimization with Business Constraints ---
    # Constraint 1: Sum of spends must equal the total budget.
    cons_dynamic = ({'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] - total_budget})

    # Constraint 2: Bounds for each spend (min and max)
    # x[0] = TV, x[1] = Radio, x[2] = Newspaper
    bnds_dynamic = (
        (total_budget * (min_tv_pct/100), None),          # TV min spend
        (total_budget * (min_radio_pct/100), None),      # Radio min spend
        (total_budget * (min_newspaper_pct/100), None)   # Newspaper min spend
    )

    initial_guess_dynamic = [total_budget / 3, total_budget / 3, total_budget / 3]

    solution = optimize.minimize(objective_function, initial_guess_dynamic, method='SLSQP', bounds=bnds_dynamic, constraints=cons_dynamic)

    if solution.success:
        optimal_spends = solution.x
        max_sales = -solution.fun

        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal TV Spend", f"${optimal_spends[0]*1000:,.0f}")
        col2.metric("Optimal Radio Spend", f"${optimal_spends[1]*1000:,.0f}")
        col3.metric("Optimal Newspaper Spend", f"${optimal_spends[2]*1000:,.0f}")
        
        st.subheader(f"Projected Sales with Optimal Spend: {max_sales*1000:,.0f} units")
    else:
        st.error("Optimization could not find a solution with the given constraints.")

# Run the optimization (we can reuse the logic from Step 3)
cons_dynamic = ({'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] - total_budget})
bnds_dynamic = ((0, total_budget), (0, total_budget), (0, total_budget))
initial_guess_dynamic = [total_budget / 3, total_budget / 3, total_budget / 3]

solution = optimize.minimize(objective_function, initial_guess_dynamic, method='SLSQP', bounds=bnds_dynamic, constraints=cons_dynamic)

if solution.success:
    optimal_spends = solution.x
    max_sales = -solution.fun

    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal TV Spend", f"${optimal_spends[0]*1000:,.0f}")
    col2.metric("Optimal Radio Spend", f"${optimal_spends[1]*1000:,.0f}")
    col3.metric("Optimal Newspaper Spend", f"${optimal_spends[2]*1000:,.0f}")
    
    st.subheader(f"Projected Sales with Optimal Spend: {max_sales*1000:,.0f} units")
else:
    st.error("Optimization could not find a solution.")


# --- Manual Simulation Section ---
st.header("2. Manual Simulation")
st.write("Manually adjust spending in different channels to see the immediate impact on sales.")

tv_spend = st.slider("TV Spend ($)", 0, total_budget_input, int(optimal_spends[0]*1000), 1000)
radio_spend = st.slider("Radio Spend ($)", 0, total_budget_input, int(optimal_spends[1]*1000), 1000)
newspaper_spend = st.slider("Newspaper Spend ($)", 0, total_budget_input, int(optimal_spends[2]*1000), 1000)

# Predict sales based on manual inputs
manual_input = [[tv_spend/1000, radio_spend/1000, newspaper_spend/1000]]
manual_sales_prediction = model.predict(manual_input)

st.subheader(f"Projected Sales with Manual Spend: {manual_sales_prediction[0]*1000:,.0f} units")

# Optional: Show a warning if manual spend exceeds total budget
if (tv_spend + radio_spend + newspaper_spend) > total_budget_input:
    st.warning("Warning: Total manual spend exceeds the defined budget.")


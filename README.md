# marketing-budget-optimizer
Interactive web app designed to find optimal marketing budget allocation using linear regression and SciPy optimization.

# Marketing Budget Optimization Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://marketing-budget-optimizer-zs-project.streamlit.app/)

An interactive web application built to help businesses make data-driven decisions on marketing budget allocation. This tool moves beyond simple analysis to provide prescriptive, optimized recommendations that maximize sales ROI, directly mirroring the type of analytical solutions developed at firms like ZS Associates.

---

## 🚀 Live Demo

**You can access the live, interactive application here:**

**[https://marketing-budget-optimizer-zs-project.streamlit.app/](https://marketing-budget-optimizer-zs-project.streamlit.app/)**

---

## 📸 Application Screenshot



---

## 🎯 Project Overview

In a competitive market, allocating a marketing budget effectively is a critical challenge. This project addresses this problem by providing a "Decision Analytics" tool that allows a user to:
1.  **Understand** the impact of spending in different channels (TV, Radio, Newspaper).
2.  **Optimize** a given budget for maximum sales, based on a predictive model.
3.  **Simulate** different spending scenarios to make informed strategic decisions.

The application is built on a robust linear regression model with an **R-squared of 0.903**, ensuring that its recommendations are based on statistically significant relationships within the data.

---

## ✨ Key Features

*   **📈 Predictive Modeling:** Uses a Scikit-learn regression model to forecast sales based on advertising spend.
*   **💡 Smart Optimization:** Leverages SciPy's optimization engine to find the ideal budget allocation that maximizes predicted sales.
*   **🎛️ Interactive Scenario Planning:** Users can manually adjust spend in each channel with sliders to see the immediate impact on sales forecasts.
*   **🏢 Business Constraint Integration:** The tool was enhanced to allow users to set minimum spending thresholds for each channel, turning a purely mathematical solution into a realistic, strategy-aware recommendation engine.

---

## 🛠️ Tech Stack

*   **Language:** Python
*   **Core Libraries:** Pandas, NumPy, Scikit-learn, SciPy
*   **Web Framework & UI:** Streamlit
*   **Visualization:** Matplotlib, Seaborn
*   **Deployment:** Streamlit Community Cloud (via GitHub)

---

## 챌 Challenge & Key Learning: From Insight to Improvement

A key challenge encountered during this project became its most valuable feature. The initial optimization model consistently recommended a 100% budget allocation to the single most effective channel (Radio).

*   **Problem:** While mathematically correct, this result lacked real-world business viability, as it ignored the strategic importance of a multi-channel presence and the concept of market saturation.
*   **Action:** I diagnosed this as a limitation of a purely linear model and reframed the problem. The goal wasn't just to find the mathematical maximum, but the *best outcome within realistic business constraints*.
*   **Solution:** I enhanced the application by adding interactive sliders for users to define their own business rules (e.g., "must spend at least 10% on TV"). The optimization engine was re-engineered to incorporate these constraints.
*   **Result:** This transformed the tool from a simple calculator into a true strategic asset, demonstrating an ability to bridge the gap between technical data analysis and actionable business strategy.

---

## ⚙️ How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/marketing-budget-optimizer.git
    cd marketing-budget-optimizer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

# 🏆 2026 FIFA World Cup Prediction Engine

A professional-grade, multi-factor Monte Carlo simulation engine that predicts outcomes for the expanded 48-team 2026 FIFA World Cup. 

The project bridges the gap between football intuition and quantitative model design, integrating **Poisson-distributed goal scoring** with **macroeconomic and environmental risk factors** (derived from Joachim Klement's econometric models) and a reactive dashboard built in **Streamlit**.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/2026-world-cup-prediction-engine.git
cd 2026-world-cup-prediction-engine
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
Start the local Streamlit dashboard:
```bash
python -m streamlit run app.py
```

---

## 🛠️ Project Structure

The codebase is designed to be modular, separating data representation, quantitative simulation, and user interface layers:

*   **`app.py`**: The Streamlit frontend dashboard. Implements reactive data tables, interactive Plotly visualization components (radar charts, convergence logs, and tournament brackets), and bypasses module-caching to allow hot-reloading of data.
*   **`world_cup_engine.py`**: The simulation core. Contains the expected goals formula, knockout tournament bracket resolver, Monte Carlo iterations loop, and shootout resolution mechanics.
*   **`team_data.py`**: The database. Stores comprehensive profiles for all 48 qualified teams including FIFA Elo, star player ratings, squad depth, recent form, tactical style, historical pedigree, and macroeconomic factors.
*   **`requirements.txt`**: Python dependencies (NumPy, Pandas, Streamlit, Plotly).

---

## 📊 The Mathematical Model

### 1. Match Simulation (Poisson Process)
Goals in football are modeled as discrete events. We draw team goal counts from independent Poisson distributions where the probability of a team scoring $k$ goals is:

$P(k \text{ goals}) = \frac{\lambda^k e^{-\lambda}}{k!}$

The goal expectancy rate ($\lambda$) is derived from the relative composite strengths of the two competing teams.

### 2. Multi-Factor Strength Score
Instead of relying solely on FIFA rankings or Elo ratings, the engine computes a composite strength score for each team using a weighted combination of eleven normalized sporting and macroeconomic indicators:

*   **Sporting Factors (88%)**: Elo (30%), Recent Form (12% - momentum), Star Player Rating (10%), Squad Depth (10% - bench liquidity), Offensive/Defensive Ratings (8% each), Tournament Pedigree (8%), and Manager Experience (5%).
*   **Macroeconomic Factors (4%)**: Log-scaled GDP per capita (2%) and Log-scaled Population (2%) — implementing diminishing returns of population size.
*   **Environmental Factors (5%)**: Host Advantage (2% - USA, Mexico, Canada) and Climate/Temperature Affinity (3%). Temperature affinity is modeled using a non-linear Gaussian bell curve centered at the optimal physical performance temperature of 14°C (57°F):

$\text{norm\_temp} = e^{-\frac{(\text{avg\_temp} - 14.0)^2}{100.0}}$

### 3. Knockout Shootout Damping
Penalty shootouts are high-variance events. To prevent elite teams from smoothly cruising through the bracket without facing knockout tail-risk, the engine applies a **shootout damping factor** (`0.25`) that compresses rating differences and models shootouts closer to a coin flip:

$\text{pen\_prob} = 0.5 + (\text{base\_prob} - 0.5) \times 0.25$

---

## 📈 Dashboard Features

*   **Monte Carlo Simulation**: Run up to 10,000 parallel simulations to calculate win probabilities for all 48 teams at every stage of the tournament.
*   **Dynamic Weight Calibration**: Adjust the weights of the 11 quantitative factors in real-time to see how your football hypotheses alter simulated outcomes.
*   **Interactive Brackets**: View the consensus (deterministic) tournament bracket dynamically rendered in the UI.
*   **Team Deep-Dives**: Compare squad profiles, tactical descriptions, star players, and radar charts of any two teams.
*   **Convergence Analysis**: Track how team win probabilities stabilize over simulation iterations.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

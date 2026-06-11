# 🏆 2026 FIFA World Cup Prediction Engine

A professional-grade, multi-factor Monte Carlo simulation engine that predicts outcomes for the expanded 48-team 2026 FIFA World Cup. 

The project bridges the gap between football intuition and quantitative model design, integrating **Poisson-distributed goal scoring** with **macroeconomic and environmental risk factors** (derived from Joachim Klement's econometric models) and a reactive dashboard built in **Streamlit**.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/2026-world-cup-prediction-engine.git](https://github.com/your-username/2026-world-cup-prediction-engine.git)
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

* **`app.py`**: The Streamlit frontend dashboard. Implements reactive data tables, interactive Plotly visualization components (radar charts, convergence logs, and tournament brackets), and bypasses module-caching to allow hot-reloading of data.
* **`world_cup_engine.py`**: The simulation core. Contains the expected goals formula, knockout tournament bracket resolver, Monte Carlo iterations loop, and shootout resolution mechanics.
* **`team_data.py`**: The database. Stores comprehensive profiles for all 48 qualified teams including FIFA Elo, star player ratings, squad depth, recent form, tactical style, historical pedigree, and macroeconomic factors.
* **`requirements.txt`**: Python dependencies (NumPy, Pandas, Streamlit, Plotly).

---

## 🗄️ Data Engineering & Curation

The foundation of the prediction model is the comprehensive dataset stored in `team_data.py`. For all 48 qualified nations, features are curated and standardized across three main categories:

* **Quantitative Sporting Factors**: Includes historical World Football Elo, a calibrated `star_player_rating` (standardizing game-changers like Erling Haaland at 95 and Santiago Gimenez at 82), `squad_depth` (overall roster depth outside the starting XI), `offensive_rating`, `defensive_rating`, `recent_form` (win-draw-loss record in the last 10 competitive matches), and `manager_experience` (international tournament track record).
* **Qualitative Descriptive Profiles**: Curated details showing preferred formations (e.g., 4-3-3, 3-5-2), tactical styles (e.g., "high-press transitions", "low-block defensive shape"), historical World Cup appearances and finishes, and detailed bullet points outlining specific team strengths and weaknesses.
* **Macroeconomic & Environmental Overlays**: Contains national statistics for GDP per capita, population size (in millions), and average temperature (in °C) during the playing season to feed the econometric and climate performance models.

These features are standardized into a uniform nested dictionary schema, enabling the core engine to normalize variables onto a 0-1 scale dynamically for composite strength evaluation.

---

## 📊 The Mathematical Model

### 1. Match Simulation (Poisson Process)
Goals in football are modeled as discrete events. We draw team goal counts from independent Poisson distributions where the probability of a team scoring $k$ goals is:

$P(k \text{ goals}) = \frac{\lambda^k e^{-\lambda}}{k!}$

The goal expectancy rate ($\lambda$) is derived from the relative composite strengths of the two competing teams.

### 2. Multi-Factor Strength Score
Instead of relying solely on FIFA rankings or Elo ratings, the engine computes a composite strength score for each team using a weighted combination of eleven normalized sporting and macroeconomic indicators:

* **Sporting Factors (88%)**: Elo (30%), Recent Form (12% - momentum), Star Player Rating (10%), Squad Depth (10% - bench liquidity), Offensive/Defensive Ratings (8% each), Tournament Pedigree (8%), and Manager Experience (5%).
* **Macroeconomic Factors (4%)**: Log-scaled GDP per capita (2%) and Log-scaled Population (2%) — implementing diminishing returns of population size.
* **Environmental Factors (5%)**: Host Advantage (2% - USA, Mexico, Canada) and Climate/Temperature Affinity (3%). Temperature affinity is modeled using a non-linear Gaussian bell curve centered at the optimal physical performance temperature of 14°C (57°F):

$\text{norm temp} = e^{-\frac{(\text{avg temp} - 14.0)^2}{100.0}}$

### 3. Knockout Shootout Damping
Penalty shootouts are high-variance events. To prevent elite teams from smoothly cruising through the bracket without facing knockout tail-risk, the engine applies a **shootout damping factor** (`0.25`) that compresses rating differences and models shootouts closer to a coin flip:

$\text{pen prob} = 0.5 + (\text{base prob} - 0.5) \times 0.25$

---

## 📈 Interactive Features & Dashboard Tabs

The frontend dashboard organizes these complex models into seven intuitive, interactive sections:

1. **Tournament Odds**: Displays the global leaderboard showing win and round-by-round advancement probabilities generated over your chosen number of Monte Carlo simulation runs (up to 10,000 iterations).
2. **Predicted Bracket**: Renders the deterministic consensus tournament bracket tree. This represents the expected value baseline path from the Group Stage matches all the way to the Final shootout.
3. **Bracket Predictor**: Allows you to generate and explore a complete, single tournament outcome for a specific random seed. Changing the seed illustrates the immense variance of the bracket structure stochastically.
4. **Group Analysis**: Breaks down the qualification chances, expected standings points, and goal differences for each group (Groups A to L), illustrating which teams are bottlenecked.
5. **Team Profiles**: Provides detailed overviews of star players, coaching staff, formations, play styles, and lists strengths and weaknesses. Plots overlapping radar charts of two selected teams across the quantitative rating factors.
6. **Head-to-Head Simulator**: Lets you select any two qualified teams and simulate up to 50,000 matches. The simulator outputs a bivariate scoreline probability matrix, expected goal distributions, and win/draw/loss odds, showing how style clashes affect scoring densities.
7. **Knockout Scenarios**: Plots a dynamic Treemap of Quarterfinal Contenders, functioning as a probability concentration heat map. Quickly displays which side of the bracket is congested and where dark horses have paths to advance.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

# FIFA Oracle — Enterprise Architecture

> Production-grade AI football analytics platform. Advanced feature engineering, stacking ensemble with probability calibration, parallel Monte Carlo simulation with ET + penalty shootouts, What-If scenario engine, and a FastAPI service spec.

## Table of Contents
1. [System Overview](#system-overview)
2. [Module 1: Feature Engineering](#module-1-feature-engineering)
3. [Module 2: Modeling Layer](#module-2-modeling-layer)
4. [Module 3: Monte Carlo Simulator](#module-3-monte-carlo-simulator)
5. [Module 4: API & Dashboard](#module-4-api--dashboard)
6. [Design Patterns](#design-patterns)
7. [File Layout](#file-layout)

---

## System Overview

The platform is split into a **TypeScript-native core** that runs entirely in the browser (no Python required for the demo), plus an optional **FastAPI backend** that wraps the same algorithms in scikit-learn / XGBoost / LightGBM for production deployment with GPU acceleration and real training data.

```
┌─────────────────────────────────────────────────────────────┐
│                   Next.js 16 Frontend                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  React 19 + Tailwind 4 + shadcn/ui + Recharts       │    │
│  │  Lazy-loaded sections · Framer Motion animations    │    │
│  └────────────────────────┬────────────────────────────┘    │
│                           │                                  │
│  ┌────────────────────────▼────────────────────────────┐    │
│  │       TypeScript ML Core (src/lib/ml)               │    │
│  │  • Dynamic World-Elo (K-factor + G-multiplier)      │    │
│  │  • Feature engineering (xG proxy, fatigue, value)   │    │
│  │  • Strategy pattern ensemble (XGB/LGB/Poisson)      │    │
│  │  • Platt + Isotonic calibration                     │    │
│  │  • Grid search / Optuna optimizer                   │    │
│  │  • Monte Carlo v2 (ET + penalties + parallel)       │    │
│  │  • Web Worker pool for 10k+ sims                    │    │
│  │  • What-If scenario engine                          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ (optional, for production)
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Python Backend                       │
│  • scikit-learn StackingClassifier + CalibratedClassifierCV │
│  • XGBoost + LightGBM + PoissonRegressor base learners      │
│  • Optuna / GridSearchCV hyperparameter tuning              │
│  • ProcessPoolExecutor for parallel simulations             │
│  • Endpoints: /predict-match, /simulate-tournament, etc.    │
└─────────────────────────────────────────────────────────────┘
```

---

## Module 1: Feature Engineering

**Files**: `src/lib/ml/elo.ts`, `src/lib/ml/features.ts`

### Dynamic World-Elo Rating System

A custom Elo that updates after every historical match. The base update formula is:

```
R_new = R_old + K * G * (W - W_e)
```

| Component | Description |
|-----------|-------------|
| `K` (importance) | Friendly=10, Qualifier=25, WC Group=40, WC R16=45, WC QF=48, WC SF=50, WC Final=60 |
| `G` (goal mult) | 1.0 for 1-goal diff, 1.5 for 2-goal, (11+N)/8 for 3+ goals |
| `W` (actual) | 1 for win, 0.5 for draw, 0 for loss |
| `W_e` (expected) | `1 / (1 + 10^((R_opp - R_self)/400))` with +100 home advantage |

`computeWorldEloRatings()` replays all 22 historical World Cup tournaments through this engine to produce current ratings. Cached after first computation.

### xG-Proxy

Since historical match data lacks Opta-level shot tracking, we approximate Expected Goals as:

```
xG ≈ shots_per_match × conversion_ratio
shots_per_match ≈ goals_per_match / 0.18  (league-average conversion)
conversion_ratio = 0.13 + (attack_rating - 80) / 100 × 0.10
```

Defensive version (`xgAgainstProxy`) inversely scales by `defenseRating`.

### Fatigue & Travel Metrics

- **Rest days**: derived from match-day delta. `fatigueIndex = 0` for 4+ days, `0.85` for 0 days.
- **Travel distance**: Haversine formula between consecutive match venues. Travel penalty amplified when rest < 3 days.
- **Rest-day penalty**: Logistic curve `1 / (1 + exp(0.7 * (restDays - 2)))`.

### Market Value Integration

`squadValueNorm = (value_in_M€ - 100) / 1100` — caps at 0/1 across €100M–€1.2B range (Transfermarkt brackets).

**Entry point**: `extractFeatures(team, context?)` returns a 17-field `FeatureVector`. `extractMatchupFeatures(teamA, teamB, ctxA?, ctxB?)` adds 8 differential features for head-to-head modeling.

---

## Module 2: Modeling Layer

**Files**: `src/lib/ml/models.ts`, `src/lib/ml/calibration.ts`

### Strategy Pattern

All models implement the `ModelStrategy` interface:

```typescript
interface ModelStrategy {
  readonly name: string;
  fit(X: MatchupFeatures[], y: MatchOutcome[]): void;
  predictProba(features: MatchupFeatures): { pA: number; pB: number; pD: number };
  predictGoals(features: MatchupFeatures): { expA: number; expB: number };
  accuracy(): number;
  isTrained(): boolean;
}
```

Three concrete strategies:

| Model | Algorithm | Notes |
|-------|-----------|-------|
| `XGBoostModel` | Gradient-boosted additive logistic regressors on feature subsets | 50 boosting iterations, lr=0.1, mimics XGBoost's residual correction |
| `LightGBMModel` | Leaf-wise splits on best-gain features | 100 leaves, faster than XGBoost, captures different feature interactions |
| `PoissonModel` | Log-linear Poisson regression on goal-scoring features | `log(λ) = β₀ + β₁·eloDiff + β₂·attackDiff - β₃·defenseDiff + ...` |

### Stacking Ensemble

`StackingEnsemble` fits all three base models, then trains a logistic regression meta-learner on their out-of-fold predictions. The meta-weights are tilted toward the Poisson model (historically most accurate for football).

### Probability Calibration

Two `Calibrator` strategies (also Strategy pattern):

- **`PlattScaling`**: fits `σ(α·p + β)` via 500 epochs of gradient descent on Platt (1999) targets.
- **`IsotonicRegression`**: PAVA (Pool Adjacent Violators Algorithm) — fits a non-decreasing step function mapping raw scores to observed frequencies. More flexible, less parametric.

Both produce calibrated probabilities so a predicted 70% actually corresponds to 70% real-world success.

### Hyperparameter Optimization

- **`gridSearch(evaluateFn, grid, folds)`**: exhaustive search with K-fold CV, optimizing multi-class log loss. Returns best params + full leaderboard.
- **`OptunaOptimizer`**: TPE-style Bayesian optimizer shim. For production, integrate Python `optuna` via subprocess.

**Multi-class log loss** evaluator: `multiclassLogLoss(predicted, actual)`.

---

## Module 3: Monte Carlo Simulator

**Files**: `src/lib/ml/simulator-v2.ts`, `src/lib/worker/sim-worker.ts`

### Goal-Scoring Stochastic Engine

Each match produces a Poisson-distributed exact score:

```typescript
const features = extractMatchupFeatures(teamA, teamB);
const { expA, expB } = ensemble.predictGoals(features);
const lambdaA = Math.max(0.2, expA * fatigueAdjustment);
const goalsA = poissonSample(lambdaA, rng);
const goalsB = poissonSample(lambdaB, rng);
```

### Knockout Extra Time + Penalties

If a knockout match is drawn after 90':

1. **Extra Time (30 min)**: `etGoals = poisson(lambda × 0.33, rng)` — reduced scoring rate reflecting fatigue.
2. **Penalty Shootout** (if still drawn):
   - 5 kicks per team, conversion rate `0.72 + (avgPlayerRating - 80) / 200` (~67–77%)
   - Sudden death rounds until decided
   - If tied after 10 SD rounds, fall back to ELO-informed coin flip via `penaltyWinProbability()`

### Parallelization via Web Workers

`runMonteCarloV2Parallel()` distributes N simulations across `min(4, hardwareConcurrency - 1)` workers:

```typescript
const workerUrl = new URL("../worker/sim-worker.ts", import.meta.url);
const worker = new Worker(workerUrl, { type: "module" });
worker.postMessage({ type: "run", year, iterations: chunkSize, seed });
```

Each worker:
- Runs its chunk single-threaded (no nested workers)
- Posts `progress` messages back as it completes batches
- Posts `done` with its slice of `TournamentSimulation[]` results
- Main thread aggregates via `aggregateResults()`

Falls back to `runMonteCarloV2Async()` (chunked `setTimeout(0)` yielding) if workers are unavailable.

### Advanced Analytics Outputs

- **Dark Horse**: lowest-FIFA-ranked team to reach Quarterfinals, with frequency across all sims.
- **Group of Death**: 4-team cluster with highest avg ELO among non-finalists. Reports top 3 most frequent clusters.
- **Covariance Matrix**: for each pair of top-12 teams, computes `Cov(X_i, X_j)` where `X_i = 1` if team i reached QF. Positive covariance (green) = teams advance together; Negative (red) = they eliminate each other. Self-covariance shown on diagonal.

---

## Module 4: API & Dashboard

**Files**: `src/components/sections/advanced/WhatIfSimulator.tsx`, `src/components/sections/advanced/AdvancedSimulator.tsx`, `api-reference.py`

### What-If Simulator

Users override any of 6 team attributes (form, attack, defense, midfield, squad value, ELO adjustment) via sliders. The engine:

1. Applies overrides to a clone of the team
2. Temporarily patches the global `teams` array
3. Re-runs `generatePredictions()` to compute new championship probabilities
4. Restores the original teams array
5. Computes factor-breakdown deltas and a sample H2H prediction

Output: baseline vs. modified win %, rank movement, factor diff chart, head-to-head preview.

### Interactive Tournament Bracket

`BracketView` renders a single simulated tournament as 4 columns (R16 → QF → SF → F). Each match shows flags, codes, scores. Click to expand and reveal:
- Extra Time score (if applicable)
- Penalty shootout result
- Upset flag with ELO differential

### FastAPI Backend (Reference)

`api-reference.py` documents the production Python service:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Service + model health check |
| `GET /teams` | List teams with computed World-Elo |
| `POST /predict-match` | Single-match calibrated probability |
| `POST /simulate-tournament` | Parallel Monte Carlo (ProcessPoolExecutor) |
| `POST /what-if` | Override attributes & re-predict |
| `GET /model/info` | Architecture + accuracy + log loss |
| `POST /model/retrain` | Background retraining (admin) |

Backend uses `StackingClassifier` (sklearn) with `CalibratedClassifierCV` wrapping, hyperparameter search via `GridSearchCV` (multi-class log loss scoring), and `ProcessPoolExecutor` for parallel sims.

---

## Design Patterns

| Pattern | Application |
|---------|-------------|
| **Strategy** | `ModelStrategy` interface — swap XGBoost/LightGBM/Poisson at runtime |
| **Strategy** | `Calibrator` interface — swap Platt/Isotonic at runtime |
| **Singleton** | `getEnsemble()` and `getWorldEloRatings()` cache single instances |
| **Factory** | `build_ensemble()` in FastAPI constructs configured stacking classifiers |
| **Facade** | `extractFeatures()` hides 17 underlying feature computations behind one call |
| **Observer** | Web Worker `onmessage` callbacks for progress + done events |
| **Template Method** | `ModelStrategy.fit()` defines training protocol; subclasses implement details |

---

## File Layout

```
src/lib/
├── data/
│   ├── tournaments.ts        # 23 WC tournaments (1930–2026)
│   ├── teams.ts              # 17 national team profiles
│   └── players.ts            # 20+ legendary + active players
├── ml/
│   ├── elo.ts                # Dynamic World-Elo (K-factor + G-multiplier)
│   ├── features.ts           # xG proxy + fatigue + travel + market value
│   ├── models.ts             # Strategy pattern ensemble (XGB/LGB/Poisson/Stacking)
│   ├── calibration.ts        # Platt + Isotonic + GridSearch + Optuna shim
│   ├── simulator-v2.ts       # Poisson + ET + penalties + dark horse + GoD + covariance
│   └── what-if.ts            # Scenario engine for attribute overrides
├── worker/
│   └── sim-worker.ts         # Web Worker for parallel simulations
├── prediction.ts             # Original prediction engine (kept for backward compat)
└── simulator.ts              # Original simulator (kept for backward compat)

src/components/sections/
├── Hero.tsx                  # Animated hero with stadium background
├── HistoricalDatabase.tsx    # 1930–2026 tournament archive
├── PredictionEngine.tsx      # Original prediction visualization
├── WorldCupSimulator.tsx     # Original simulator UI
├── TeamAnalysis.tsx          # 16 national team profiles
├── HeadToHead.tsx            # Match predictor
├── HistoricalTrends.tsx      # 96-year trend analysis
├── PlayerAnalytics.tsx       # Player profiles + xG scatter
├── WorldMap.tsx              # Interactive host nations map
├── PredictionDashboard.tsx   # KPI dashboard
├── AdminPanel.tsx            # Admin console
└── advanced/
    ├── AdvancedSimulator.tsx # v2 simulator with bracket + covariance
    └── WhatIfSimulator.tsx   # What-If scenario engine

api-reference.py              # FastAPI reference implementation
requirements.txt              # Python backend dependencies
ARCHITECTURE.md               # This file
```

---

## Performance Benchmarks

| Operation | Single-threaded | 4 Web Workers | Speedup |
|-----------|-----------------|---------------|---------|
| 1,000 sims | ~80ms | ~30ms | 2.7× |
| 10,000 sims | ~800ms | ~200ms | 4.0× |
| 100,000 sims | ~8,500ms | ~2,100ms | 4.0× |

The UI remains fully responsive during simulation thanks to chunked async yielding on the main thread + true parallelism in workers.

---

## Production Deployment

1. **Frontend**: Deploy Next.js app to Vercel — all ML runs client-side.
2. **Backend** (optional): Deploy FastAPI to AWS ECS/Fargate with `gunicorn` + `uvicorn` workers.
3. **Model training**: Run `python -c "from api import hyperparameter_search; ..."` nightly on a GPU instance.
4. **Database**: PostgreSQL on Supabase for team/match/player persistence.
5. **Monitoring**: Prometheus + Grafana for `/predict-match` latency, ensemble log loss, worker pool utilization.

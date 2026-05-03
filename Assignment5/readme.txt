Assignment V: Goal-Based Portfolio Optimisation
================================================
data.py      — Downloads 10 years of NSE price data for 5 Indian stocks, computes annualised
               expected returns (mu), the 5x5 covariance matrix (cov), and a 20-year savings
               schedule (₹20,000/month growing at 4% p.a.).
simulator.py — Monte Carlo engine that runs 5,000 portfolio paths over 20 years, applying
               annual returns, savings contributions, goal deductions, and a 12% borrowing
               penalty for shortfalls, returning final portfolio values at Year 20.
optimiser.py — Brute-force search over all valid discrete weight combinations
               ({0, 0.25, 0.5, 0.75, 1.0} summing to 1.0) to find the portfolio that
               maximises the probability of reaching the ₹1.5 Crore terminal goal.
bonus.py     — Continuous optimiser using scipy.optimize.minimize (allowing short-selling)
               to push beyond the discrete grid and find a globally better weight vector.
main.ipynb   — End-to-end notebook that imports all modules, runs both goal sequences
               (A and B), displays results, fan charts, and the bonus optimiser comparison.
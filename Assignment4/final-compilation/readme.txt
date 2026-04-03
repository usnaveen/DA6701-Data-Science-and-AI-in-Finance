Assignment 4: Notes
================================

Contents in submission
---------------------------
1. Report.pdf

2. data-preparation.ipynb
   Notebook used to clean the raw OHLCV CSV files, filter invalid/incomplete tickers,
   create the final stock universe, compute log returns, and create the train /
   validation / holdout datasets.

3. generate_sector_map.py
   Script used to create the sector mapping file from ticker metadata.

4. portfolio-replication.ipynb
   Main notebook used to run the replication methods, compare results, and
   generate the figures used in the report.

5. method_lasso.py
   Lasso-based stock selection method (plus QP weight solver).

6. method_autoencoder.py
   Autoencoder-based stock selection method, plus the shared QP weight solver.

7. method_ga.py
   Genetic Algorithm based stock selection method.

8. eval.py
   Shared evaluation utilities such as Tracking Error, Information Ratio, and
   portfolio return computation.

9. plots.py
    Plotting helpers used to generate the final figures.

10. figures/
    Contains the generated PNG plots used in the final report.

11. data/
    Contains processed outputs created from the data-preparation notebook and
    the replication notebook (may not contain train-val-hold out splits, and OHLCV_Data.zip).

12. Group details.txt and requirements.txt

Recommended order to reproduce everything
-----------------------------------------
Step 1. Unzip the raw data
    unzip OHLCV_Data.zip
This creates:
    Mega/
which contains the benchmark file (^GSPC.csv) and constituent stock CSVs.


Step 2. Run the data preparation notebook (data-preparation.ipynb)
This creates/refreshes the cleaned datasets in:
    data/
including:
- train_returns.parquet
- val_returns.parquet
- test_returns.parquet
- returns_all.parquet
- stock_returns.parquet
- bench_returns.parquet
- final_prices.parquet
- kept_universe.parquet
- excluded_universe.parquet
- summary.csv
- file_issues.csv
- train_gap_exclusions.csv


Step 3. Generate the sector map
    python generate_sector_map.py
This creates:
    data/sector_map.csv


Step 4. Run the main replication notebook (portfolio-replication.ipynb)
This notebook:
- runs the methods
- evaluates validation and holdout performance
- writes result CSVs into data/
- writes figures into figures/


Main generated outputs after Step 4
-----------------------------------
In data/:
- lasso_results.csv
- ae_results.csv
- ga_results.csv

In figures/:
- te_vs_k.png
- information_ratio.png
- sector_drift_lasso.png
- sector_drift_ae.png
- sector_drift_ga.png
- cumret_val.png
- cumret_hold.png
- communality_distribution.png


Minimal reproduction sequence
-----------------------------
The shortest correct order of execution of notebooks/scripts:

1. unzip OHLCV_Data.zip
2. run data-preparation.ipynb
3. run generate_sector_map.py
4. run portfolio-replication.ipynb
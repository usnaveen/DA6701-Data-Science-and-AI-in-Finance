from data_fetch import fetch_price_data
from preprocess import compute_monthly_returns, train_test_split
from covariance_analysis import compute_sample_covariance, compute_condition_number
from bootstrap import bootstrap_covariance_stability, plot_bootstrap_distribution
from visualization import plot_covariance_heatmap

def main():

    print("Fetching Data...")
    prices = fetch_price_data()

    print("Computing Monthly Returns...")
    monthly_returns = compute_monthly_returns(prices)

    print("Splitting Train/Test...")
    train, test = train_test_split(monthly_returns)

    print("Computing Sample Covariance...")
    S = compute_sample_covariance(train)

    print("Computing Condition Number...")
    kappa = compute_condition_number(S)
    print(f"Condition Number (κ): {kappa}")

    print("Bootstrap Stability Analysis...")
    condition_numbers = bootstrap_covariance_stability(train)
    plot_bootstrap_distribution(condition_numbers)

    print("Plotting Covariance Heatmap...")
    plot_covariance_heatmap(S, "Sample Covariance Matrix", "sample_covariance.png")

    print("Done.")

if __name__ == "__main__":
    main()
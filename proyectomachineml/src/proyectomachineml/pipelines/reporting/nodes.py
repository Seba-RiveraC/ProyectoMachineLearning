import pandas as pd
import matplotlib.pyplot as plt

def plot_gdp_vs_life(merged_data: pd.DataFrame):
    plt.scatter(merged_data["total_gdp"], merged_data["life_expectancy"], alpha=0.5)
    plt.xlabel("GDP (USD)")
    plt.ylabel("Life Expectancy (Years)")
    plt.title("GDP vs Life Expectancy")
    plt.savefig("data/04_feature/gdp_vs_life.png")
    plt.close()


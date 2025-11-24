import pandas as pd
import matplotlib.pyplot as plt


def load_and_clean_data():
    data_path = "data/covid_data.csv"
    df = pd.read_csv(data_path)

    
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


    rename_map = {
        "date_reported": "date",
        "country": "location",
        "new_cases": "new_cases",
        "cumulative_cases": "total_cases",
        "new_deaths": "new_deaths",
        "cumulative_deaths": "total_deaths",
        "country_code": "country_code",
    }
    df.rename(columns=rename_map, inplace=True)


    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return df

# Main analysis block
if __name__ == "__main__":
    df = load_and_clean_data()
    print("Raw shape:", df.shape)


    if "location" in df.columns:
        country_name = "India"
        df_country = df[df["location"] == country_name].copy()
    else:
        df_country = df.copy()
        country_name = "Selected Region"

    df_country = df_country.sort_values("date")
    print("Country data shape:", df_country.shape)

    
    numeric_cols = df_country.select_dtypes(include="number").columns
    df_country[numeric_cols] = df_country[numeric_cols].fillna(0)


    if "new_cases" in df_country.columns:
        df_country["new_cases_7d_avg"] = df_country["new_cases"].rolling(window=7).mean()
    if "new_deaths" in df_country.columns:
        df_country["new_deaths_7d_avg"] = df_country["new_deaths"].rolling(window=7).mean()


    print("\nSummary statistics:")
    print(df_country.describe())


    plt.figure(figsize=(10, 5))
    plt.plot(df_country["date"], df_country["new_cases"], label="Daily New Cases", alpha=0.7)
    if "new_cases_7d_avg" in df_country.columns:
        plt.plot(df_country["date"], df_country["new_cases_7d_avg"], label="7-Day Average", linewidth=2)
    plt.title(f"Daily New COVID-19 Cases - {country_name}")
    plt.xlabel("Date")
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if "new_deaths" in df_country.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df_country["date"], df_country["new_deaths"], label="Daily New Deaths", color="red", alpha=0.7)
        if "new_deaths_7d_avg" in df_country.columns:
            plt.plot(df_country["date"], df_country["new_deaths_7d_avg"], label="7-Day Average", color="black", linewidth=2)
        plt.title(f"Daily New COVID-19 Deaths - {country_name}")
        plt.xlabel("Date")
        plt.ylabel("Number of Deaths")
        plt.legend()
        plt.tight_layout()
        plt.show()


    corr_cols = []
    if "new_cases" in df_country.columns:
        corr_cols.append("new_cases")
    corr_cols += [c for c in ["new_deaths", "total_cases", "total_deaths"] if c in df_country.columns]
    corr_cols = list(dict.fromkeys(corr_cols))

    if len(corr_cols) > 1:
        print("\nCorrelation matrix:")
        print(df_country[corr_cols].corr())

    if "new_cases" in df_country.columns:
        peak_cases = df_country["new_cases"].max()
        peak_date = df_country.loc[df_country["new_cases"].idxmax(), "date"]
        print(f"\nPeak daily new cases in {country_name}: {int(peak_cases)} on {peak_date.date()}")

    if "total_cases" in df_country.columns:
        total_cases = int(df_country["total_cases"].max())
        print(f"Total confirmed cases in {country_name}: {total_cases}")

    if "total_deaths" in df_country.columns:
        total_deaths = int(df_country["total_deaths"].max())
        print(f"Total deaths recorded in {country_name}: {total_deaths}")

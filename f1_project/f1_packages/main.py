from data_preparation import baseline_data_prep
from data_cleaning import baseline_cleaning


# Data preparation

df = baseline_data_prep()

# Data cleaning

df_baseline = baseline_cleaning(df)

df_baseline.to_csv("df_baseline2.csv")

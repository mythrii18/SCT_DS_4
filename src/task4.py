# TASK: TRAFFIC ACCIDENT ANALYSIS (INDIA, 2018–2023)
# DATASET: INDIA ROAD ACCIDENT DATASET (KAGGLE)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = (r"C:\Users\MYTHRI  MR\OneDrive\Desktop\SCT_DS_4\data\accident_prediction_india.csv")
df = pd.read_csv(file_path)
df = df.dropna(subset=["State Name", "Weather Conditions", "Road Type"], how="any")

# 1. TREND & GEOGRAPHY
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

yearly = df.groupby("Year")["Accident Severity"].count()
sns.lineplot(x=yearly.index, y=yearly.values, marker="o", color="red", ax=axes[0])
axes[0].set_title("Accidents in India (2018–2023)")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Accident Count")

state_accidents = df["State Name"].value_counts().head(10)
sns.barplot(x=state_accidents.values, y=state_accidents.index, palette="plasma", ax=axes[1])
axes[1].set_title("Top 10 States with Most Accidents")
axes[1].set_xlabel("Accident Count")
axes[1].set_ylabel("State")

plt.tight_layout()
plt.show()

# 2. ENVIRONMENT
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot(y="Weather Conditions", data=df,
              order=df["Weather Conditions"].value_counts().index,
              palette="coolwarm", ax=axes[0])
axes[0].set_title("Accidents by Weather Condition")

sns.countplot(x="Road Type", data=df,
              order=df["Road Type"].value_counts().index,
              palette="viridis", ax=axes[1])
axes[1].set_title("Accidents by Road Type")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()

# 3. TIMING & HUMAN FACTORS
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot(x="Time of Day", data=df,
              order=df["Time of Day"].value_counts().index,
              palette="pastel", ax=axes[0])
axes[0].set_title("Accidents by Time of Day")
axes[0].tick_params(axis="x", rotation=30)

sns.countplot(x="Alcohol Involvement", data=df, palette="Set2", ax=axes[1])
axes[1].set_title("Accidents by Alcohol Involvement")

plt.tight_layout()
plt.show()

# 4. SEVERITY & VEHICLES
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot(x="Accident Severity", data=df, palette="muted", ax=axes[0])
axes[0].set_title("Accidents by Severity")

sns.countplot(y="Vehicle Type Involved", data=df,
              order=df["Vehicle Type Involved"].value_counts().head(10).index,
              palette="Spectral", ax=axes[1])
axes[1].set_title("Top 10 Vehicle Types in Accidents")

plt.tight_layout()
plt.show()

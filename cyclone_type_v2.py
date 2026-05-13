import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

original_dataframe = pd.read_csv("cyclone_impact.csv")
df = original_dataframe.copy()

df = pd.get_dummies(df, columns=["impact_level"])
df["impact_level_Extreme"] = df["impact_level_Extreme"].astype(int)
df["impact_level_High"] = df["impact_level_High"].astype(int)
df["impact_level_Moderate"] = df["impact_level_Moderate"].astype(int)

scale = StandardScaler()
X = df[["wind_speed_kmh",
        "pressure_hpa",
        "rainfall_mm",
        "storm_surge_m",
        "cyclone_size_km",
        "movement_speed_kmh",
        "ocean_temperature_c",
        "impact_score"]]
df[X.columns] = scale.fit_transform(X)
##df["wind_speed_kmh"] = df2[:,0]
##df["pressure_hpa"] = df2[:,1]
##df["rainfall_mm"] = df2[:,2]
##df["storm_surge_m"] = df2[:,3]
##df["cyclone_size_km"] = df2[:,4]
##df["movement_speed_kmh"] = df2[:,5]
##df["ocean_temperature_c"] = df2[:,6]
##df["impact_score"] = df2[:,7]

X_scaled = df[["wind_speed_kmh",
        "pressure_hpa",
        "rainfall_mm",
        "storm_surge_m",
        "cyclone_size_km",
        "movement_speed_kmh",
        "ocean_temperature_c",
        "impact_score"]]
kmeans = KMeans(n_clusters=3 ,random_state=42 ,n_init=80)
df["cluster"] = kmeans.fit_predict(X_scaled)

plt.scatter(
    df["wind_speed_kmh"],
    df["pressure_hpa"],
    c=df["cluster"]
)

plt.xlabel("Wind Speed")
plt.ylabel("Pressure")
plt.title("K-Means Clusters")

plt.show()




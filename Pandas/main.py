import pandas as pd
import matplotlib.pyplot as plt

population = pd.read_csv("world_population.csv", index_col="CCA3")
continent = population["Continent"].unique()

population_column = list()
country_count = list()
for c in continent:
    s = population.loc[population["Continent"] == c]
    population_column.append(s["2022 Population"].sum())
    country_count.append(s.shape[0])

df = pd.DataFrame({"continent": continent, "population": population_column})
df1 = pd.DataFrame({"continent": continent, "countries": country_count})

df = df.sort_values("population")
df1 = df1.sort_values("countries")
print(df)
print()
print(df1)

plt.style.use("seaborn-v0_8")
plt.xlabel("Continents")
plt.ylabel("Population")
plt.bar(df["continent"], df["population"])
plt.show()



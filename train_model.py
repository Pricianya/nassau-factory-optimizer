import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("Nassau Candy Distributor.csv")

# -----------------------------
# CREATE LEAD TIME
# -----------------------------
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

df["Lead_Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

# -----------------------------
# ADD FACTORY
# -----------------------------
factory_map = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
}

df["Factory"] = df["Product Name"].map(factory_map)

# -----------------------------
# CLEAN DATA
# -----------------------------
df = df.dropna(subset=["Lead_Time", "Factory"])

# -----------------------------
# SELECT FEATURES
# -----------------------------
df = df[[
    "Sales",
    "Units",
    "Gross Profit",
    "Cost",
    "Ship Mode",
    "Region",
    "Product Name",
    "Factory",
    "Lead_Time"
]]

# -----------------------------
# ENCODING
# -----------------------------
df = pd.get_dummies(df)

# -----------------------------
# SPLIT
# -----------------------------
X = df.drop("Lead_Time", axis=1)
y = df["Lead_Time"]

# 🔥 SAVE COLUMNS
columns = X.columns

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestRegressor()
model.fit(X, y)

# -----------------------------
# SAVE FILES
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(columns, open("columns.pkl", "wb"))

print("✅ model.pkl and columns.pkl created successfully")
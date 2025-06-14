import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(path="data/personality_dataset.csv"):
    data = pd.read_csv(path)
    data.drop_duplicates(inplace=True)

    le = LabelEncoder()
    data["Stage_fear"] = le.fit_transform(data["Stage_fear"])
    data["Drained_after_socializing"] = le.fit_transform(data["Drained_after_socializing"])
    data["Personality"] = le.fit_transform(data["Personality"])

    X = data.drop(columns=["Personality"])
    y = data["Personality"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
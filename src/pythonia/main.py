import pandas as pd


def fill_age(row):
    if pd.isnull(row["Age"]):
        if row["Pclass"] == 1:
            return age_1
        if row["Pclass"] == 2:
            return age_2
        return age_3
    return row["Age"]


def fill_sex(sex):
    if sex == "male":
        return 1
    return 0


if __name__ == "__main__":
    df = pd.read_csv("src/pythonia/assets/Titanic-Dataset.csv")
    age_1 = df[df["Pclass"] == 1]["Age"].median()
    age_2 = df[df["Pclass"] == 2]["Age"].median()
    age_3 = df[df["Pclass"] == 3]["Age"].median()

    df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    df["Age"] = df.apply(fill_age, axis=1)
    df["Sex"] = df["Sex"].apply(fill_sex)

    df[list(pd.get_dummies(df["Embarked"]).columns)] = pd.get_dummies(df["Embarked"])
    df.drop("Embarked", axis=1, inplace=True)

    df.info()
    print(df.head())

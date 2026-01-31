import pandas as pd
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def fill_sex(sex):
    return sex - 1

def fill_age(birthday:str):
    if pd.isnull(birthday):
        return 18
    year = int(birthday.split('-')[0])
    current_year = 2026
    return current_year - year

def last_seen_to_int(last_seen:str):
    if pd.isnull(last_seen):
        return 0
    # example format: '2023-05-15T14:30:00'
    last_seen_dt = datetime.datetime.strptime(last_seen, '%Y-%m-%dT%H:%M:%S')
    current_dt = datetime.datetime.now()
    delta = current_dt - last_seen_dt
    return delta.days

def count_languages(langs:str):
    if pd.isnull(langs):
        return 0
    return len(langs.split(','))

def education_status(education_status:str):
    if pd.isnull(education_status):
        return 0
    status_mapping = {
        'student': 0,
        'graduated': 1,
    }
    return status_mapping.get(education_status.lower(), 0)


if __name__ == "__main__":
    df = pd.read_csv("src/pythonia/assets/world_of_code.csv")
    
    df.drop(["id", "has_photo", "has_mobile", "city", "followers_count", "relation", "occupation_name", "people_main", "career_start","career_end"], axis=1, inplace=True)
    
    dummy_education_form = pd.get_dummies(df["education_form"], prefix="education_form")
    df[list(dummy_education_form.columns)] = dummy_education_form

    dummy_occupation_type = pd.get_dummies(df["occupation_type"], prefix="occupation_type")
    df[list(dummy_occupation_type.columns)] = dummy_occupation_type


    df["sex"] = df["sex"].apply(fill_sex)
    df["age"] = df["bdate"].apply(fill_age)
    df["last_seen"] = df["last_seen"].apply(last_seen_to_int)
    df["langs_count"] = df["langs"].apply(count_languages)
    df["education_status"] = df["education_status"].apply(education_status)
    df.drop(["education_form", "bdate", "langs", "occupation_type"], axis=1, inplace=True)
    
    print(df.head())
    df.info()
    
    x = df.drop("result", axis=1)  # Datos de los usuarios
    y = df["result"]  # Variable objetivo
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    percent = accuracy_score(y_test, y_pred) * 100
    matrix = confusion_matrix(y_test, y_pred)

    print(percent)
    print(matrix)
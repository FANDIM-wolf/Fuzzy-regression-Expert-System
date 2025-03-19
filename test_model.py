#ЗАГРУЗКА МОДЕЛИ И ПРЕДСКАЗЫВАНИЕ
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from catboost import CatBoostRegressor

# Глобальные переменные для хранения состояния
label_encoders = {}
scalers = {}
model = None
feature_columns = []

# Функция для перевода баллов SAT в баллы ЕГЭ
def sat_to_ege(sat_score):
    """
    Переводит баллы SAT (400–1600) в баллы ЕГЭ (40–100).
    """
    sat_min, sat_max = 400, 1600
    ege_min, ege_max = 40, 100
    if not (sat_min <= sat_score <= sat_max):
        raise ValueError(f"Балл SAT должен быть в диапазоне {sat_min}–{sat_max}.")
    ege_score = ege_min + (sat_score - sat_min) * (ege_max - ege_min) / (sat_max - sat_min)
    return round(ege_score, 2)

def load_and_prepare_data(data_path):
    global label_encoders, scalers, feature_columns
    
    # Загрузка данных
    df = pd.read_csv(data_path)
    
    # Преобразование баллов SAT в баллы ЕГЭ
    df['Балл ЕГЭ'] = df['Балл SAT'].apply(sat_to_ege)
    df.drop(columns=['Балл SAT'], inplace=True)  # Удаляем старый столбец "Балл SAT"
    
    # Кодирование категориальных переменных
    categorical_columns = ['Пол', 'Область изучения', 'Текущий уровень работы', 'Предпринимательство']
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(df[col].unique())
        df[col] = le.transform(df[col])
        label_encoders[col] = le
    
    # Исключение столбцов с кластерами
    cluster_columns = [col for col in df.columns if '_cluster' in col or col == 'Cluster']
    X = df.drop(columns=['Идентификатор студента', 'Начальная зарплата', 'Годы до повышения', 
                         'Удовлетворенность карьерой', 'Баланс между работой и личной жизнью'] + cluster_columns)
    feature_columns = X.columns.tolist()  # Сохраняем только релевантные признаки
    y = df[['Начальная зарплата', 'Годы до повышения', 'Удовлетворенность карьерой', 'Баланс между работой и личной жизнью']]
    
    # Нормализация признаков
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    scalers['X'] = scaler_X
    
    # Масштабирование целевых переменных
    target_scalers = {
        'salary': MinMaxScaler(feature_range=(25000, 150000)),
        'promotion': MinMaxScaler(feature_range=(1, 5)),
        'satisfaction': MinMaxScaler(feature_range=(1, 10)),
        'balance': MinMaxScaler(feature_range=(1, 10))
    }
    
    y_scaled = np.hstack((
        target_scalers['salary'].fit_transform(y[['Начальная зарплата']]),
        target_scalers['promotion'].fit_transform(y[['Годы до повышения']]),
        target_scalers['satisfaction'].fit_transform(y[['Удовлетворенность карьерой']]),
        target_scalers['balance'].fit_transform(y[['Баланс между работой и личной жизнью']])
    ))
    
    scalers.update(target_scalers)
    
    return df  # Возвращаем DataFrame для использования в predict

def load_model(model_load_path="model.cbm"):
    """
    Загружает модель из файла.
    """
    global model
    model = CatBoostRegressor()
    model.load_model(model_load_path)
    print(f"Модель загружена из файла: {model_load_path}")

def predict(age, gender, hs_gpa, sat, uni_rank, uni_gpa, field_of_study, 
           internships, projects, certifications, soft_skills, networking, 
           job_offers, current_job_level, entrepreneurship):
    # Преобразование баллов SAT в баллы ЕГЭ
    ege_score = sat_to_ege(sat)
    
    # Кодирование входных данных
    input_data = {
        'Возраст': age,
        'Пол': label_encoders['Пол'].transform([gender])[0],
        'Средний балл в школе': hs_gpa,
        'Балл ЕГЭ': ege_score,  # Используем баллы ЕГЭ вместо SAT
        'Рейтинг университета': uni_rank,
        'Средний балл в университете': uni_gpa,
        'Область изучения': label_encoders['Область изучения'].transform([field_of_study])[0],
        'Количество стажировок': internships,
        'Количество проектов': projects,
        'Сертификаты': certifications,
        'Оценка мягких навыков': soft_skills,
        'Оценка сетевого взаимодействия': networking,
        'Предложения о работе': job_offers,
        'Текущий уровень работы': label_encoders['Текущий уровень работы'].transform([current_job_level])[0],
        'Предпринимательство': label_encoders['Предпринимательство'].transform([entrepreneurship])[0]
    }
    
    # Создание входного вектора в правильном порядке
    X_new = np.array([[input_data[col] for col in feature_columns]])
    
    # Нормализация и предсказание
    X_scaled = scalers['X'].transform(X_new)
    predictions = model.predict(X_scaled)
    
    # Обратное преобразование результатов
    salary = scalers['salary'].inverse_transform(predictions[:, 0].reshape(-1, 1))[0][0]
    promotion = scalers['promotion'].inverse_transform(predictions[:, 1].reshape(-1, 1))[0][0]
    satisfaction = scalers['satisfaction'].inverse_transform(predictions[:, 2].reshape(-1, 1))[0][0]
    balance = scalers['balance'].inverse_transform(predictions[:, 3].reshape(-1, 1))[0][0]
    
    # Проверка и ограничение значений в допустимых границах
    salary = max(25000, min(150000, salary))  # Ограничение зарплаты
    promotion = max(1, min(5, promotion))    # Ограничение лет до повышения
    satisfaction = max(1, min(10, satisfaction))  # Ограничение удовлетворенности
    balance = max(1, min(10, balance))       # Ограничение баланса
    
    return {
        'salary': salary,
        'promotion': promotion,
        'satisfaction': satisfaction,
        'balance': balance
    }

if __name__ == "__main__":
    # Загрузка данных для подготовки кодировщиков и масштабирования
    data_path = 'education_career_success_translated.csv'
    load_and_prepare_data(data_path)
    
    # Загрузка модели из файла
    load_model("model.cbm")
    
    # Предсказание параметров
    result = predict(
        age=21,
        gender='Male',
        hs_gpa=3.8,
        sat=1600,  # Балл SAT будет преобразован в балл ЕГЭ
        uni_rank=5,
        uni_gpa=3.5,
        field_of_study='Computer Science',
        internships=2,
        projects=1,
        certifications=3,
        soft_skills=8,
        networking=7,
        job_offers=1,
        current_job_level='Entry',
        entrepreneurship='No'
    )
    
    print("Predicted Values:")
    print(f"Starting Salary: ${result['salary']:.2f}")
    print(f"Years to Promotion: {result['promotion']:.1f}")
    print(f"Career Satisfaction: {result['satisfaction']:.1f}/10")
    print(f"Work-Life Balance: {result['balance']:.1f}/10")
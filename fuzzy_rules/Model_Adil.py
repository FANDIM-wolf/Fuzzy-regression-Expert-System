import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from catboost import CatBoostRegressor

# Глобальные переменные
label_encoders = {}
scalers = {}
model = None
feature_columns = []

def calculate_fuzzy_membership(value, min_val, max_val):
    """
    Вычисляет степень принадлежности значения к нечетким множествам.
    """
    k = (max_val - min_val) / 4
    k1 = min_val + k
    k2 = k1 + k
    k3 = k2 + k
    
    degree_low = max(0, min((value - min_val) / (k1 - min_val), (k2 - value) / (k2 - k1)))
    degree_medium = max(0, min((value - k1) / (k2 - k1), (k3 - value) / (k3 - k2)))
    degree_high = max(0, min((value - k2) / (k3 - k2), (max_val - value) / (max_val - k3)))
    
    # Определение категории
    if degree_low > degree_medium and degree_low > degree_high:
        category = "low"
    elif degree_medium > degree_low and degree_medium > degree_high:
        category = "medium"
    else:
        category = "high"
    
    return {
        'low': degree_low,
        'medium': degree_medium,
        'high': degree_high,
        'category': category
    }

def apply_fuzzy_rules(promotion, satisfaction, balance):
    """
    Применяет нечеткие правила для определения уровня зарплаты.
    Возвращает категорию зарплаты и степень уверенности.
    """
    # Нечеткие правила
    rules = [
        {'conditions': [('promotion', 'high'), ('satisfaction', 'high'), ('balance', 'high')], 'output': 'high'},
        {'conditions': [('promotion', 'medium'), ('satisfaction', 'high'), ('balance', 'high')], 'output': 'high'},
        {'conditions': [('promotion', 'low'), ('satisfaction', 'high'), ('balance', 'high')], 'output': 'medium'},
        {'conditions': [('promotion', 'high'), ('satisfaction', 'medium'), ('balance', 'high')], 'output': 'medium'},
        {'conditions': [('promotion', 'medium'), ('satisfaction', 'medium'), ('balance', 'medium')], 'output': 'medium'},
        {'conditions': [('promotion', 'low'), ('satisfaction', 'medium'), ('balance', 'medium')], 'output': 'low'},
        {'conditions': [('promotion', 'high'), ('satisfaction', 'low'), ('balance', 'low')], 'output': 'low'},
        {'conditions': [('promotion', 'medium'), ('satisfaction', 'low'), ('balance', 'low')], 'output': 'low'},
        {'conditions': [('promotion', 'low'), ('satisfaction', 'low'), ('balance', 'low')], 'output': 'low'},
        {'conditions': [('promotion', 'high'), ('satisfaction', 'high'), ('balance', 'medium')], 'output': 'high'}
    ]
    
    # Применение правил
    output_strength = {'low': 0.0, 'medium': 0.0, 'high': 0.0}
    
    for rule in rules:
        strengths = []
        for (var, term) in rule['conditions']:
            if var == 'promotion':
                mf = promotion
            elif var == 'satisfaction':
                mf = satisfaction
            elif var == 'balance':
                mf = balance
            else:
                mf = {'low': 0, 'medium': 0, 'high': 0}
            strengths.append(mf[term])
        rule_strength = min(strengths)
        output_term = rule['output']
        if output_strength[output_term] < rule_strength:
            output_strength[output_term] = rule_strength
    
    # Определение итоговой категории
    max_category = max(output_strength, key=output_strength.get)
    return max_category, output_strength[max_category]

def calculate_fuzzy_salary(promotion, satisfaction, balance):
    """
    Вычисляет зарплату на основе нечетких правил.
    """
    # Диапазон зарплаты
    salary_min, salary_max = 25000, 150000
    
    # Применение нечетких правил
    category, confidence = apply_fuzzy_rules(promotion, satisfaction, balance)
    
    # Определение зарплаты на основе категории
    if category == 'low':
        return salary_min + (salary_max - salary_min) * 0.25 * confidence
    elif category == 'medium':
        return salary_min + (salary_max - salary_min) * 0.5 * confidence
    else:
        return salary_min + (salary_max - salary_min) * 0.75 * confidence

def sat_to_ege(sat_score):
    """
    Переводит баллы SAT в баллы ЕГЭ.
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
    df['Балл ЕГЭ'] = df['SAT_Score'].apply(sat_to_ege)
    df.drop(columns=['SAT_Score'], inplace=True)
    
    # Кодирование категориальных переменных
    categorical_columns = ['Gender', 'Field_of_Study', 'Current_Job_Level', 'Entrepreneurship']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Исключение ненужных столбцов
    cluster_columns = [col for col in df.columns if '_cluster' in col or col == 'Cluster']
    X = df.drop(columns=['Student_ID', 'Starting_Salary', 'Career_Satisfaction', 
                         'Years_to_Promotion', 'Work_Life_Balance'] + cluster_columns)
    feature_columns = X.columns.tolist()
    y = df[['Starting_Salary', 'Years_to_Promotion', 'Career_Satisfaction', 'Work_Life_Balance']]
    
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
        target_scalers['salary'].fit_transform(y[['Starting_Salary']]),
        target_scalers['promotion'].fit_transform(y[['Years_to_Promotion']]),
        target_scalers['satisfaction'].fit_transform(y[['Career_Satisfaction']]),
        target_scalers['balance'].fit_transform(y[['Work_Life_Balance']])
    ))
    
    scalers.update(target_scalers)
    
    return df

def load_model(model_path="model.cbm"):
    """
    Загружает модель CatBoost из файла.
    """
    global model
    model = CatBoostRegressor()
    model.load_model(model_path)
    print(f"Модель загружена из файла: {model_path}")

def predict_for_test_data(test_data_path):
    """
    Предсказывает значения для тестовых данных.
    """
    test_df = pd.read_csv(test_data_path)
    
    # Преобразование баллов SAT в баллы ЕГЭ
    test_df['Балл ЕГЭ'] = test_df['SAT_Score'].apply(sat_to_ege)
    test_df.drop(columns=['SAT_Score'], inplace=True)
    
    # Кодирование категориальных переменных
    for col in ['Gender', 'Field_of_Study', 'Current_Job_Level', 'Entrepreneurship']:
        test_df[col] = label_encoders[col].transform(test_df[col])
    
    # Подготовка данных для предсказания
    X_new = test_df[feature_columns].values
    X_scaled = scalers['X'].transform(X_new)
    
    # Предсказание модели
    predictions = model.predict(X_scaled)
    
    # Обратное преобразование предсказаний
    salaries = scalers['salary'].inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
    promotions = scalers['promotion'].inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()
    satisfactions = scalers['satisfaction'].inverse_transform(predictions[:, 2].reshape(-1, 1)).flatten()
    balances = scalers['balance'].inverse_transform(predictions[:, 3].reshape(-1, 1)).flatten()
    
    # Добавление предсказаний в DataFrame
    test_df['Predicted_Salary'] = salaries
    test_df['Predicted_Years_to_Promotion'] = promotions
    test_df['Predicted_Career_Satisfaction'] = satisfactions
    test_df['Predicted_Work_Life_Balance'] = balances
    
    # Применение нечеткой логики для определения зарплаты
    fuzzy_salaries = []
    for i in range(len(test_df)):
        promotion = calculate_fuzzy_membership(promotions[i], 1, 5)
        satisfaction = calculate_fuzzy_membership(satisfactions[i], 1, 10)
        balance = calculate_fuzzy_membership(balances[i], 1, 10)
        fuzzy_salary = calculate_fuzzy_salary(promotion, satisfaction, balance)
        fuzzy_salaries.append(fuzzy_salary)
    
    test_df['Fuzzy_Salary'] = fuzzy_salaries
    
    # Гибридная зарплата (среднее между ML и нечеткой логикой)
    test_df['Hybrid_Salary'] = (test_df['Predicted_Salary'] + test_df['Fuzzy_Salary']) / 2
    
    # Применение нечеткой логики для категоризации
    parameters = {
        'Hybrid_Salary': (25000, 150000),
        'Predicted_Years_to_Promotion': (1, 5),
        'Predicted_Career_Satisfaction': (1, 10),
        'Predicted_Work_Life_Balance': (1, 10)
    }
    
    for param, (min_val, max_val) in parameters.items():
        test_df[f'{param}_Fuzzy'] = test_df[param].apply(
            lambda x: calculate_fuzzy_membership(x, min_val, max_val)
        )
    
    return test_df

def filter_results_by_user_preferences(df, user_preferences):
    """
    Фильтрует строки на основе предпочтений пользователя.
    """
    mask = True
    for param, categories in user_preferences.items():
        param_mask = df[f'{param}_Fuzzy'].apply(lambda x: x['category'] in categories)
        mask &= param_mask
    return df[mask]

if __name__ == "__main__":
    # Загрузка данных и подготовка
    data_path = 'education_career_success.csv'
    load_and_prepare_data(data_path)
    
    # Загрузка модели
    load_model("model.cbm")
    
    # Предсказание для тестовых данных
    test_data_path = 'test.csv'
    results = predict_for_test_data(test_data_path)
    
    # Сохранение всех данных с предсказанными категориями
    results.to_csv('all_predictions_with_fuzzy.csv', index=False)
    print("Все данные с предсказанными категориями сохранены в файл: all_predictions_with_fuzzy.csv")
    
    # Пользовательские предпочтения
    user_preferences = {
        'Hybrid_Salary': ['medium', 'low'],
        'Predicted_Years_to_Promotion': ['low', 'medium'],
        'Predicted_Career_Satisfaction': ['medium', 'high'],
        'Predicted_Work_Life_Balance': ['medium', 'high']
    }
    
    # Фильтрация строк по предпочтениям пользователя
    filtered_results = filter_results_by_user_preferences(results, user_preferences)
    
    # Сохранение отфильтрованных данных
    filtered_results.to_csv('filtered_results.csv', index=False)
    print("Отфильтрованные данные сохранены в файл: filtered_results.csv")
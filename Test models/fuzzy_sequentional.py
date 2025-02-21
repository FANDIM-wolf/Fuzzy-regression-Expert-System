# Загрузка библиотек
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from catboost import CatBoostRegressor, Pool
import skfuzzy as fuzz

# Загрузка данных из CSV файла
df = pd.read_csv('education_career_success.csv')

# Кодирование категориальных переменных
label_encoders = {}
for column in ['Gender', 'Field_of_Study', 'Current_Job_Level', 'Entrepreneurship']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Разделение данных на признаки и целевые переменные
X = df.drop(columns=['Student_ID', 'Starting_Salary', 'Years_to_Promotion', 'Career_Satisfaction', 'Work_Life_Balance'])
y = df[['Starting_Salary', 'Years_to_Promotion', 'Career_Satisfaction', 'Work_Life_Balance']]

# Сохраняем порядок признаков
X_columns = X.columns

# Нормализация данных
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Масштабирование целевых переменных в диапазон [0, 1]
salary_scaler = MinMaxScaler(feature_range=(25000, 150000))
promotion_scaler = MinMaxScaler(feature_range=(1, 5))
satisfaction_scaler = MinMaxScaler(feature_range=(1, 10))
balance_scaler = MinMaxScaler(feature_range=(1, 10))

y_scaled = np.hstack((
    salary_scaler.fit_transform(y[['Starting_Salary']]),
    promotion_scaler.fit_transform(y[['Years_to_Promotion']]),
    satisfaction_scaler.fit_transform(y[['Career_Satisfaction']]),
    balance_scaler.fit_transform(y[['Work_Life_Balance']])
))

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Создание и обучение модели CatBoost
model = CatBoostRegressor(iterations=2000, learning_rate=0.1, depth=10, loss_function='MultiRMSE', verbose=0)
model.fit(X_train, y_train)

# Предсказания модели
predictions = model.predict(X_test)

# Обратное масштабирование предсказанных значений
salary_preds = salary_scaler.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
promotion_preds = promotion_scaler.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()
satisfaction_preds = satisfaction_scaler.inverse_transform(predictions[:, 2].reshape(-1, 1)).flatten()
balance_preds = balance_scaler.inverse_transform(predictions[:, 3].reshape(-1, 1)).flatten()

# Проверка границ
salary_in_range = np.all((salary_preds >= 25000) & (salary_preds <= 150000))
promotion_in_range = np.all((promotion_preds >= 1) & (promotion_preds <= 5))
satisfaction_in_range = np.all((satisfaction_preds >= 1) & (satisfaction_preds <= 10))
balance_in_range = np.all((balance_preds >= 1) & (balance_preds <= 10))

print("All salary predictions in range:", salary_in_range)
print("All promotion predictions in range:", promotion_in_range)
print("All satisfaction predictions in range:", satisfaction_in_range)
print("All balance predictions in range:", balance_in_range)

# Функция для проверки и дополнения примера данных
def complete_example_data(example_data, feature_order, df):
    # Создаем полный словарь с признаками
    full_example = {}
    for col in feature_order:
        if col in example_data:
            full_example[col] = example_data[col]
        else:
            # Если признак отсутствует, используем среднее значение из обучающего набора
            if df[col].dtype == 'object':  # Категориальный признак
                full_example[col] = df[col].mode()[0]  # Наиболее частое значение
            else:  # Числовой признак
                full_example[col] = df[col].mean()  # Среднее значение
    return full_example

# Обновляем функцию preprocess_example_data
def preprocess_example_data(example_data, label_encoders, feature_order, df):
    # Проверяем и дополняем данные
    processed_data = complete_example_data(example_data, feature_order, df)
    
    # Преобразуем категориальные переменные
    for key, encoder in label_encoders.items():
        if key in processed_data:
            processed_data[key] = encoder.transform([processed_data[key]])[0]
    
    # Создаем массив в том же порядке, что и обучающий набор
    example_array = np.array([processed_data[col] for col in feature_order]).reshape(1, -1)
    return example_array

# Обновляем функцию predict_example
def predict_example(model, scalers, encoders, example_data, feature_order, df):
    # Преобразование категориальных переменных
    example_array = preprocess_example_data(example_data, encoders, feature_order, df)
    
    # Нормализация данных
    example_scaled = scalers['X'].transform(example_array)
    
    # Предсказание модели
    prediction_scaled = model.predict(example_scaled)
    
    # Обратное масштабирование предсказанных значений
    salary_pred = scalers['salary'].inverse_transform(prediction_scaled[:, 0].reshape(-1, 1)).flatten()[0]
    promotion_pred = scalers['promotion'].inverse_transform(prediction_scaled[:, 1].reshape(-1, 1)).flatten()[0]
    satisfaction_pred = scalers['satisfaction'].inverse_transform(prediction_scaled[:, 2].reshape(-1, 1)).flatten()[0]
    balance_pred = scalers['balance'].inverse_transform(prediction_scaled[:, 3].reshape(-1, 1)).flatten()[0]
    
    return salary_pred, promotion_pred, satisfaction_pred, balance_pred

# Пример данных для предсказания с добавлением признака 'Age'
example_data = {
    'Age': 27 , # Добавляем возраст
    'Gender': 'Male',
    'Field_of_Study': 'Medicine',
    'GPA': 3.6,
    'Internships': 2,
    'Projects': 3,
    'Extra_Curricular_Activities': 1,
    'Research_Publications': 7,
    'Networking': 6,
    'Skills': 2,
    'Current_Job_Level': 'Mid',
    'Entrepreneurship': 'No',
 
}

# Словарь скалеров и кодировщиков
scalers = {
    'X': scaler_X,
    'salary': salary_scaler,
    'promotion': promotion_scaler,
    'satisfaction': satisfaction_scaler,
    'balance': balance_scaler
}

# Предсказание для примера
salary, promotion, satisfaction, balance = predict_example(model, scalers, label_encoders, example_data, X_columns, df)
print(f"Predicted Starting Salary: {salary:.2f}")
print(f"Predicted Years to Promotion: {promotion:.2f}")
print(f"Predicted Career Satisfaction: {satisfaction:.2f}")
print(f"Predicted Work-Life Balance: {balance:.2f}")

# Применение нечеткой логики
def fuzzy_classification(salary, promotion, satisfaction, balance):
    # Функции принадлежности
    salary_range = np.arange(25000, 151000, 1000)
    promotion_range = np.arange(1, 6, 1)
    satisfaction_range = np.arange(1, 11, 1)
    balance_range = np.arange(1, 11, 1)
    
    salary_hi = fuzz.trimf(salary_range, [60000, 75000, 90000])
    promotion_lo = fuzz.trimf(promotion_range, [1, 2, 3])
    satisfaction_hi = fuzz.trimf(satisfaction_range, [7, 8, 9])
    balance_hi = fuzz.trimf(balance_range, [7, 8, 9])
    
    # Определение степени принадлежности
    salary_level = fuzz.interp_membership(salary_range, salary_hi, salary)
    promotion_level = fuzz.interp_membership(promotion_range, promotion_lo, promotion)
    satisfaction_level = fuzz.interp_membership(satisfaction_range, satisfaction_hi, satisfaction)
    balance_level = fuzz.interp_membership(balance_range, balance_hi, balance)
    
    # Правила
    rule1 = np.fmin(salary_level, np.fmin(promotion_level, np.fmin(satisfaction_level, balance_level)))
    rule2 = np.fmin(salary_level, np.fmin(promotion_level, np.fmin(satisfaction_level, 1 - balance_level)))
    rule3 = np.fmin(1 - salary_level, np.fmin(1 - promotion_level, np.fmin(satisfaction_level, balance_level)))
    rule4 = np.fmin(1 - salary_level, np.fmin(1 - promotion_level, np.fmin(1 - satisfaction_level, 1 - balance_level)))
    rule5 = np.fmin(salary_level, np.fmin(1 - promotion_level, np.fmin(satisfaction_level, balance_level)))
    rule6 = np.fmin(1 - salary_level, np.fmin(promotion_level, np.fmin(1 - satisfaction_level, balance_level)))
    rule7 = np.fmin(salary_level, np.fmin(promotion_level, np.fmin(1 - satisfaction_level, 1 - balance_level)))
    rule8 = np.fmin(1 - salary_level, np.fmin(1 - promotion_level, np.fmin(satisfaction_level, 1 - balance_level)))
    
    # Классификация
    if np.max(rule1) > 0.5:
        return "(Q1 и Q2) - высокий успех в карьере и высокий баланс между работой и личной жизнью"
    elif np.max(rule2) > 0.5:
        return "(Q1 и !Q2) - высокий успех в карьере, но низкий баланс между работой и личной жизнью"
    elif np.max(rule3) > 0.5:
        return "(!Q1 и Q2) - низкий успех в карьере, но высокий баланс между работой и личной жизнью"
    elif np.max(rule4) > 0.5:
        return "(!Q1 и !Q2) - низкий успех в карьере и низкий баланс между работой и личной жизнью"
    elif np.max(rule5) > 0.5:
        return "(Q1 и Q2) - средний успех в карьере и средний баланс между работой и личной жизнью"
    elif np.max(rule6) > 0.5:
        return "(Q1 и !Q2) - средний успех в карьере, но низкий баланс между работой и личной жизнью"
    elif np.max(rule7) > 0.5:
        return "(!Q1 и Q2) - низкий успех в карьере, но средний баланс между работой и личной жизнью"
    elif np.max(rule8) > 0.5:
        return "(!Q1 и !Q2) - низкий успех в карьере и низкий баланс между работой и личной жизнью"

# Классификация предсказания
classification = fuzzy_classification(salary, promotion, satisfaction, balance)
print(classification)
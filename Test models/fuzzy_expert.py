import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from catboost import CatBoostRegressor, Pool
import skfuzzy as fuzz
import optuna

class FuzzyExpert:
    def __init__(self, data_path):
        # Загрузка данных
        self.df = pd.read_csv(data_path)
        self.label_encoders = {}
        self._prepare_data()
        self._optimize_hyperparameters_with_optuna()

    def _prepare_data(self):
        # Кодирование категориальных переменных
        for column in ['Gender', 'Field_of_Study', 'Current_Job_Level', 'Entrepreneurship']:
            self.label_encoders[column] = LabelEncoder()
            self.df[column] = self.label_encoders[column].fit_transform(self.df[column])

        # Разделение на признаки и целевые переменные
        self.X = self.df.drop(columns=['Student_ID', 'Starting_Salary', 'Years_to_Promotion', 'Career_Satisfaction', 'Work_Life_Balance'])
        self.y = self.df[['Starting_Salary', 'Years_to_Promotion', 'Career_Satisfaction', 'Work_Life_Balance']]

        # Нормализация данных
        self.scaler_X = MinMaxScaler()
        self.X_scaled = self.scaler_X.fit_transform(self.X)

        # Масштабирование целевых переменных
        self.salary_scaler = MinMaxScaler(feature_range=(25000, 150000))
        self.promotion_scaler = MinMaxScaler(feature_range=(1, 5))
        self.satisfaction_scaler = MinMaxScaler(feature_range=(1, 10))
        self.balance_scaler = MinMaxScaler(feature_range=(1, 10))

        self.y_scaled = np.hstack((
            self.salary_scaler.fit_transform(self.y[['Starting_Salary']]),
            self.promotion_scaler.fit_transform(self.y[['Years_to_Promotion']]),
            self.satisfaction_scaler.fit_transform(self.y[['Career_Satisfaction']]),
            self.balance_scaler.fit_transform(self.y[['Work_Life_Balance']])
        ))

        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y_scaled, test_size=0.2, random_state=42
        )

    def _objective(self, trial):
        # Определение пространства поиска гиперпараметров
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'verbose': 0
        }

        # Создание и обучение модели
        model = CatBoostRegressor(**params, loss_function='MultiRMSE')

        # Кросс-валидация
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            train_data = Pool(data=X_train_fold, label=y_train_fold)
            val_data = Pool(data=X_val_fold, label=y_val_fold)

            model.fit(train_data, eval_set=val_data, verbose=0)
            predictions = model.predict(X_val_fold)

            # Вычисление ошибки MultiRMSE
            error = np.sqrt(np.mean((predictions - y_val_fold) ** 2))
            scores.append(error)

        # Средняя ошибка кросс-валидации
        avg_error = np.mean(scores)
        return avg_error

    def _optimize_hyperparameters_with_optuna(self):
        # Настройка Optuna для поиска гиперпараметров
        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective, n_trials=20, show_progress_bar=True)

        print("Лучшие гиперпараметры:", study.best_params)
        print("Лучшее значение ошибки:", study.best_value)

        # Создание модели с лучшими параметрами
        self.model = CatBoostRegressor(**study.best_params, loss_function='MultiRMSE', verbose=0)
        self.model.fit(self.X_train, self.y_train)

    def predict_values(self, age, gender, hs_gpa, sat, uni_rank, uni_gpa, field_of_study, internships, projects, certifications, soft_skills, networking, job_offers, current_job_level, entrepreneurship):
        # Кодирование категориальных переменных
        gender_encoded = self.label_encoders['Gender'].transform([gender])[0]
        field_of_study_encoded = self.label_encoders['Field_of_Study'].transform([field_of_study])[0]
        current_job_level_encoded = self.label_encoders['Current_Job_Level'].transform([current_job_level])[0]
        entrepreneurship_encoded = self.label_encoders['Entrepreneurship'].transform([entrepreneurship])[0]

        # Создание массива с уже закодированными категориальными признаками
        X_new = np.array([
            [age, hs_gpa, sat, uni_rank, uni_gpa, internships, projects, certifications, soft_skills,
             networking, job_offers, gender_encoded, field_of_study_encoded, current_job_level_encoded, entrepreneurship_encoded]
        ])

        # Нормализация данных
        X_new_scaled = self.scaler_X.transform(X_new)

        # Предсказание модели
        predictions = self.model.predict(X_new_scaled)

        # Обратное масштабирование предсказанных значений
        salary_pred = self.salary_scaler.inverse_transform(predictions[:, 0].reshape(-1, 1))[0][0]
        promotion_pred = self.promotion_scaler.inverse_transform(predictions[:, 1].reshape(-1, 1))[0][0]
        satisfaction_pred = self.satisfaction_scaler.inverse_transform(predictions[:, 2].reshape(-1, 1))[0][0]
        balance_pred = self.balance_scaler.inverse_transform(predictions[:, 3].reshape(-1, 1))[0][0]

        return salary_pred, promotion_pred, satisfaction_pred, balance_pred

    def fuzzy_classification(self, salary, promotion, satisfaction, balance):
        # Функции принадлежности
        salary_range = np.arange(25000, 151000, 1000)
        promotion_range = np.arange(1, 6, 1)
        satisfaction_range = np.arange(1, 11, 1)
        balance_range = np.arange(1, 11, 1)

        # Определение функций принадлежности для low, medium и high
        salary_low = fuzz.trimf(salary_range, [25000, 40000, 55000])
        salary_medium = fuzz.trimf(salary_range, [40000, 60000, 80000])
        salary_high = fuzz.trimf(salary_range, [60000, 75000, 90000])

        promotion_low = fuzz.trimf(promotion_range, [1, 2, 3])
        promotion_medium = fuzz.trimf(promotion_range, [2, 3, 4])
        promotion_high = fuzz.trimf(promotion_range, [3, 4, 5])

        satisfaction_low = fuzz.trimf(satisfaction_range, [1, 3, 5])
        satisfaction_medium = fuzz.trimf(satisfaction_range, [3, 5, 7])
        satisfaction_high = fuzz.trimf(satisfaction_range, [5, 7, 9])

        balance_low = fuzz.trimf(balance_range, [1, 3, 5])
        balance_medium = fuzz.trimf(balance_range, [3, 5, 7])
        balance_high = fuzz.trimf(balance_range, [5, 7, 9])

        # Определение степени принадлежности
        salary_level_low = fuzz.interp_membership(salary_range, salary_low, salary)
        salary_level_medium = fuzz.interp_membership(salary_range, salary_medium, salary)
        salary_level_high = fuzz.interp_membership(salary_range, salary_high, salary)

        promotion_level_low = fuzz.interp_membership(promotion_range, promotion_low, promotion)
        promotion_level_medium = fuzz.interp_membership(promotion_range, promotion_medium, promotion)
        promotion_level_high = fuzz.interp_membership(promotion_range, promotion_high, promotion)

        satisfaction_level_low = fuzz.interp_membership(satisfaction_range, satisfaction_low, satisfaction)
        satisfaction_level_medium = fuzz.interp_membership(satisfaction_range, satisfaction_medium, satisfaction)
        satisfaction_level_high = fuzz.interp_membership(satisfaction_range, satisfaction_high, satisfaction)

        balance_level_low = fuzz.interp_membership(balance_range, balance_low, balance)
        balance_level_medium = fuzz.interp_membership(balance_range, balance_medium, balance)
        balance_level_high = fuzz.interp_membership(balance_range, balance_high, balance)

        # Правила
        rule1 = np.fmin(salary_level_high, np.fmin(promotion_level_high, np.fmin(satisfaction_level_high, balance_level_high)))
        rule2 = np.fmin(salary_level_high, np.fmin(promotion_level_high, np.fmin(satisfaction_level_high, balance_level_medium)))
        rule3 = np.fmin(salary_level_high, np.fmin(promotion_level_high, np.fmin(satisfaction_level_medium, balance_level_medium)))
        rule4 = np.fmin(salary_level_high, np.fmin(promotion_level_medium, np.fmin(satisfaction_level_medium, balance_level_medium)))
        rule5 = np.fmin(salary_level_high, np.fmin(promotion_level_high, np.fmin(satisfaction_level_medium, balance_level_medium)))
        rule6 = np.fmin(salary_level_medium, np.fmin(promotion_level_high, np.fmin(satisfaction_level_medium, balance_level_medium)))
        rule7 = np.fmin(salary_level_low, np.fmin(promotion_level_low, np.fmin(satisfaction_level_low, balance_level_low)))
        rule8 = np.fmin(salary_level_medium, np.fmin(promotion_level_low, np.fmin(satisfaction_level_low, balance_level_low)))
        rule9 = np.fmin(salary_level_medium, np.fmin(promotion_level_medium, np.fmin(satisfaction_level_low, balance_level_low)))
        rule10 = np.fmin(salary_level_medium, np.fmin(promotion_level_medium, np.fmin(satisfaction_level_low, balance_level_low)))
        rule11 = np.fmin(salary_level_medium, np.fmin(promotion_level_medium, np.fmin(satisfaction_level_medium, balance_level_low)))
        rule12 = np.fmin(salary_level_medium, np.fmin(promotion_level_medium, np.fmin(satisfaction_level_medium, balance_level_medium)))
        rule13 = np.fmin(salary_level_medium, np.fmin(promotion_level_medium, np.fmin(satisfaction_level_medium, balance_level_medium)))

        # Классификация
        if np.max(rule1) > 0.5:
            return "(Q1 и Q2) - высокий успех в карьере и высокий баланс между работой и личной жизнью"
        elif np.max(rule2) > 0.5 or np.max(rule3) > 0.5 or np.max(rule4) > 0.5 or np.max(rule5) > 0.5:
            return "(Q1 и Q2) - высокий успех в карьере и средний баланс между работой и личной жизнью"
        elif np.max(rule6) > 0.5:
            return "(!Q1 и Q2) - низкий успех в карьере, но высокий баланс между работой и личной жизнью"
        elif np.max(rule7) > 0.5 or np.max(rule8) > 0.5 or np.max(rule9) > 0.5 or np.max(rule10) > 0.5:
            return "(!Q1 и !Q2) - низкий успех в карьере и низкий баланс между работой и личной жизнью"
        elif np.max(rule11) > 0.5 or np.max(rule12) > 0.5 or np.max(rule13) > 0.5:
            return "(Q1 и !Q2) - высокий успех в карьере, но низкий баланс между работой и личной жизнью"
        else:
            return "Неопределенный результат"

    def test_model(self):
        # Тестирование модели на тестовых данных
        predictions = self.model.predict(self.X_test)

        # Обратное масштабирование предсказанных значений
        salary_preds = self.salary_scaler.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
        promotion_preds = self.promotion_scaler.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()
        satisfaction_preds = self.satisfaction_scaler.inverse_transform(predictions[:, 2].reshape(-1, 1)).flatten()
        balance_preds = self.balance_scaler.inverse_transform(predictions[:, 3].reshape(-1, 1)).flatten()

        # Проверка границ
        salary_in_range = np.all((salary_preds >= 25000) & (salary_preds <= 150000))
        promotion_in_range = np.all((promotion_preds >= 1) & (promotion_preds <= 5))
        satisfaction_in_range = np.all((satisfaction_preds >= 1) & (satisfaction_preds <= 10))
        balance_in_range = np.all((balance_preds >= 1) & (balance_preds <= 10))

        if not salary_in_range or not promotion_in_range or not satisfaction_in_range or not balance_in_range:
            print("Предсказания выходят за допустимые границы!")
        else:
            print("Тестирование завершено успешно.")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
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
        self.model = CatBoostRegressor(**study.best_params, verbose=0)
        self.model.fit(self.X_train, self.y_train, verbose=0)

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

        print("All salary predictions in range:", salary_in_range)
        print("All promotion predictions in range:", promotion_in_range)
        print("All satisfaction predictions in range:", satisfaction_in_range)
        print("All balance predictions in range:", balance_in_range)

# Пример использования
if __name__ == "__main__":
    fuzzy_expert = FuzzyExpert('education_career_success.csv')

    # Пример использования
    example_predictions = fuzzy_expert.predict_values(
        27, 'Male', 3.6, 1300, 75, 3.5, 'Medicine', 2, 3, 1, 7, 6, 2, 'Mid', 'No'
    )
    salary, promotion, satisfaction, balance = example_predictions

    print(f"Predicted Starting Salary: {salary:.2f}")
    print(f"Predicted Years to Promotion: {promotion:.2f}")
    print(f"Predicted Career Satisfaction: {satisfaction:.2f}")
    print(f"Predicted Work-Life Balance: {balance:.2f}")

    # Классификация предсказания
    classification = fuzzy_expert.fuzzy_classification(salary, promotion, satisfaction, balance)
    print(classification)

    # Тестирование модели
    fuzzy_expert.test_model()

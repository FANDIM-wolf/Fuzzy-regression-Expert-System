"""Схема данных и константы предметной области."""

from __future__ import annotations

# Канонические (английские) имена колонок исходного датасета.
ID_COLUMN = "Student_ID"

CATEGORICAL_FEATURES = [
    "Gender",
    "Field_of_Study",
    "Current_Job_Level",
    "Entrepreneurship",
]

NUMERIC_FEATURES = [
    "Age",
    "High_School_GPA",
    "SAT_Score",
    "University_Ranking",
    "University_GPA",
    "Internships_Completed",
    "Projects_Completed",
    "Certifications",
    "Soft_Skills_Score",
    "Networking_Score",
    "Job_Offers",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = [
    "Starting_Salary",
    "Years_to_Promotion",
    "Career_Satisfaction",
    "Work_Life_Balance",
]

# Допустимые диапазоны целевых переменных (используются для клиппинга
# предсказаний и для построения нечётких разбиений).
TARGET_RANGES = {
    "Starting_Salary": (25_000.0, 150_000.0),
    "Years_to_Promotion": (1.0, 5.0),
    "Career_Satisfaction": (1.0, 10.0),
    "Work_Life_Balance": (1.0, 10.0),
}

# Русские заголовки из education_career_success_translated.csv -> канонические.
RU_TO_EN = {
    "Идентификатор студента": "Student_ID",
    "Возраст": "Age",
    "Пол": "Gender",
    "Средний балл в школе": "High_School_GPA",
    "Балл SAT": "SAT_Score",
    "Рейтинг университета": "University_Ranking",
    "Средний балл в университете": "University_GPA",
    "Область изучения": "Field_of_Study",
    "Количество стажировок": "Internships_Completed",
    "Количество проектов": "Projects_Completed",
    "Сертификаты": "Certifications",
    "Оценка мягких навыков": "Soft_Skills_Score",
    "Оценка сетевого взаимодействия": "Networking_Score",
    "Предложения о работе": "Job_Offers",
    "Начальная зарплата": "Starting_Salary",
    "Удовлетворенность карьерой": "Career_Satisfaction",
    "Годы до повышения": "Years_to_Promotion",
    "Текущий уровень работы": "Current_Job_Level",
    "Баланс между работой и личной жизнью": "Work_Life_Balance",
    "Предпринимательство": "Entrepreneurship",
}

# Колонки, которые нельзя подавать в модель: они получены из целевых
# переменных (утечка целевого признака) либо являются идентификаторами.
LEAKY_COLUMN_MARKERS = ("_cluster", "Cluster")

SAT_RANGE = (400.0, 1600.0)
EGE_RANGE = (40.0, 100.0)

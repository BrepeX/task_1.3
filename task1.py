import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка данных
data = pd.read_csv('Fish.csv')

# Разведочный анализ данных (EDA)
print("Первые строки датасета:")
print(data.head())
print("\nПоследние строки датасета:")
print(data.tail())
print("\nТип данных:")
print(data.dtypes)
print("\nРазмер датасета:")
print(data.shape)
print("\nПроверка на пропуски:")
print(data.isnull().sum())

# Категоризация нечисловых признаков
data = pd.get_dummies(data)

# Описательная статистика
print("\nОписательная статистика:")
print(data.describe())

# Тепловая карта корреляции
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Тепловая карта корреляции')
plt.show()

# График попарных значений признаков
sns.pairplot(data)
plt.show()

# Стандартизация признаков
scaler = StandardScaler()
X = data.drop('Weight', axis=1)
y = data['Weight']
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Коэффициенты модели
print("\nКоэффициенты модели линейной регрессии:")
print(model.coef_)

# Оценка качества модели
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)``

print("\nКачество модели на обучающей выборке:")
print("MSE:", mean_squared_error(y_train, y_train_pred))
print("R2:", r2_score(y_train, y_train_pred))
print("MAE:", mean_absolute_error(y_train, y_train_pred))

print("\nКачество модели на тестовой выборке:")
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("R2:", r2_score(y_test, y_test_pred))
print("MAE:", mean_absolute_error(y_test, y_test_pred))



# Task 2

X = data.drop('Species', axis=1)
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for max_depth in [3, 5]:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Оценка качества модели
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"\nКачество модели деревьев решений (max_depth={max_depth}) на обучающей выборке:")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Precision:", precision_score(y_train, y_train_pred, average='weighted'))
    print("Recall:", recall_score(y_train, y_train_pred, average='weighted'))
    print("F1 Score:", f1_score(y_train, y_train_pred, average='weighted'))

    print(f"\nКачество модели деревьев решений (max_depth={max_depth}) на тестовой выборке:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_test_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_test_pred, average='weighted'))

# Выводы о лучших параметрах модели
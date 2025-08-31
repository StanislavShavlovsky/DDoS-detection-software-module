import pandas as pd

# Загрузка датасета
data = pd.read_csv(r'D:\CIC-DDoS2019 Dataset\cicddos2019_dataset.csv')


print("Общая информация о датасете:")
print(data.info())


print("\nПервые несколько строк датасета:")
print(data.head())


print("\nСтатистика по числовым признакам:")
print(data.describe())


print("\nУникальные значения в категориальных столбцах:")
for column in data.select_dtypes(include=['object']).columns:
    print(f"{column}: {data[column].unique()}")


print("\nПроверка наличия пропущенных значений:")
print(data.isnull().sum())


print("\nПроверка наличия дубликатов:")
print("Количество дубликатов:", data.duplicated().sum())


if 'Label' in data.columns:
    print("\nБаланс классов:")
    print(data['Label'].value_counts())


print("\nИнформация о каждом столбце:")
print(data.info(verbose=True))



















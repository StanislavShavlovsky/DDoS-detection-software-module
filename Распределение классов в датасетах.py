import pandas as pd
import matplotlib.pyplot as plt

def class_distribution(dataset, class_column):

    class_counts = dataset[class_column].value_counts()
    total_samples = dataset.shape[0]
    unique_classes = dataset[class_column].nunique()
    print(f"Общее количество строк: {total_samples}")
    print(f"Общее количество классов: {unique_classes}")
    print("Распределение классов:")
    for class_label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"Класс {class_label}: {count} примеров ({percentage:.2f}%)")

    # Построение графика
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar', color='skyblue')
    plt.title('Распределение классов')
    plt.xlabel('Класс')
    plt.ylabel('Количество классов')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


dataset = pd.read_csv("D:\downloads\ML-EdgeIIoT-dataset.csv") # можем указывать путь к любому другому датасету

class_distribution(dataset, "Attack_type")


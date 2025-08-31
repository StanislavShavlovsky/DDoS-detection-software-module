import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_recall_curve, average_precision_score


data = pd.read_csv(r"D:\downloads\datasets\combined_data_IoT_Fridge.csv")


print(data.head())


data.dropna(subset=['type'], inplace=True)


y = data['type']  # Целевая переменная - метка атаки или нормального трафика
X = data.drop(['type'], axis=1)  # Признаки - характеристики сетевого трафика


encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)


le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)


decision_tree_classifier = DecisionTreeClassifier()  # Используем дерево принятия решений
decision_tree_classifier.fit(X_train, y_train)


y_pred = decision_tree_classifier.predict(X_test)



class_names = le.classes_


plt.figure(figsize=(16, 8))  
plt.subplot(1, 2, 2)
sns.countplot(x=y_train, order=np.arange(len(class_names)))
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, fontsize=10)  # Установка большего размера шрифта


plt.subplot(1, 2, 2)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)  # Установка имен классов напрямую
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(fontsize=10)  # Установка большего размера шрифта
plt.yticks(fontsize=10)  # Установка большего размера шрифта

plt.tight_layout()
plt.show()


# дополнительные метрики
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# результаты
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


data = pd.read_csv(r'D:\CIC-DDoS2019 Dataset\cicddos2019_dataset.csv')


data.dropna(subset=['Label'], inplace=True)


total_rows = data.shape[0]
print("Общее количество строк в наборе данных:", total_rows)


y = data['Label']  
X = data.drop(['Label'], axis=1)  


encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


test_classes = np.unique(y_test)

test_classes_text = le.inverse_transform(test_classes)
print("Классы, которые будут в тестовой выборке:", test_classes_text)

svm = SVC(kernel='poly', probability=True) 
svm.fit(X_train, y_train)  


y_pred_proba = svm.predict_proba(X_test)


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(test_classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure(figsize=(8, 6))
for i in range(len(test_classes)):
    plt.plot(fpr[i], tpr[i], label='ROC-кривая (класс {}) (AUC = {:.2f})'.format(test_classes_text[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительная оценка')
plt.ylabel('Истинно положительная оценка')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.show()


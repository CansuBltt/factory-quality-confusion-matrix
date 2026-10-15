import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
#satir=samples sutun=features
x,y=make_classification(
    n_samples=4000,
    n_features=10,
    n_informative=6,
    n_classes=4,
    weights=[0.70,0.12,0.10,0.08],
    random_state=42
)
class_name=["saglam","cizik","leke","kirik"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.25,random_state=42,stratify=y
)
print(f"Egitim veri sayisi: {x_train.shape[0]}")
print(f"Test veri sayisi: {x_test.shape[0]}")
rf_model=RandomForestClassifier(n_estimators=100,max_depth=4 ,random_state=42)
rf_model.fit(x_train,y_train)
y_pred=rf_model.predict(x_test)

print("\n Detayli  siniflandirma raporu")
print(classification_report(y_test,y_pred,target_names=class_name))

# Confusion matrix hesaplama
cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Greens",
    xticklabels=class_name,
    yticklabels=class_name
)
plt.xlabel("Tahmin Edilen Sinif")
plt.ylabel("Gercek Sinif")
plt.title("Fabrika urun Kalite - Confusion Matrix")
plt.show()


feature_names = [f"Sensor_{i+1}" for i in range(10)]

#  Önem skorlarını alıp sıralıyoruz
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

#  Grafiği çizdiriyoruz
plt.figure(figsize=(10, 5))
sns.barplot(
    x=importances[indices], 
    y=[feature_names[i] for i in indices], 
    palette="viridis"
)
plt.title("Kaliteyi Etkileyen En Önemli Sensörler (Feature Importance)")
plt.xlabel("Önem Skoru")
plt.ylabel("Sensörler")
# Grafikteki başlıkların ve eksen etiketlerinin dışarı taşmasını veya çakışmasını önler.
plt.tight_layout()

plt.show()
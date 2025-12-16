# ML-in-Engineering-Cost-Feasibility
The project idea is to develop a predictive program for future risks in engineering projects. This project is based on the principles of cybersecurity risk management
# Risk Treatment Decision - Python Implementation

هذا المشروع يوضح كيفية استخدام **Random Forest** لاتخاذ قرارات إدارة المخاطر في المشاريع الهندسية، مع تحليل الحساسية وتوضيح أهمية الميزات.

## الكود الرئيسي

```python
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# load the dataset
df = pd.read_csv("/content/Engineering_Cost_Feasibility_Dataset.csv")

# print first 5 rows
print("First 5 rows of the dataset:\n", df.head())

# drop missing values
df = df.dropna() 

# decision logic function
def decide_treatment(impact, likelihood):
    if impact >= 70 and likelihood >= 70:
        return "Avoid"
    elif 40 <= impact < 70 and 40 <= likelihood < 70:
        return "Mitigate"
    elif impact >= 70 and likelihood < 40:
        return "Transfer"
    elif impact < 40:
        return "Accept"
    else:
        return "Mitigate"

# create target column
df["Best_Decision"] = df.apply(
    lambda row: decide_treatment(row["Environmental_Impact_Score"], row["Risk_Assessment_Score"]),
    axis=1
)

# define features and target
features = [
    'Estimated_Cost_USD',
    'Resource_Allocation_Score',
    'Historical_Cost_Deviation_%',
    'Stakeholder_Priority_Score',
    'Time_Estimate_Days',
    'Scope_Complexity_Numeric'
]
target = 'Best_Decision'

# encode target labels
encoder = LabelEncoder()
df[target] = encoder.fit_transform(df[target])

# split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.3, random_state=42, stratify=df[target]
)

# scale numeric features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

# evaluate model
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (After Removing Leakage)")
plt.show()

# feature importance
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print("\nFeature Importances:")
for i in sorted_idx:
    print(f"{features[i]}: { importances[i]:.4f}")

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(9,5))
plt.barh(feat_imp['Feature'], feat_imp['Importance'])
plt.title("Feature Importance in Risk Treatment Decision")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# sensitivity analysis
sample = df.sample(1, random_state=10)
base_risk = sample["Risk_Assessment_Score"].values[0]
base_impact = sample["Environmental_Impact_Score"].values[0]

risk_range = np.arange(max(0, base_risk - 30), min(100, base_risk + 30), 5)
impact_range = np.arange(max(0, base_impact - 30), min(100, base_impact + 30), 5)

results = []
for r in risk_range:
    for i in impact_range:
        decision = decide_treatment(i, r)
        results.append((r, i, decision))

sensitivity_df = pd.DataFrame(results, columns=["Risk_Assessment", "Environmental_Impact", "Decision"])

pivot = sensitivity_df.pivot(index="Risk_Assessment", columns="Environmental_Impact", values="Decision")
pivot_numeric = pivot.apply(lambda col: pd.factorize(col)[0])

plt.figure(figsize=(10,6))
sns.heatmap(
    pivot_numeric,
    annot=pivot,        
    fmt='',
    cmap='tab10',
    cbar_kws={'label': 'Decision Code'}
)
plt.xlabel("Environmental Impact Score")
plt.ylabel("Risk Assessment Score")
plt.title("Sensitivity Analysis: Decision Changes with Risk and Impact")
plt.tight_layout()
plt.show()

# save dataset after update
df.to_csv("/content/Civil_Projects_With_Sensitivity.csv", index=False)
print("📁 File saved as: /content/Civil_Projects_With_Sensitivity.csv")

# print first 5 decisions
print(df["Best_Decision"].head(5))
print(encoder.inverse_transform(df["Best_Decision"].head(5)))

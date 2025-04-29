import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取數據
train_data = pd.read_csv('Ttrain.csv')
test_data = pd.read_csv('Ttest.csv')

# 查看數據基本資訊
print(train_data.head())
print(train_data.info())

# 將所有列顯示出來
pd.set_option('display.max_columns', None)

# 將所有行顯示出來（如果需要）
pd.set_option('display.max_rows', None)

# 控制數據框的寬度，避免顯示被截斷
pd.set_option('display.width', 1000)

# 重新查看數據
print(train_data.head())

# 查看空值數據
print(train_data.isnull().sum())

# 繪製生還與性別的分佈圖
sns.countplot(data=train_data, x='Survived', hue='Sex')
plt.title('Survival Count by Gender')
plt.show()

# 繪製票價分佈
sns.histplot(train_data['Fare'], kde=True)
plt.title('Fare Distribution')
plt.show()

# 填補年齡的空值 (用中位數)
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

# 填補票價的空值 (用中位數)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# 填補 Embarked (用最多的值)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# 轉換類別型變數 (性別和登船港口)
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# 刪除無關欄位 (如 'Name', 'Ticket', 'Cabin')
train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 定義特徵和標籤
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# 拆分訓練集和測試集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立隨機森林模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 驗證模型
y_pred = model.predict(X_val)
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred)}')
print('Confusion Matrix:', confusion_matrix(y_val, y_pred))

# 預測測試數據
predictions = model.predict(test_data)

# 生成提交檔案
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)

print("Prediction completed and saved as 'submission.csv'")

submission = pd.read_csv('submission.csv')
gender_submission = pd.read_csv('gender_submission.csv')

# 合併兩個dataframe，基於PassengerId
merged = pd.merge(submission, gender_submission, on='PassengerId', suffixes=('_model', '_gender'))

# 準確度
accuracy = (merged['Survived_model'] == merged['Survived_gender']).mean()

print(f"Model Accuracy: {accuracy * 100:.2f}%")
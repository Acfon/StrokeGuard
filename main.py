from pandas.core.interchange.dataframe_protocol import DataFrame

from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
STAT = []
df = pd.read_csv('dataset.csv')
df.drop('id', axis=1, inplace=True)
df.fillna(df['bmi'].mean(), inplace=True)
df["gender"] = df["gender"].replace({'Male':0,'Female':1,'Other':-1}).astype(np.int64)
x = df[['gender','age','hypertension','heart_disease','avg_glucose_level','bmi']]
dt_max = pd.Series(x.max(), index=x.columns)
dt_min = pd.Series(x.min(), index=x.columns)

y = df['stroke']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(x)
x = pd.DataFrame(scaled, columns=x.columns)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
x_train_new, y_train_new = smote.fit_resample(x_train, y_train)
xgb = XGBClassifier(
    learning_rate=0.01,
    max_depth=3,
    scale_pos_weight=30,
    random_state=42
)
xgb.fit(x_train, y_train)


class RiskOpred(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('main.ui', self)
        self.pushButton.clicked.connect(self.risk)

    def risk(self):
        global xgb
        gend = self.comboBox.currentIndex()
        age = self.spinBox.value()
        hipert = self.comboBox_2.currentIndex()
        dis = self.comboBox_3.currentIndex()
        glukose = self.doubleSpinBox.value()
        bmi = self.doubleSpinBox_2.value()
        dt = pd.DataFrame({"gender": [gend], "age": [age], "hypertension": [hipert], "heart_disease": [dis], "avg_glucose_level": [glukose], "bmi": [bmi] })
        dt.loc[len(dt)] = dt_max
        dt.loc[len(dt)] = dt_min
        dt = dt.rename(str, axis="columns")
        print(dt)
        scaler2 = MinMaxScaler()
        scaled2 = scaler2.fit_transform(dt)
        a = pd.Series(scaled2[0], index=x.columns)
        b = pd.DataFrame({"gender": [a[0]], "age": [a[1]], "hypertension": [a[2]], "heart_disease": [a[3]], "avg_glucose_level": [a[4]], "bmi": [a[5]]})
        xgb_pred2 = xgb.predict(b)
        if xgb_pred2[0] == 1:
            self.label_11.setText("ВНИМАНИЕ ВЫ В ЗОНЕ РИСКА, СОВЕТУЕМ ПРОВЕРИТЬСЯ У ВРАЧА")
        else:
            self.label_11.setText("Риска не обнаружено, но если вы сомневаетесь лучше обратитесь к врачу")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RiskOpred()
    ex.show()
    sys.exit(app.exec())
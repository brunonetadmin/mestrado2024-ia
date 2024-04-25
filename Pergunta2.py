import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("CreditScore.csv")

print(data["Credit_Score"].value_counts())

data["Credit_Mix"] = data["Credit_Mix"].map({"Standard": 1, "Good": 2, "Bad": 0})

x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Credit_Mix", "Outstanding_Debt", 
                   "Credit_History_Age", "Monthly_Balance"]])

y = np.array(data[["Credit_Score"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    test_size=0.33, 
                                                    random_state=42)

model = RandomForestClassifier()
model.fit(xtrain, np.ravel(ytrain))

print("Análise de Risco de Crédito : ")
a = float(input("Salário Anual: "))
b = float(input("Salário Mensal: "))
c = float(input("Número de Contas em Bancos: "))
d = float(input("Número de Cartões de Crédito: "))
e = float(input("Taxa de Interesse: "))
f = float(input("Quantidade de Empréstimos: "))
g = float(input("Maior número de dias de atraso da pessoa: "))
h = float(input("Número de Pagamentos Atrasados: "))
i = input("Credit Mix (Baixo: 0, Moderado: 1, Alto: 3) : ")
j = float(input("Débitos Pententes: "))
k = float(input("Histórico de Crédito em Anos: "))
l = float(input("Balanço Mensal: "))

features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])

print("Análise de Crédito Prevista = ", model.predict(features))

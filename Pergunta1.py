import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


print("=====================================================================================================")
print("## Lista de Exercios de IA - Exercicio 1 ")
print("## Bruno Cavalcante Barbosa e Mariah Tenório")
print("=====================================================================================================")

# Etapa 1 - Carregamento e Tratamento dos Dados

# Carregando os dados
maindata=pd.read_csv("diabetes_data.csv")
print("")
print("Carregando dados ...")
print("")

#maindata.info()

# Excluindo os zeros de todas as colunas, menos a do alvo
maindata_new = maindata[(maindata[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age',]] != 0).all(axis=1)]

#maindata_new.info()

novo_df = maindata_new.select_dtypes(include=[np.number])
novo_df = maindata_new.dropna(axis="columns", how="any")

X = novo_df.drop(columns=["Outcome"])
Y = novo_df["Outcome"]

num_obs = len(novo_df)
num_true = len(novo_df.loc[novo_df['Outcome'] == 1])
num_false = len(novo_df.loc[novo_df['Outcome'] == 0])

#print("Número de Casos Verdadeiros:  {0} ({1:2.2f}%)".format(num_true, (num_true/num_obs) * 100))
#print("Número de Casos Falsos: {0} ({1:2.2f}%)".format(num_false, (num_false/num_obs) * 100))

# Etapa 2 - Separação dos dados de Treinamento e Testes

feature_col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class_names = ['Outcome']

X = novo_df[feature_col_names].values # esses são os fatores para predição
y = novo_df[predicted_class_names].values # isso é o que queremso prever

# Separamos nossos dados em 30% para testes (0.3) e os 70% restante (0.7) para treinamento
split_test_size = 0.3

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = split_test_size,random_state=42)
print("")
print("# Conjunto de Dados:")
print("{0:0.2f}% no Conjunto de Treinamento".format((len(X_train)/len(novo_df.index)) * 100))
print("{0:0.2f}% no Conjunto de Testes".format((len(X_test)/len(novo_df.index)) * 100))
print("")
print("Verdadeiros (Original) : {0} ({1:0.2f}%)".format(len(novo_df.loc[novo_df['Outcome'] == 1]), (len(novo_df.loc[novo_df['Outcome'] == 1])/len(novo_df.index)) * 100.0))
print("Falsos (Original) : {0} ({1:0.2f}%)".format(len(novo_df.loc[novo_df['Outcome'] == 0]), (len(novo_df.loc[novo_df['Outcome'] == 0])/len(novo_df.index)) * 100.0))
print("")
print("Verdadeiros (Treinamento) : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
print("Falsos (Treinamento) : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
print("")
print("Verdadeiros (Testes)      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
print("Falsos (Testes)     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))

# Etapa 3 - Inicio do Treinamento

# Cria um objeto Gausiano usando o modelo Naive Bayes e treinando-o com os dados que temos
nb_model = GaussianNB()

nb_model.fit(X_train, y_train.ravel())

GaussianNB(priors=None)

# Prediz os valores usando os dados de treinamento
nb_predict_train = nb_model.predict(X_train)

# Etapa 4 - Resultados
print("")
print("# Treinamento ------------------------------------")
print("")
# Checa a acurácia do modelo naive bayes
print("Acurácia (Dados de Treinamento) : {0:.4f}".format(metrics.accuracy_score(y_train,nb_predict_train)))
print("")

nb_predict_test=nb_model.predict(X_test)

print("Acurácia (Dados de Testes) :{0:.4f}".format(metrics.accuracy_score(y_test,nb_predict_test)))
print("")

print("Matriz de Confusão")
print("{0}".format(metrics.confusion_matrix(y_test,nb_predict_test)))
print("")

print("========================================================================")
print("Relatório de Exclarecimentos")
print("{0}".format(metrics.classification_report(y_test,nb_predict_test)))

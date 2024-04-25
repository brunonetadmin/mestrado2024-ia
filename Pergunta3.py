import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define o range de cada variável
range_idade = np.arange(0, 101, 1)
range_sexo = np.arange(0, 2, 1)
range_pressao_alta = np.arange(0, 201, 1)
range_colesterol_alto = np.arange(0, 301, 1)
range_diabetes = np.arange(0, 2, 1)
range_dor_no_peito = np.arange(0, 11, 1)
range_falta_de_ar = np.arange(0, 11, 1)
range_tontura = np.arange(0, 11, 1)

# Cria os antecedentes
idade = ctrl.Antecedent(range_idade, 'idade')
sexo = ctrl.Antecedent(range_sexo, 'sexo')
pressao_alta = ctrl.Antecedent(range_pressao_alta, 'pressao_alta')
colesterol_alto = ctrl.Antecedent(range_colesterol_alto, 'colesterol_alto')
diabetes = ctrl.Antecedent(range_diabetes, 'diabetes')
dor_no_peito = ctrl.Antecedent(range_dor_no_peito, 'dor_no_peito')
falta_de_ar = ctrl.Antecedent(range_falta_de_ar, 'falta_de_ar')
tontura = ctrl.Antecedent(range_tontura, 'tontura')

# Define as funções membro
idade['jovem'] = fuzz.trimf(range_idade, [0, 25, 50])
idade['meia_idade'] = fuzz.trimf(range_idade, [25, 50, 75])
idade['idoso'] = fuzz.trimf(range_idade, [50, 75, 100])
sexo.automf(names=['feminino', 'masculino'])
pressao_alta.automf(names=['baixo', 'medio', 'alto'])
colesterol_alto.automf(names=['baixo', 'medio', 'alto'])
diabetes.automf(names=['nao', 'sim'])
dor_no_peito.automf(names=['leve', 'moderada', 'severa'])
falta_de_ar.automf(names=['leve', 'moderada', 'severa'])
tontura.automf(names=['leve', 'moderada', 'severa'])

# Define o consequente
avaliacao_cardiaca = ctrl.Consequent(np.arange(0, 101, 1), 'avaliacao cardiaca')

# Define as funções membro para o Consequente
avaliacao_cardiaca['baixo'] = fuzz.trimf(avaliacao_cardiaca.universe, [0, 25, 50])
avaliacao_cardiaca['medio'] = fuzz.trimf(avaliacao_cardiaca.universe, [25, 5440, 75])
avaliacao_cardiaca['alto'] = fuzz.trimf(avaliacao_cardiaca.universe, [50, 75, 100])

# Define as REgras
regra1 = ctrl.Rule(idade['jovem'] & sexo['masculino'] & pressao_alta['baixo'] & colesterol_alto['baixo'] & diabetes['nao'] & dor_no_peito['leve'] & falta_de_ar['leve'] & tontura['leve'], avaliacao_cardiaca['baixo'])
regra2 = ctrl.Rule(idade['meia_idade'] & sexo['feminino'] & pressao_alta['medio'] & colesterol_alto['medio'] & diabetes['sim'] & dor_no_peito['moderada'] & falta_de_ar['moderada'] & tontura['moderada'], avaliacao_cardiaca['medio'])
regra3 = ctrl.Rule(idade['idoso'] & sexo['masculino'] & pressao_alta['alto'] & colesterol_alto['alto'] & diabetes['sim'] & dor_no_peito['severa'] & falta_de_ar['severa'] & tontura['severa'], avaliacao_cardiaca['alto'])

# Cria o Sistema de Controle
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3])

# Roda a teste de avaliação
teste = ctrl.ControlSystemSimulation(sistema_controle)

# Função para Desfuzificar os conjuntos de Fuzzy
def desfuzificar(idade, sexo, pressao_alta, colesterol_alto, diabetes, dor_no_peito, falta_de_ar, tontura):
    teste.input['idade'] = idade
    teste.input['sexo'] = sexo
    teste.input['pressao_alta'] = pressao_alta
    teste.input['colesterol_alto'] = colesterol_alto
    teste.input['diabetes'] = diabetes
    teste.input['dor_no_peito'] = dor_no_peito
    teste.input['falta_de_ar'] = falta_de_ar
    teste.input['tontura'] = tontura

    # Computar os resultados
    teste.compute()

    return teste.output['avaliacao cardiaca']

# Pergunta ao usuário para preencher seus dados
idade_usuario = float(input("Por favor, insira sua idade: "))
sexo_usuario = float(input("Por favor, insira seu sexo (0 para feminino, 1 para masculino): "))
pressao_alta_usuario = float(input("Por favor, insira sua pressão arterial (em mmHg): "))
colesterol_alto_usuario = float(input("Por favor, insira seu nível de colesterol (em mg/dL): "))
diabetes_usuario = float(input("Por favor, insira se você tem diabetes (0 para não, 1 para sim): "))
dor_no_peito_usuario = float(input("Por favor, insira a intensidade da sua dor no peito (em uma escala de 0 a 10): "))
falta_de_ar_usuario = float(input("Por favor, insira a intensidade da sua falta de ar (em uma escala de 0 a 10): "))
tontura_usuario = float(input("Por favor, insira a intensidade da sua tontura (em uma escala de 0 a 10): "))

# Desfuzificar os conjuntos Fuzzy com os dados do usuário
resultado = desfuzificar(idade_usuario, sexo_usuario, pressao_alta_usuario, colesterol_alto_usuario, diabetes_usuario, dor_no_peito_usuario, falta_de_ar_usuario, tontura_usuario)

print("A avaliação cardíaca é: ", resultado)

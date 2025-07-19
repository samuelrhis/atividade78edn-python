import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# 1. Carregar o conjunto de dados
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Mapeamento para clareza: 0 = Maligno, 1 = Benigno
y = y.map({0: 'Maligno', 1: 'Benigno'})

# 2. Divisão dos dados em treino e teste
# 70% para treino e 30% para teste. random_state garante a reprodutibilidade.
# stratify=y garante que a proporção de classes seja a mesma nos conjuntos de treino e teste.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. Criação e Treinamento do Modelo
# Usando RandomForestClassifier com random_state para reprodutibilidade do modelo.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Realização das Previsões
y_pred = model.predict(X_test)
# Previsão de probabilidades para a classe positiva ('Benigno'), necessário para a curva ROC
y_pred_proba = model.predict_proba(X_test)[:, 1]


# 5. Cálculo das Métricas
# Nota: 'pos_label' define qual classe é considerada a "positiva".
# Foca na detecção do caso "Maligno" como o evento de interesse principal.
# Para isso, scikit-learn trata a primeira classe alfabeticamente ('Benigno') como 0 e a segunda ('Maligno') como 1 internamente
# ou especificar `pos_label='Maligno'` para clareza.

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Maligno')
recall = recall_score(y_test, y_pred, pos_label='Maligno')
f1 = f1_score(y_test, y_pred, pos_label='Maligno')

# Para AUC-ROC, precisa dos rótulos numéricos (0 e 1)
y_test_numeric = y_test.map({'Maligno': 0, 'Benigno': 1})
# A probabilidade calculada foi para a classe 'Benigno' (classe 1), então 1 - prob para a classe 'Maligno'
auc_roc = roc_auc_score(y_test_numeric, 1 - y_pred_proba)


# 6. Apresentação dos Resultados
print("--- Métricas de Desempenho do Modelo ---")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão (para 'Maligno'): {precision:.4f}")
print(f"Recall (Sensibilidade) (para 'Maligno'): {recall:.4f}")
print(f"F1-Score (para 'Maligno'): {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}\n")

# 7. Matriz de Confusão
print("--- Matriz de Confusão ---")
cm = confusion_matrix(y_test, y_pred, labels=['Maligno', 'Benigno'])
# Plotando a matriz de confusão para melhor visualização
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Maligno', 'Benigno'],
            yticklabels=['Maligno', 'Benigno'])
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

# 8. Curva ROC
fpr, tpr, thresholds = roc_curve(y_test_numeric, 1 - y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade/Recall)')
plt.title('Curva ROC para Detecção de Tumor Maligno')
plt.legend(loc="lower right")
plt.show()
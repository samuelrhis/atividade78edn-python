📊 Análise de Métricas de Classificação para Diagnóstico Médico
Este repositório contém um script em Python para treinar um modelo de Machine Learning (Random Forest) e realizar uma análise aprofundada das principais métricas de classificação, utilizando o dataset breast_cancer como estudo de caso.

🎯 Sobre o Projeto
O objetivo principal deste projeto não é apenas construir um classificador, mas demonstrar a importância de escolher e interpretar as métricas de avaliação corretas, especialmente em contextos de alto risco, como o diagnóstico médico.

Em um cenário de diagnóstico de câncer, um erro de classificação não é apenas um número. A falha em detectar uma doença existente (Falso Negativo) tem consequências muito mais graves do que um alarme falso (Falso Positivo). Este script ajuda a visualizar e quantificar essa diferença.

🛠️ Tecnologias Utilizadas
Python 3

Scikit-learn: Para o modelo, métricas e o dataset.

Pandas: Para manipulação de dados.

Matplotlib & Seaborn: Para a visualização dos resultados (Matriz de Confusão e Curva ROC).

🚀 Como Executar
Siga os passos abaixo para executar a análise em sua máquina local.

Clone o repositório:

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

Crie um ambiente virtual (Recomendado):

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

Instale as dependências:
Crie um arquivo requirements.txt com o conteúdo abaixo e execute o comando pip.

# requirements.txt
numpy
scikit-learn
pandas
matplotlib
seaborn
```bash
pip install -r requirements.txt

Execute o script:

python analise_metricas_classificacao.py

📄 Resultados Esperados
Ao executar o script, você verá no terminal a saída com os valores calculados para cada métrica. Além disso, duas janelas de gráfico serão exibidas:

Matriz de Confusão: Visualiza os acertos e erros do modelo, separando os tipos de erro (Falsos Positivos e Falsos Negativos).
[Imagem de uma matriz de confusão para diagnóstico de cancro]

Curva ROC: Mostra a performance do classificador e sua capacidade de discriminação entre as classes.
[Imagem de uma curva ROC para um modelo de classificação]

🤝 Contribuição
Contribuições para melhorar este e outros laboratórios são muito bem-vindas!

Para este Projeto
Se encontrar bugs ou tiver sugestões, sinta-se à vontade para abrir uma issue ou enviar um pull request seguindo o fluxo padrão do GitHub.

📜 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

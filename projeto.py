# %% [markdown]
# Dataset: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn
# 
# ___
# ### **Integrantes:**
# - Débora 
# - Fernanda Vaz 
# - Gabriel Cardoso
# - Mayara Chew
# 
# ___
# ### **Enunciado**
# 
# Neste mini-projeto, o objetivo é conectar esses conceitos a um problema real do ambiente bancário, construindo uma prova de conceito (PoC) baseada em dados.
# 
# O foco não é apenas o modelo, mas a capacidade de identificar um gap real e propor uma solução viável com ML.
# 
# 
# **Desafio:**
# 
# Cada grupo deverá:
# 
# - Identificar um problema relevante do contexto bancário que possa ser endereçado com Machine Learning
# - explicitar o gap existente
# - apresentar uma solução baseada em dados.
# 
# ___
# ### **1. O problema e o gap**
# 
# **Qual é o problema?**
# 
# Perda de clientes (churn), impactando em:
# - Alto custo na aquisição de novos clientes
# - Venda de produtos bancários (cross-sell)
# - Lucro
# 
# ___
# **Qual decisão hoje é mal feita, lenta ou inexistente?**
# - Ações de retenção são reativas, e não preventivas
# - Campanhas são enviadas para todos os clientes, falta priorização de clientes com maior risco de saída (churn)
# - Isso gera desperdício de recursos, e baixa eficiência em retenção
# 
# ___
# **Onde está o gap que justifica o uso de ML?**
# - Previsão antecipada, agir preventivamente e com assertividade
# - Relações nem sempre são lineares
# - Interações entre variáveis são difíceis de identificar manualmente
# - Não ficar refém da subjetividade da estratégia de cada agência. Montar uma estratégia unificada e assertiva.
# 
# ___
# ### **2. Formulação do problema em ML**
# 
# **Classificação ou regressão?**
# 
# O problema apresentado é de classificação, pois precisamos classificar o cliente dada à sua probabilidade de dar churn (0 ou 1)
# 
# ___
# **Qual é a variável alvo?**
# 
# A variável alvo é a ``Exited``
# 
# ___
# **O que o modelo está tentando prever?**
# 
# Está tentando prever se o cliente irá sair do banco. (dar churn)
# 
# ___
# 
# ### **3. Análise dos dados**
# 
# **Breve análise exploratória**
# 
# ___
# **Correlação (uni e multivariada)**
# 
# ___
# **Missings e possíveis vieses**
# 
# ___
# ### **4. Preparação dos dados**
# 
# **Normalização ou padronização**
# 
# ___
# **Tratamento de desbalanceamento (se aplicável)**
# 
# ___
# **Justificativa das escolhas**
# 
# ___
# ### **5. Modelagem**
# 
# 
# **Pelo menos 2 modelos entre:**
# - Regressão logística
# - Árvore de decisão
# - Random Forest
# 
# É permitido (mas não obrigatório) utilizar outras técnicas não vistas em aula, desde que:
# - Sejam explicadas
# - Façam sentido para o problema
# 
# ___
# ### **6. Avaliação**
# 
# - Métrica adequada ao problema (accuracy, recall, precision, AUC, etc.)
# - Comparação entre modelos
# - Trade-offs relevantes (ex.: falso positivo vs falso negativo)
# 
# ___
# ### **7. Conclusão executiva**
# 
# **O modelo resolve o gap identificado?**
# 
# ___
# **Onde ele falha?**
# 
# ___
# **O que seria necessário para levar isso para produção?**
# 
# ___
# **Há potencial de geração de valor (ex.: redução de custo, aumento de eficiência, mitigação de risco)?**
# 
# **Caso positivo, explore os resultados sob uma ótica qualitativa ou estimada, considerando impacto financeiro, ROI ou payback.**
# 
# ___
# ### 8. Riscos e limitações
# 
# **Riscos técnicos**
# 
# ___
# **Riscos de viés**
# 
# ___
# **Riscos regulatórios ou operacionais**
# 
# ___
# ### **9. Apresentação do PoC**
# 
# **Demonstração da solução concebida em funcionamento, preferencialmente como:**
# 
# - Uma aplicação completa (ex.: Streamlit), e/ou
# 
# - Um notebook ou programa em Python executável via linha de comando.

# %% [markdown]
# ___
# 
# **Visão geral do Dataset**

# %%
import pandas as pd
pd.set_option('display.max_columns', None) # Exibe todas as colunas do dataframe
pd.set_option('display.max_rows', None)    # Exibe todas as linhas do dataframe
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import math
import numpy as np
import pingouin as pg
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# %%
try:
    df_raw = pd.read_csv("Customer-Churn-Records.csv")
except:
    path = kagglehub.dataset_download("radheshyamkollipara/bank-customer-churn")
    df_raw = pd.read_csv(f"{path}/Customer-Churn-Records.csv")

print(df_raw.shape)
df_raw.head()

# %%
df_raw.describe()

# %% [markdown]
# ___
# ### **3 - Análise dos Dados**
# 
# **Análises:**
# - Matriz de correlação
# - Boxplot
# - Odds Ratio
# 
# **Tratamento de variáveis:**
# - Dar atenção às variáveis demográficas (gênero, idade, país de origem) -> Podem implicar discriminação
# - Tratamento dos nulos
# - Dummyzação das variáveis categóricas (Card Type)
# - Relação das variáveis ``IsActiveMember`` e ``Complain`` com a ``Exited``. Seria uma forma de vazamento de dados?

# %% [markdown]
# ___
# #### **3.1 - Breve Análise Exploratória**

# %% [markdown]
# **Eliminando colunas irrelevantes**

# %%
df = df_raw.copy()

print(df.shape)
display(df.head())

# Eliminando colunas irrelevantes
cols_drop = ['RowNumber', 'CustomerId', 'Surname']
df = df.drop(columns=cols_drop)

# %% [markdown]
# **Verificando distribuição de categorias**
# 
# Insights:
# - Desbalanceamento no churn (80/20)
# - Card Type e Geography parecidos (dados sintéticos?)

# %%
cols_cat = ['Exited', 'IsActiveMember', 'Complain', 'Card Type', 'Geography', 'Gender']
linhas = math.ceil(len(cols_cat) / 3)

plt.figure(figsize=(15, 4 * linhas))

for i, col in enumerate(cols_cat):
    plt.subplot(linhas, 3, i + 1)
    ax = sns.countplot(data=df, x=col, hue=col, palette='Set2', legend=False)
    plt.title(f"Distribuição de {col}") # Adiciona o título
    
    # Adiciona a porcentagem acima de cada barra
    for c in ax.containers:
        labels = [f'{val / len(df):.1%}' for val in c.datavalues]
        ax.bar_label(c, labels=labels)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Distribuição das categorias por Churn**
# - Mulheres dão mais churn proporcionalmente do que homens
# - Alemães dão mais churn proporcionalmente
# - Clientes DIAMOND dão um pouco mais de churn
# 
# **OBS: SOMA DAS BARRAS = 100%**

# %%
cols_cat = ['IsActiveMember', 'Complain', 'Card Type', 'Geography', 'Gender']
linhas = math.ceil(len(cols_cat) / 3)

plt.figure(figsize=(15, 4 * linhas))

for i, col in enumerate(cols_cat):
    ax = plt.subplot(linhas, 3, i + 1)
    sns.countplot(data=df, x=col, hue='Exited', palette='Set1', ax=ax)
    plt.title(f"Churn por {col}")
    
    # Adiciona a porcentagem
    for c in ax.containers:
        labels = [f'{val / len(df):.1%}' for val in c.datavalues]
        ax.bar_label(c, labels=labels)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Pairplot**
# 
# Insights:
# - Fazer análise direcionada a clientes de alto valor usando o balance e o quartil 0.75.
# 

# %%
g = sns.pairplot(
    df[["Age", "Balance", "CreditScore", "NumOfProducts", "Exited"]],
    hue="Exited"
)

# Itera sobre as variáveis de Y (linhas) e X (colunas) para nomear cada plot
for i, y_var in enumerate(g.y_vars):
    for j, x_var in enumerate(g.x_vars):
        g.axes[i, j].set_title(f"{y_var} vs {x_var}", fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Kernel Density Estimation (KDE)**
# 
# Insights:
# - Diferenças em ``Age``, ``Balance`` e ``NumOfProducts``

# %%
# Filtra colunas numéricas que tenham mais de 2 valores (não binárias) e que não sejam o target
num_cols = [c for c in df.select_dtypes(include='number').columns if df[c].nunique() > 2 and c != 'Exited']
linhas = math.ceil(len(num_cols) / 4)

plt.figure(figsize=(16, 4 * linhas))

for i, col in enumerate(num_cols):
    plt.subplot(linhas, 4, i + 1)
    # common_norm=False garante que as curvas sejam comparáveis mesmo se os grupos tiverem tamanhos diferentes
    sns.kdeplot(data=df, x=col, hue='Exited', fill=True, common_norm=False, palette='Set1')
    plt.title(f"KDE de {col}")

plt.tight_layout()
plt.show()

# %% [markdown]
# **Boxplot de variáveis para Churn/Não Churn**
# 
# Insights:
# - Diferença notável nas distribuições de ``Age`` e ``Balance``

# %%
import math

col_target = ['Exited']
boxplot_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned']

n_cols = 4
n_rows = math.ceil(len(boxplot_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(boxplot_cols):
    sns.boxplot(x='Exited', y=col, data=df, ax=axes[i])
    axes[i].set_title(f"Boxplot de {col}")

# Oculta eixos extras caso o número de colunas não seja múltiplo de 4
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Insights Finais:**
# - Verificar se os valores ``Gender=Female``, ``Card_Type=Platinum``, ``Age``, ``Balance``, ``NumOfProducts`` e ``Geography=Germany`` podem contribuir à previsão de Churn

# %% [markdown]
# ___
# #### **3.2 - Correlação uni e multivariada**

# %% [markdown]
# **Dummyzação das variáveis**

# %%
# Dummyzar variáveis categóricas para viabilizar análise de correlação
cols_cat = ['Card Type', 'Geography', 'Gender']
df_dummies = pd.get_dummies(df, columns=cols_cat, drop_first=True, dtype=int)
df_dummies.head()

# %% [markdown]
# **Matriz de Correlação**
# 
# Insights:
# - Grande maioria possui correlação nula
# - Investigar relação de ``age`` e ``balance`` com ``Exited``

# %%
corr = df_dummies.corr()

# 2. Configurar o tamanho do gráfico (Largura x Altura)
# 20 para a largura e 12 para a altura costuma ser ideal para ~20 variáveis
plt.figure(figsize=(20, 12))

# 3. Criar uma máscara para esconder a metade superior (Opcional, mas ajuda muito na leitura)
# Como a correlação de A com B é a mesma de B com A, o gráfico fica mais limpo.
mask = np.triu(np.ones_like(corr, dtype=bool))

# 4. Plotar o Heatmap
sns.heatmap(
    corr, 
    mask=mask,              # Aplica a máscara (opcional, comente se preferir o quadrado cheio)
    annot=True,             # Mostra os valores de correlação
    fmt=".2f",              # Arredonda para 2 casas decimais
    cmap='coolwarm',        # Escala de Azul (negativo) para Vermelho (positivo)
    center=0,               # Garante que o branco seja o ponto neutro (zero)
    linewidths=.5,          # Adiciona uma linha fina entre os quadrados
    cbar_kws={"shrink": .8} # Ajusta o tamanho da barra de legenda lateral
)

# 5. Ajustes estéticos finais
plt.title('Matriz de Correlação Multivariada', fontsize=20)
plt.xticks(rotation=45, ha='right') # Rotaciona os nomes das colunas para não embolar
plt.show()

# %% [markdown]
# **Correlação - Pearson e Spearman**

# %%
pearson_corr = df_dummies.corr(method='pearson', numeric_only=True)
spearman_corr = df_dummies.corr(method='spearman', numeric_only=True)

corr = pd.DataFrame({
    "Pearson": pearson_corr["Exited"],
    "Spearman": spearman_corr["Exited"]
}).sort_values(by="Pearson", ascending=False)

corr

# %% [markdown]
# **Correlação Parcial**

# %%
target = "Exited"

# Pega todas as colunas do df_dummies, exceto o target
cols_to_use = [c for c in df_dummies.columns if c != target]

partial_corr_results = []

for col in cols_to_use:
    # As covariáveis são todas as outras colunas numéricas, exceto a atual
    covar_cols = [c for c in cols_to_use if c != col]
    
    # Calcula Pearson e Spearman usando df_dummies
    pc_pearson = pg.partial_corr(data=df_dummies, x=col, y=target, covar=covar_cols, method="pearson")
    pc_spearman = pg.partial_corr(data=df_dummies, x=col, y=target, covar=covar_cols, method="spearman")
    
    # Salva os resultados
    partial_corr_results.append((
        col, 
        pc_pearson["r"].values[0], 
        pc_spearman["r"].values[0]
    ))

# Cria o DataFrame final e ordena pela coluna Pearson de forma descendente
partial_corr_df = pd.DataFrame(partial_corr_results, columns=["Feature", "Pearson", "Spearman"])
partial_corr_df = partial_corr_df.sort_values(by="Pearson", ascending=False).reset_index(drop=True)

partial_corr_df

# %% [markdown]
# ___
# #### **3.3 - Missings e possíveis vieses**
# 
# - Não existem valores faltantes
# - Correlação quase perfeita entre ``Complain`` e ``Exited``
# - Dúvida no ``IsActiveMember``: um cliente é ativo e deu churn ao mesmo tempo?

# %% [markdown]
# Nulos

# %%
display(df_dummies.isnull().sum())

# %% [markdown]
# Qual o sentido de um cliente ser ativo e churn ao mesmo tempo?

# %%
df_dummies[(df_dummies['IsActiveMember'] == 1) & (df_dummies['Exited'] == 1)].head(5)

# %% [markdown]
# Quase todos os complains são churn.

# %%
pd.crosstab(df['Exited'], df['Complain'])


# %%
cols_remove = ['Complain', 'IsActiveMember']
df_model = df_dummies.drop(cols_remove, axis=1)

# %% [markdown]
# **Decisão final:**
# - Remover as variáveis ``Complain`` e ``IsActiveMember`` do treinamento do modelo 
# 

# %% [markdown]
# ___
# ### **4. Preparação dos dados**

# %% [markdown]
# ___
# ### **4.1 - Normalização ou padronização**
# 
# - Aplicada após holdout para evitar vazamento de dados (normalização)

# %%


# %% [markdown]
# ___
# ### **4.2 - Tratamento de desbalanceamento (se aplicável)**
# 
# - Aplicada após holdout para evitar vazamento de dados (SMOTE)

# %%


# %% [markdown]
# ___
# ### **4.3 - Justificativa das escolhas**
# - **StandardScaler:** Colocar todas as features na mesma escala, conseguir comparar ``Age``, ``Balance`` e ``Satisfaction Score`` de maneira direta. Facilita para o modelo entender a importância de cada variável.
# 
# - **SMOTE:** O undersampling faria a gente "jogar fora" grande parte dos dados. E como a base é relativamente pequena (10K), a base ficaria ainda menor.
# 
# - **Aplicação após holdout:** Caso fosse aplicado antes, a escala seria distorcida com base nos dados de teste. Assim como os dados sintéticos do SMOTE também teriam influencia dos dados de teste. A aplicação após o holdout é para evitar vazamento de dados.
# 

# %% [markdown]
# ___
# ### **5. Modelagem**

# %% [markdown]
# **Regressão Logística e Odds Ratio**

# %%
import pandas as pd
import numpy as np


df_relog = df_dummies.copy()

X = df_relog.drop(columns=['Exited'])
y = df_relog['Exited']

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treinamento
modelo_log = LogisticRegression()
modelo_log.fit(X_scaled, y)

# Extraindo métricas
coeficientes = modelo_log.coef_[0]
odds_ratios = np.exp(coeficientes)

# Montando o DataFrame
tabela_resultados = pd.DataFrame({
    'Variável': X.columns,
    'Coeficiente': coeficientes,
    'Odds Ratio': odds_ratios
})

# Adicionando o Coeficiente Absoluto usando a função abs()
tabela_resultados['Coeficiente Absoluto'] = tabela_resultados['Coeficiente'].abs()
tabela_resultados = tabela_resultados[['Variável', 'Odds Ratio', 'Coeficiente', 'Coeficiente Absoluto']]
tabela_resultados = tabela_resultados.sort_values(by='Odds Ratio', ascending=False).reset_index(drop=True)

print(tabela_resultados)

# %%
# Remover colunas com coeficiente muito baixo:
cols_remover = ['EstimatedSalary', 'Satisfaction Score', 'Point Earned', 'Geography_Spain', 'HasCrCard', 'Point Earned', 'Tenure', 'Card Type_PLATINUM']
df_model_1 = df_model.drop(columns=cols_remover)
df_model_1.head()

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import loguniform, randint
from imblearn.pipeline import Pipeline

def encontrar_melhor_modelo(X_train, y_train, X_test, y_test):
    """
    Roda um RandomizedSearchCV para 4 modelos usando StandardScaler e SMOTE,
    avalia todos no conjunto de teste para comparação, plota as matrizes de confusão
    e retorna o melhor modelo treinado.
    """
    print("Iniciando a busca pelo melhor modelo...\n")

    # 1. Definindo os modelos base
    base_tree = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=42)
    
    # 2. Dicionário com modelos e seus parâmetros
    modelos_params = {
        "Regressão Logística": (
            LogisticRegression(class_weight="balanced", max_iter=4000, random_state=42),
            {
                "model__C": loguniform(1e-3, 1e3),
                "model__solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                "model__penalty": ["l2"]
            }
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {
                "model__n_estimators": randint(10, 500),
                "model__max_depth": randint(1, 100),
                "model__min_samples_split": randint(2, 60),
                "model__min_samples_leaf": randint(1, 30),
                "model__max_features": ["sqrt", "log2", None],
                "model__bootstrap": [True, False]
            }
        ),
        "AdaBoost": (
            AdaBoostClassifier(estimator=base_tree, random_state=42),
            {
                "model__n_estimators": [100, 200, 400, 600],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.3, 0.5]
            }
        ),
        "SVM": (
            SVC(class_weight="balanced", random_state=42),
            {
                "model__kernel": ["linear", "poly", "rbf"],
                "model__C": loguniform(1e-3, 1e3),
                "model__gamma": ["scale", "auto"]
            }
        )
    }

    melhor_f1_global = 0
    melhor_modelo_global = None
    nome_melhor_modelo = ""
    
    # Dicionário para armazenar as matrizes de confusão de cada modelo
    resultados_teste = {}

    # 3. Loop para testar cada modelo
    for nome, (modelo, params) in modelos_params.items():
        print(f"--- Treinando {nome} ---")

        # Construindo a esteira de pré-processamento (Scaler -> SMOTE -> Modelo)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", modelo)
        ])

        # Configurando a busca aleatória
        random_search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=params,
            n_iter=10,        # Quantidade de combinações testadas por modelo
            scoring="f1",     # Focando no F1-Score devido ao desbalanceamento
            cv=3,             # Divisões da validação cruzada (pode aumentar para 5)
            random_state=42,
            n_jobs=-1,
            refit=True        # Já devolve o modelo retreinado com os melhores parâmetros
        )

        # Rodando o treinamento
        random_search.fit(X_train, y_train)
        f1_atual = random_search.best_score_
        
        print(f"Melhor F1 na Validação Cruzada ({nome}): {f1_atual:.4f}")
        print(f"Melhores parâmetros: {random_search.best_params_}\n")

        # Avaliando o modelo atual nos dados de teste para salvar a Matriz de Confusão
        modelo_treinado = random_search.best_estimator_
        y_pred = modelo_treinado.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Salvando os resultados para plotar e printar depois
        resultados_teste[nome] = cm

        # Atualizando o campeão global (Baseado no F1 da Validação Cruzada)
        if f1_atual > melhor_f1_global:
            melhor_f1_global = f1_atual
            melhor_modelo_global = modelo_treinado
            nome_melhor_modelo = nome

    # 4. Plotando as Matrizes de Confusão Lado a Lado (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comparação das Matrizes de Confusão (Dados de Teste)', fontsize=16)
    axes = axes.flatten()

    for idx, (nome, cm) in enumerate(resultados_teste.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0 (Ficou)", "1 (Churn)"])
        disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
        axes[idx].set_title(f"Modelo: {nome}")

    plt.tight_layout()
    plt.show()

    # 5. Printando os valores da Matriz de Confusão de forma seguida
    print("="*50)
    print(" COMPARAÇÃO DETALHADA DAS MATRIZES DE CONFUSÃO")
    print("="*50)
    for nome, cm in resultados_teste.items():
        print(f"\n--- {nome} ---")
        print(f"Verdadeiros Negativos (Classe 0 - Ficou e Previu que Ficaria): {cm[0][0]}")
        print(f"Falsos Positivos      (Classe 0 - Ficou, mas Previu Churn)   : {cm[0][1]}")
        print(f"Falsos Negativos      (Classe 1 - Deu Churn, previu Ficou)   : {cm[1][0]}")
        print(f"Verdadeiros Positivos (Classe 1 - Deu Churn e Previu Churn)  : {cm[1][1]}")

    # 6. Resultado final do Vencedor
    print("\n" + "="*50)
    print(f"🏆 O GRANDE CAMPEÃO FOI: {nome_melhor_modelo}")
    print(f"F1-Score na Validação Cruzada: {melhor_f1_global:.4f}")
    print("="*50)

    return melhor_modelo_global



X = df_model.drop('Exited', axis=1)
y = df_model['Exited']

# 1. Primeiro separamos os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Chamamos a função passando treino e teste (para ela conseguir plotar/comparar)
modelo_vencedor = encontrar_melhor_modelo(X_train, y_train, X_test, y_test)

# 3. Avaliação final detalhada apenas do modelo campeão
y_pred_campeao = modelo_vencedor.predict(X_test)

print("\n" + "="*50)
print(f" AVALIAÇÃO FINAL DO MODELO CAMPEÃO NOS DADOS DE TESTE")
print("="*50)

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred_campeao))

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Extraindo o modelo real de dentro do Pipeline
# Lembra que demos o nome de "model" para a etapa do algoritmo no Pipeline?
rf_model = modelo_vencedor.named_steps["model"]

# 2. Pegando os valores de importância calculados pelo Random Forest
importancias = rf_model.feature_importances_

# 3. Pegando os nomes das colunas originais do seu DataFrame de treino
# (Substitua X_train pelo nome da sua variável com as features, se for diferente)
nomes_features = X_train.columns

# 4. Criando um DataFrame para facilitar a ordenação
df_importancias = pd.DataFrame({
    'Feature': nomes_features,
    'Importância': importancias
})

# 5. Ordenando da mais importante para a menos importante
df_importancias = df_importancias.sort_values(by='Importância', ascending=False)

# 6. Plotando o Gráfico de Barras Horizontal
plt.figure(figsize=(10, 8)) # Ajuste o tamanho conforme necessário

# O Seaborn facilita muito a criação de gráficos de barras bonitos
sns.barplot(
    x='Importância', 
    y='Feature', 
    data=df_importancias, 
    palette='viridis' # Uma paleta de cores agradável e profissional
)

# Adicionando títulos e rótulos
plt.title('Importância das Variáveis (Feature Importance) - Random Forest', fontsize=14)
plt.xlabel('Importância Relativa (Quanto maior, mais impacto)', fontsize=12)
plt.ylabel('Variáveis do Cliente', fontsize=12)

# Ajusta o layout para não cortar os nomes das variáveis
plt.tight_layout()

# Exibe o gráfico no seu Jupyter Notebook ou interface
plt.show()

# %% [markdown]
# ___
# ### **6. Avaliação**
# 
# - Métrica adequada ao problema (accuracy, recall, precision, AUC, etc.)
# - Comparação entre modelos
# - Trade-offs relevantes (ex.: falso positivo vs falso negativo)

# %%


# %% [markdown]
# **Persistência dos Modelos**

# %%
import joblib

# ==========================================
# ARTEFATO 1: SALVANDO O MODELO (.pkl)
# ==========================================
nome_arquivo_modelo = 'rf.pkl'
joblib.dump(modelo_vencedor, nome_arquivo_modelo)
print(f"✅ Modelo salvo com sucesso: {nome_arquivo_modelo}")


# ==========================================
# ARTEFATO 2: GERANDO O DATASET FINAL (.csv)
# ==========================================
# Supondo que 'df_model' é o seu dataframe completo lá do início
# e 'X' são todas as suas features antes da divisão de treino/teste.

# 1. Fazemos uma cópia do dataframe original para não bagunçar os dados iniciais
df_final = df_model.copy()

# 2. Criamos a coluna com a previsão (0 ou 1)
df_final['CHURN_PREDICT'] = modelo_vencedor.predict(X)

# 3. Criamos a coluna com a probabilidade
# predict_proba retorna duas colunas: [prob_classe_0, prob_classe_1]
# Pegamos apenas a coluna índice 1 ([:, 1]), que é a probabilidade de dar Churn
df_final['CHURN_PROB'] = modelo_vencedor.predict_proba(X)[:, 1]

# (Opcional) Podemos arredondar a probabilidade para 4 casas decimais para ficar mais limpo
df_final['CHURN_PROB'] = df_final['CHURN_PROB'].round(4)

# 4. Salvamos o resultado em um arquivo CSV
nome_arquivo_csv = 'customer-churn-predict.csv'
df_final.to_csv(nome_arquivo_csv, index=False)
print(f"✅ Dataset final salvo com sucesso: {nome_arquivo_csv}")

# %% [markdown]
# ___
# ### **7. Conclusão executiva**
# 
# **O modelo resolve o gap identificado?**
# 
# ___
# **Onde ele falha?**
# 
# ___
# **O que seria necessário para levar isso para produção?**
# 
# ___
# **Há potencial de geração de valor (ex.: redução de custo, aumento de eficiência, mitigação de risco)?**
# 
# **Caso positivo, explore os resultados sob uma ótica qualitativa ou estimada, considerando impacto financeiro, ROI ou payback.**

# %%


# %% [markdown]
# ___
# ### 8. Riscos e limitações
# 
# **Riscos técnicos**
# 
# ___
# **Riscos de viés**
# 
# ___
# **Riscos regulatórios ou operacionais**

# %%


# %% [markdown]
# ___
# ### **9. Apresentação do PoC**
# 
# **Demonstração da solução concebida em funcionamento, preferencialmente como:**
# 
# - Uma aplicação completa (ex.: Streamlit), e/ou
# 
# - Um notebook ou programa em Python executável via linha de comando.



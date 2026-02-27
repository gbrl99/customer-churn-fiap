import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------
# CONFIGURAÇÃO INICIAL E IDENTIDADE VISUAL
# ---------------------------------------------------------
st.set_page_config(page_title="Projeto IA - Previsão de Churn", layout="wide")

# CSS customizado para identidade visual (Inspirado na Caixa Econômica Federal)
st.markdown("""
    <style>
    /* Paleta de Cores: Azul Escuro (#005CA9), Laranja (#F39200), Azul Claro (#00A3E0) */
    
    .main-header {
        background-color: #005CA9;
        color: #FFFFFF;
        padding: 20px;
        border-bottom: 5px solid #F39200;
        text-align: center;
        border-radius: 8px;
        margin-bottom: 30px;
    }
    
    .section-title {
        color: #F39200;
        border-bottom: 2px solid #00A3E0;
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 20px;
        font-family: 'Arial', sans-serif;
    }
    
    .card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00A3E0;
        margin-bottom: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    
    .highlight-text {
        color: #005CA9;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# CABEÇALHO DO APLICATIVO
st.markdown('<div class="main-header"><h1>📊 Previsão de Churn Bancário</h1></div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# SEÇÃO 1: CENÁRIO DO PROBLEMA
# ---------------------------------------------------------
st.markdown('<h2 class="section-title">Cenário do Problema</h2>', unsafe_allow_html=True)

st.markdown("""
A evasão de clientes, conhecida como churn, representa um desafio crítico para a instituição bancária. A perda de um consumidor acarreta impactos diretos, como a diminuição imediata do lucro e a perda de valiosas oportunidades de vendas cruzadas (cross-sell) de novos produtos. Além disso, como o custo para adquirir novos clientes no setor financeiro é historicamente elevado, perder um cliente cujo custo de aquisição já foi pago gera um desperdício financeiro significativo para a operação.
""")

# Divisão em duas colunas para melhorar o design visual
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📉 Qual é o problema?")
    st.markdown("""
    A **perda de clientes (churn)** é um desafio constante que impacta diretamente os resultados da instituição em diversas frentes:
    * Elevado custo na aquisição de novos clientes para repor a base.
    * Queda nas oportunidades de venda de produtos bancários (cross-sell).
    * Redução geral na rentabilidade e no lucro.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🛑 Onde está o gap atual?")
    st.markdown("""
    O processo atual de retenção sofre com lentidão e ineficiência devido a fatores como:
    * **Ações Reativas:** Tentativas de retenção ocorrem apenas quando o cliente já decidiu sair.
    * **Falta de Priorização:** Campanhas massivas disparam para toda a base, gerando grande desperdício de recursos.
    * **Subjetividade:** Falta de uma estratégia unificada, deixando as decisões reféns do julgamento individual de cada agência.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 💡 A Solução Proposta")
st.markdown("""
Desenvolvemos uma solução preditiva de Classificação para antecipar a probabilidade de um cliente evadir (churn), permitindo que a instituição aja de maneira preventiva e assertiva.

O que o modelo tenta prever? Se o cliente irá sair do banco (1) ou permanecer (0).

Variável Alvo: A coluna ``Exited``.

Aprofundamento Técnico: Para lidar com a natureza do negócio, onde os dados apresentam uma proporção de churn de 80/20 (desbalanceamento histórico), optamos por utilizar a técnica de class-weight (pesos de classe) diretamente na etapa de modelagem, associada ao Standard Scaler na preparação para garantir a padronização das features. Essa estrutura assegura que o algoritmo penalize com maior rigor os erros na predição da classe minoritária, aprendendo os padrões reais de evasão sem ser enviesado pela classe majoritária.""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# DETALHES DO DATASET (Utilizando um expander para não poluir a tela inicial)
with st.expander("🔍 Explorar as Variáveis do Dataset", expanded=False):
    st.markdown("""
    Os dados baseiam-se em um histórico de clientes (10.000 registros e 18 colunas), carregados através do repositório `artefatos/customer-churn-predict.csv`. 
    
    Abaixo estão as características mapeadas para compreender o comportamento do consumidor:
    
    * **CreditScore:** Pontuação de crédito. Clientes com maior pontuação tendem a permanecer no banco.
    * **Geography:** Localização geográfica do cliente.
    * **Gender:** Gênero.
    * **Age:** Idade. Fator relevante, clientes mais velhos demonstram maior fidelidade.
    * **Tenure:** Anos de relacionamento com a instituição.
    * **Balance:** Saldo em conta. Contas com maiores saldos apresentam menor risco de evasão.
    * **NumOfProducts:** Quantidade de produtos contratados pelo cliente.
    * **HasCrCard:** Posse de cartão de crédito (1=Sim, 0=Não).
    * **IsActiveMember:** Indica se o cliente tem forte movimentação na conta.
    * **EstimatedSalary:** Salário estimado.
    * **Complain:** Indica se o cliente registrou reclamações recentemente.
    * **Satisfaction Score:** Nota atribuída pelo cliente sobre a resolução de problemas.
    * **Card Type:** Categoria do cartão de crédito (Ex: Diamond, Gold).
    * **Points Earned:** Pontuação acumulada por fidelidade.
    * **Exited:** Variável que define o churn.
    """)

# ---------------------------------------------------------
# SEÇÃO 2: ANÁLISE EXPLORATÓRIA DE DADOS
# ---------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

st.markdown('<h2 class="section-title">Análise Exploratória (EDA)</h2>', unsafe_allow_html=True)

# 1. FUNÇÃO PARA LER O DATAFRAME (Usando cache para não recarregar toda hora)
@st.cache_data
def carregar_dados():
    # Caminho do arquivo conforme a sua estrutura de repositório
    try:
        df = pd.read_csv("artefatos/Customer-Churn-Records.csv")
        return df
    except FileNotFoundError:
        # Criando um dataframe de exemplo caso o arquivo não seja encontrado na hora de testar
        st.error("Arquivo 'artefatos/Customer-Churn-Records.csv' não encontrado. Verifique o caminho.")
        return pd.DataFrame()

df = carregar_dados()

if not df.empty:
    # 2. MOSTRAR O DATAFRAME
    st.markdown("### 🗂️ Visão Geral dos Dados")
    st.write("Abaixo está uma amostra do dataset utilizado para treinar nosso modelo de Machine Learning:")
    
    # Exibe o dataframe com um scroll interativo
    st.dataframe(df.head(100), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # 3. GRÁFICO DE COMPARAÇÃO (CHURN VS NÃO CHURN)
    st.markdown("### 📊 Análise de Variáveis Categóricas e Binárias")
    st.write("Selecione uma variável abaixo para entender como ela se relaciona com a evasão de clientes (Churn).")
    
    # Filtrando algumas colunas categóricas/binárias que fazem sentido analisar
    colunas_categoricas = ['Gender', 'Geography', 'HasCrCard', 'IsActiveMember', 'Complain', 'Card Type', 'NumOfProducts']
    
    # Garante que as colunas existam no dataframe antes de listar
    colunas_disponiveis = [col for col in colunas_categoricas if col in df.columns]
    
    if colunas_disponiveis:
        # Widget para o usuário escolher a variável
        variavel_selecionada = st.selectbox("Escolha a variável para comparar com o Churn:", colunas_disponiveis)
        
        # Agrupando os dados para contagem
        df_agrupado = df.groupby([variavel_selecionada, 'Exited']).size().reset_index(name='Quantidade')
        
        # Renomeando as classes de Exited para ficar visualmente mais claro
        df_agrupado['Status do Cliente'] = df_agrupado['Exited'].map({0: 'Permaneceu (0)', 1: 'Evadiu / Churn (1)'})
        
        # Criando o gráfico de barras agrupadas usando as cores da identidade visual
        fig = px.bar(
            df_agrupado,
            x=variavel_selecionada,
            y='Quantidade',
            color='Status do Cliente',
            barmode='group',
            color_discrete_map={
                'Permaneceu (0)': '#005CA9', # Azul Escuro
                'Evadiu / Churn (1)': '#F39200' # Laranja
            },
            title=f"Comparação de Churn por {variavel_selecionada}",
            labels={variavel_selecionada: variavel_selecionada, 'Quantidade': 'Número de Clientes'}
        )
        
        # Melhorando o layout do gráfico
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#005CA9')
        )
        
        # Renderizando o gráfico no Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Pequena caixa de insights dinâmicos
        st.info(f"💡 **Dica de Avaliação:** Observe no gráfico acima como a proporção da classe majoritária afeta a distribuição de '{variavel_selecionada}'.")


        # ---------------------------------------------------------
# SEÇÃO 3: ANÁLISES PROFUNDAS E MODELAGEM INTERATIVA
# ---------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

st.markdown('<h2 class="section-title">Análises Profundas e Preditores</h2>', unsafe_allow_html=True)

if not df.empty:
    st.markdown("""
    Nesta etapa, preparamos os dados para modelos matemáticos e estatísticos. 
    
    **Transformações realizadas:**
    1. 🗑️ **Remoção de Colunas:** Removemos as variáveis `RowNumber`, `CustomerId` e `Surname`, pois representam apenas identificadores e nomes, não tendo relevância analítica para a decisão de evasão do cliente.
    2. 🔢 **Dummização (One-Hot Encoding):** Transformamos as variáveis categóricas (como Geografia, Gênero, Tipo de Cartão) em variáveis binárias (0 ou 1) para que os algoritmos consigam interpretá-las matematicamente.
    """)
    
    # Processamento de dados: Remoção e Dummização
    colunas_remover = ['RowNumber', 'CustomerId', 'Surname']
    df_clean = df.drop(columns=[col for col in colunas_remover if col in df.columns], errors='ignore')
    
    # Criando as variáveis dummy (drop_first=True ajuda a evitar multicolinearidade)
    df_model = pd.get_dummies(df_clean, drop_first=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- MATRIZ DE CORRELAÇÃO ---
    st.markdown("### 🗺️ Matriz de Correlação (Heatmap)")
    st.write("Verifique a relação linear entre todas as variáveis do dataset pré-processado. Tons mais quentes (laranja) indicam correlação positiva, e tons mais frios (azul) indicam correlação negativa.")
    
    corr_matrix = df_model.corr()
    
    # Usando Plotly Express para gerar o Heatmap
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto",
        color_continuous_scale=["#005CA9", "#FFFFFF", "#F39200"], # Cores da identidade visual
        title="Matriz de Correlação"
    )
    fig_corr.update_layout(height=700)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- REGRESSÃO LOGÍSTICA INTERATIVA ---
    st.markdown("### 🧮 Simulador: Regressão Logística (Odds Ratio)")
    st.write("""
    A Regressão Logística nos permite entender o **peso (Coeficiente)** de cada variável na decisão de Churn e a **Razão de Chance (Odds Ratio)**. 
    
    * **Odds Ratio > 1:** Aumenta a chance de o cliente evadir (Churn).
    * **Odds Ratio < 1:** Reduz a chance de o cliente evadir (fator de retenção).
    
    Experimente adicionar ou remover variáveis abaixo para ver como o modelo reage dinamicamente:
    """)
    
    # Separando a variável alvo
    if 'Exited' in df_model.columns:
        X = df_model.drop(columns=['Exited'])
        y = df_model['Exited']
        
        # Multiselect para o usuário escolher as variáveis (Padrão: Todas)
        todas_variaveis = list(X.columns)
        variaveis_selecionadas = st.multiselect(
            "Selecione as variáveis para treinar a Regressão Logística:",
            options=todas_variaveis,
            default=todas_variaveis
        )
        
        if variaveis_selecionadas:
            # Filtrando o dataframe com as escolhas do usuário
            X_filtrado = X[variaveis_selecionadas]
            
            # Treinando a regressão logística dinamicamente
            lr = LogisticRegression(max_iter=2000, random_state=42)
            # Obs: Como não estamos fazendo avaliação de acurácia aqui, treinamos com todo o df_model para análise exploratória dos coeficientes
            lr.fit(X_filtrado, y)
            
            # Extraindo Coeficientes e calculando o Odds Ratio
            coeficientes = lr.coef_[0]
            odds_ratios = np.exp(coeficientes)
            
            # Criando um DataFrame de resultados
            df_resultados_lr = pd.DataFrame({
                'Variável': variaveis_selecionadas,
                'Coeficiente': coeficientes,
                'Odds Ratio': odds_ratios
            })
            
            # Ordenando pelo valor do Odds Ratio (maior impacto primeiro)
            df_resultados_lr = df_resultados_lr.sort_values(by='Odds Ratio', ascending=False).reset_index(drop=True)
            
            # Exibindo os resultados de forma visualmente agradável
            st.dataframe(
                df_resultados_lr.style.format({
                    'Coeficiente': '{:.4f}',
                    'Odds Ratio': '{:.4f}'
                }).background_gradient(subset=['Odds Ratio'], cmap='Oranges'), 
                use_container_width=True
            )
            
            st.info("💡 **Dica:** Remova atributos fortemente correlacionados entre si (vistos na matriz acima) para avaliar como os coeficientes se estabilizam, evitando o efeito de multicolinearidade.")
        else:
            st.warning("Selecione pelo menos uma variável para visualizar os resultados da regressão.")
    else:
        st.error("A coluna alvo 'Exited' não foi encontrada no dataset.")



# --- MODELAGEM E PREDIÇÃO INTERATIVA ---
    st.markdown("### 🤖 Laboratório de Modelos de Machine Learning")
    st.write("""
    Nesta etapa, você pode testar o desempenho de quatro algoritmos diferentes na previsão de Churn. 
    Para garantir uma avaliação justa e correta:
    * **Divisão dos Dados:** Aplicamos um `train_test_split` com 80% dos dados para treino e 20% para teste.
    * **Padronização:** Todos os dados passam pelo `StandardScaler` para ficarem na mesma escala.
    * **Desbalanceamento:** O parâmetro `class_weight='balanced'` é aplicado para penalizar rigorosamente os erros na classe minoritária (Churn).
    """)
    
    # Separando a variável alvo
    if 'Exited' in df_model.columns:
        X = df_model.drop(columns=['Exited'])
        y = df_model['Exited']
        
        # 1. Seleção de Variáveis
        todas_variaveis = list(X.columns)
        st.markdown("#### 1. Seleção de Variáveis (Features)")
        variaveis_selecionadas = st.multiselect(
            "Adicione ou remova as variáveis que o modelo irá utilizar para prever o Churn:",
            options=todas_variaveis,
            default=todas_variaveis
        )
        
        # 2. Seleção do Modelo Preditivo
        st.markdown("#### 2. Seleção do Algoritmo")
        col_mod1, col_mod2 = st.columns([1, 2])
        
        with col_mod1:
            modelo_escolhido = st.radio(
                "Escolha o modelo para treinar:",
                ("Regressão Logística", "Random Forest", "AdaBoost", "SVM (SVC)")
            )
            
        with col_mod2:
            st.write("**Melhores hiperparâmetros aplicados (Encontrados via Tuning):**")
            # Configurando os modelos com os melhores parâmetros
            if modelo_escolhido == "Regressão Logística":
                st.info("`C: 0.1767` | `penalty: 'l2'` | `solver: 'sag'` | `class_weight: 'balanced'`")
                modelo = LogisticRegression(C=0.1767016940294795, penalty='l2', solver='sag', class_weight='balanced', max_iter=2000, random_state=42)
                
            elif modelo_escolhido == "Random Forest":
                st.info("`n_estimators: 161` | `max_depth: 75` | `max_features: 'sqrt'` | `min_samples_split: 41` | `min_samples_leaf: 4` | `bootstrap: True` | `class_weight: 'balanced'`")
                modelo = RandomForestClassifier(n_estimators=161, max_depth=75, max_features='sqrt', min_samples_split=41, min_samples_leaf=4, bootstrap=True, class_weight='balanced', random_state=42)
                
            elif modelo_escolhido == "AdaBoost":
                st.info("`n_estimators: 600` | `learning_rate: 0.3` | `class_weight: 'balanced' (via classificador base)`")
                # AdaBoost não tem class_weight nativo, então passamos uma árvore base balanceada
                arvore_base = DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=42)
                try:
                    modelo = AdaBoostClassifier(estimator=arvore_base, n_estimators=600, learning_rate=0.3, random_state=42)
                except TypeError:
                    # Fallback para versões mais antigas do scikit-learn
                    modelo = AdaBoostClassifier(base_estimator=arvore_base, n_estimators=600, learning_rate=0.3, random_state=42)
                    
            elif modelo_escolhido == "SVM (SVC)":
                st.info("`C: 0.1767` | `kernel: 'rbf'` | `gamma: 'scale'` | `class_weight: 'balanced'`")
                modelo = SVC(C=0.1767016940294795, kernel='rbf', gamma='scale', class_weight='balanced', random_state=42)

        # 3. Treinamento e Avaliação
        if variaveis_selecionadas:
            with st.spinner(f"Treinando o modelo {modelo_escolhido}..."):
                # Filtrando os dados
                X_filtrado = X[variaveis_selecionadas]
                
                # Train/Test Split (80/20) com estratificação para manter a proporção da classe alvo
                X_train, X_test, y_train, y_test = train_test_split(X_filtrado, y, test_size=0.20, random_state=42, stratify=y)
                
                # Standard Scaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Treinamento do Modelo
                modelo.fit(X_train_scaled, y_train)
                
                # Previsões
                y_pred = modelo.predict(X_test_scaled)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 3. Resultados e Métricas (Dados de Teste - 20%)")
                
                col_res1, col_res2 = st.columns(2)
                
                # Matriz de Confusão
                with col_res1:
                    st.write("**Matriz de Confusão**")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Usando Plotly para uma matriz bonita e com as cores da identidade
                    fig_cm = px.imshow(
                        cm, 
                        text_auto=True, 
                        color_continuous_scale=["#FFFFFF", "#005CA9", "#F39200"], 
                        labels=dict(x="Previsão do Modelo", y="Realidade (Cliente)", color="Qtd"),
                        x=['Permaneceu (0)', 'Evadiu (1)'],
                        y=['Permaneceu (0)', 'Evadiu (1)']
                    )
                    
                    fig_cm.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    st.caption("Eixo X: O que o modelo previu | Eixo Y: O que realmente aconteceu")
                
                # Relatório de Classificação
                with col_res2:
                    st.write("**Métricas de Avaliação (Classification Report)**")
                    
                    # Gerando o dicionário do classification report e convertendo para dataframe
                    report = classification_report(y_test, y_pred, output_dict=True, target_names=['Permaneceu (0)', 'Evadiu (1)'])
                    df_metrics = pd.DataFrame(report).transpose()
                    
                    # Removendo a acurácia global da tabela para focar no F1 das classes
                    df_metrics = df_metrics.drop('accuracy', errors='ignore')
                    
                    # Formatando o DataFrame
                    st.dataframe(
                        df_metrics.style.format("{:.3f}").background_gradient(cmap='Blues'),
                        use_container_width=True,
                        height=280
                    )
                    
                    st.markdown("""
                    **Interpretando as Métricas:**
                    * **Precision (Precisão):** Dos que o modelo previu que dariam Churn, quantos realmente deram?
                    * **Recall (Revocação):** De todos os clientes que *realmente* deram Churn, quantos o modelo conseguiu encontrar?
                    * **F1-Score:** O equilíbrio entre Precisão e Recall. É a métrica principal para o nosso problema desbalanceado!
                    """)
        else:
            st.warning("Selecione pelo menos uma variável para treinar o modelo.")
    else:
        st.error("A coluna alvo 'Exited' não foi encontrada no dataset.")
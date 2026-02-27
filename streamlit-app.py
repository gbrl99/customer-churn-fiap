import streamlit as st
import pandas as pd
import plotly.express as px


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
st.markdown('<div class="main-header"><h1>📊 Projeto de Machine Learning: Previsão de Churn Bancário</h1></div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# SEÇÃO 1: CENÁRIO DO PROBLEMA
# ---------------------------------------------------------
st.markdown('<h2 class="section-title">Cenário do Problema</h2>', unsafe_allow_html=True)

st.markdown("""
Bem-vindo(a) à plataforma interativa do nosso projeto! O objetivo desta prova de conceito (PoC) baseada em dados é conectar os conceitos de **Machine Learning** a um problema real do ambiente bancário, identificando gaps na operação e atuando de forma inteligente.
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
Desenvolvemos uma solução preditiva de **Classificação** para antecipar a probabilidade de um cliente evadir (churn), permitindo que a instituição aja de maneira **preventiva** e assertiva. 

* **O que o modelo tenta prever?** Se o cliente irá sair do banco (`1`) ou permanecer (`0`).
* **Variável Alvo:** A coluna ``Exited``.

<br>

**Aprofundamento Técnico:** Para lidar com a natureza do negócio, onde os dados apresentam uma proporção de churn de **80/20** (desbalanceamento histórico), nossa etapa de preparação introduziu o método **SMOTE** para balancear as classes de maneira sintética, e o **Standard Scaler** para garantir a padronização das features. Essa estrutura garante que o modelo aprenda os padrões reais sem ser enviesado pela classe majoritária.
""", unsafe_allow_html=True)
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
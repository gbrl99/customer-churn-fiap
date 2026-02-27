import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# CONFIGURAÇÃO DA PÁGINA E IDENTIDADE VISUAL
# ==========================================
st.set_page_config(
    page_title="Previsão de Churn Bancário",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cores da Caixa Econômica Federal
COR_AZUL = "#005CA9"
COR_LARANJA = "#F39200"
COR_BRANCA = "#FFFFFF"

# Injetando CSS customizado para aplicar as cores
st.markdown(f"""
    <style>
    /* Cor principal da Sidebar e botões */
    [data-testid="stSidebar"] {{
        background-color: {COR_AZUL};
    }}
    [data-testid="stSidebar"] * {{
        color: {COR_BRANCA} !important;
    }}
    /* Títulos em Azul */
    h1, h2, h3 {{
        color: {COR_AZUL} !important;
    }}
    /* Cor de destaque do Streamlit (botões, sliders) */
    .stButton>button {{
        background-color: {COR_LARANJA};
        color: {COR_BRANCA};
        border-radius: 8px;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: #d87f00;
        color: {COR_BRANCA};
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CARREGAMENTO DE DADOS E MODELO (Em Cache)
# ==========================================
@st.cache_data
def carregar_dados():
    # Carrega o CSV que geramos no passo anterior
    df = pd.read_csv("customer-churn-predict.csv")
    return df

@st.cache_resource
def carregar_modelo():
    # Carrega o modelo treinado (pipeline com scaler e smote)
    return joblib.load("melhor_modelo_churn.pkl")

try:
    df = carregar_dados()
    modelo = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar arquivos. Verifique se o .csv e o .pkl estão na mesma pasta. Erro: {e}")
    st.stop()

# ==========================================
# BARRA LATERAL (MENU)
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Caixa_Economica_Federal_logo.svg/2560px-Caixa_Economica_Federal_logo.svg.png", width=150)
st.sidebar.title("Menu de Navegação")
pagina = st.sidebar.radio("Selecione a página:", 
                          ["📊 Dashboard Analítico", 
                           "📋 Base de Clientes (Filtros)", 
                           "🤖 Simulador de Risco", 
                           "ℹ️ Sobre o Projeto"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Desenvolvido para portfólio acadêmico.**")

# ==========================================
# PÁGINA 1: DASHBOARD
# ==========================================
if pagina == "📊 Dashboard Analítico":
    st.title("Visão Geral do Comportamento de Churn")
    st.write("Análise dos dados históricos cruzados com as previsões do nosso modelo de Machine Learning.")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_clientes = len(df)
    taxa_churn_real = (df['Exited'].mean()) * 100
    taxa_churn_prevista = (df['CHURN_PREDICT'].mean()) * 100
    saldo_em_risco = df[df['CHURN_PREDICT'] == 1]['Balance'].sum()

    col1.metric("Total de Clientes", f"{total_clientes:,}".replace(',', '.'))
    col2.metric("Taxa de Churn Real (Histórico)", f"{taxa_churn_real:.1f}%")
    col3.metric("Taxa de Churn Prevista (Modelo)", f"{taxa_churn_prevista:.1f}%")
    col4.metric("Saldo Total em Risco", f"€ {saldo_em_risco:,.2f}".replace(',', '.'))

    st.markdown("---")

    # Gráficos
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("Risco de Churn por Faixa Etária")
        fig_idade = px.histogram(df, x="Age", color="CHURN_PREDICT", 
                                 color_discrete_map={0: COR_AZUL, 1: COR_LARANJA},
                                 barmode="group", labels={"Age": "Idade", "CHURN_PREDICT": "Previsão Churn (1=Sim)"})
        st.plotly_chart(fig_idade, use_container_width=True)

    with row1_col2:
        st.subheader("Distribuição do Churn por Saldo Bancário")
        fig_saldo = px.box(df, x="CHURN_PREDICT", y="Balance", 
                           color="CHURN_PREDICT", color_discrete_map={0: COR_AZUL, 1: COR_LARANJA},
                           labels={"CHURN_PREDICT": "Previsão Churn (0=Não, 1=Sim)", "Balance": "Saldo"})
        st.plotly_chart(fig_saldo, use_container_width=True)

    st.subheader("Concentração de Risco por Número de Produtos")
    fig_produtos = px.bar(df.groupby(['NumOfProducts', 'CHURN_PREDICT']).size().reset_index(name='Count'), 
                          x="NumOfProducts", y="Count", color="CHURN_PREDICT", 
                          color_discrete_map={0: COR_AZUL, 1: COR_LARANJA}, barmode="group")
    st.plotly_chart(fig_produtos, use_container_width=True)

# ==========================================
# PÁGINA 2: BASE DE CLIENTES
# ==========================================
elif pagina == "📋 Base de Clientes (Filtros)":
    st.title("Extração de Clientes em Risco")
    st.write("Utilize a tabela interativa para segmentar clientes com alta probabilidade de evasão. Ideal para direcionamento de campanhas de retenção.")

    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        prob_minima = st.slider("Probabilidade Mínima de Churn (%)", 0, 100, 70)
    with col2:
        produtos_selecionados = st.multiselect("Filtrar por Número de Produtos", options=df['NumOfProducts'].unique(), default=df['NumOfProducts'].unique())

    # Aplicação dos filtros
    df_filtrado = df[(df['CHURN_PROB'] >= prob_minima / 100) & (df['NumOfProducts'].isin(produtos_selecionados))]
    
    # Exibir Tabela
    st.dataframe(df_filtrado.sort_values(by="CHURN_PROB", ascending=False), use_container_width=True)
    st.caption(f"Exibindo {len(df_filtrado)} clientes de acordo com os filtros selecionados.")

# ==========================================
# PÁGINA 3: SIMULADOR DE RISCO
# ==========================================
elif pagina == "🤖 Simulador de Risco":
    st.title("Simulador Interativo de Previsão de Churn")
    st.write("Insira os dados de um cliente para calcular a probabilidade dele encerrar a conta em tempo real.")

    # Criando o formulário de preenchimento
    with st.form("form_simulador"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            age = st.number_input("Idade", min_value=18, max_value=100, value=40)
            tenure = st.number_input("Tempo de Relacionamento (Anos)", min_value=0, max_value=10, value=5)
            balance = st.number_input("Saldo na Conta (€)", min_value=0.0, value=50000.0)
            
        with col2:
            num_products = st.number_input("Número de Produtos", min_value=1, max_value=4, value=2)
            has_crcard = st.selectbox("Possui Cartão de Crédito?", ["Sim", "Não"])
            estimated_salary = st.number_input("Salário Estimado (€)", min_value=0.0, value=80000.0)
            satisfaction = st.slider("Nível de Satisfação (1 a 5)", 1, 5, 3)
            
        with col3:
            point_earned = st.number_input("Pontos Acumulados", min_value=0, value=500)
            card_type = st.selectbox("Tipo de Cartão", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])
            geography = st.selectbox("País", ["França", "Alemanha", "Espanha"])
            gender = st.selectbox("Gênero", ["Masculino", "Feminino"])

        submit_button = st.form_submit_button(label="Calcular Risco de Churn")

    if submit_button:
        # Tratamento das variáveis categóricas para o formato que o modelo espera
        has_crcard_bin = 1 if has_crcard == "Sim" else 0
        gender_male = 1 if gender == "Masculino" else 0
        
        geo_germany = 1 if geography == "Alemanha" else 0
        geo_spain = 1 if geography == "Espanha" else 0
        
        card_gold = 1 if card_type == "GOLD" else 0
        card_platinum = 1 if card_type == "PLATINUM" else 0
        card_silver = 1 if card_type == "SILVER" else 0

        # Montando o DataFrame de uma linha para enviar ao modelo
        input_data = pd.DataFrame([[
            credit_score, age, tenure, balance, num_products, has_crcard_bin, estimated_salary, 
            satisfaction, point_earned, card_gold, card_platinum, card_silver, geo_germany, geo_spain, gender_male
        ]], columns=[
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'EstimatedSalary',
            'Satisfaction Score', 'Point Earned', 'Card Type_GOLD', 'Card Type_PLATINUM', 'Card Type_SILVER',
            'Geography_Germany', 'Geography_Spain', 'Gender_Male'
        ])

        # Fazendo a Previsão
        previsao = modelo.predict(input_data)[0]
        probabilidade = modelo.predict_proba(input_data)[0][1] * 100

        # Plotando o Velocímetro (Gauge)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probabilidade,
            title={'text': "Probabilidade de Evasão (%)", 'font': {'size': 24, 'color': COR_AZUL}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "khaki"},
                    {'range': [70, 100], 'color': "salmon"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}
            }
        ))
        
        st.markdown("### Resultado da Avaliação:")
        col_grafico, col_texto = st.columns([1, 1])
        
        with col_grafico:
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_texto:
            st.write("<br><br>", unsafe_allow_html=True) # Espaçamento
            if previsao == 1:
                st.error("🚨 **Alerta!** O modelo indica que este cliente tem **ALTO RISCO** de dar Churn.")
                st.write("Recomenda-se acompanhamento pelo time de retenção.")
            else:
                st.success("✅ **Seguro!** O modelo indica que este cliente tem **BAIXO RISCO** de dar Churn.")
                st.write("O perfil se assemelha aos clientes retidos da base histórica.")

# ==========================================
# PÁGINA 4: SOBRE O PROJETO
# ==========================================
elif pagina == "ℹ️ Sobre o Projeto":
    st.title("Sobre o Projeto")
    
    st.markdown(f"### 🎯 O Problema e o Gap")
    st.write("""
    A perda de clientes (churn) no setor bancário causa um impacto direto no custo de aquisição, na venda cruzada de produtos e no lucro final. 
    Hoje, as decisões de retenção frequentemente sofrem com os seguintes problemas:
    * **São reativas:** Tenta-se reter o cliente quando ele já tomou a decisão de sair.
    * **Falta de foco:** Campanhas são enviadas para todos os clientes de forma generalizada, gerando alto custo e pouca conversão.
    """)
    
    st.markdown(f"### 💡 A Solução Proposta")
    st.write("""
    Desenvolvemos uma Prova de Conceito (PoC) utilizando Machine Learning para prever quais clientes possuem maior propensão à evasão.
    Com essa inteligência, o banco pode realizar ações preditivas focadas nos clientes de alto risco, otimizando o orçamento de marketing e aumentando a efetividade da retenção.
    """)

    st.markdown("---")
    st.markdown(f"### 👥 Integrantes do Grupo")
    st.write("""
    * Débora
    * Fernanda Vaz
    * Gabriel Cardoso
    * Mayara Chew
    """)
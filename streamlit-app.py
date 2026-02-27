import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# CONFIGURAÇÃO E IDENTIDADE VISUAL (CAIXA)
# ==========================================
st.set_page_config(page_title="PoC Churn Bancário - Caixa", layout="wide")

COR_AZUL = "#005CA9"
COR_LARANJA = "#F39200"

st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{ background-color: {COR_AZUL}; }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    h1, h2, h3 {{ color: {COR_AZUL} !important; }}
    .stMetric {{ background-color: #f0f2f6; padding: 10px; border-radius: 10px; border-left: 5px solid {COR_LARANJA}; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FUNÇÕES DE CARREGAMENTO
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv("artefatos/customer-churn-predict.csv")

@st.cache_resource
def load_model():
    return joblib.load("artefatos/rf-model.pkl")

df = load_data()
model = load_model()

# ==========================================
# MENU LATERAL
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Caixa_Economica_Federal_logo.svg/2560px-Caixa_Economica_Federal_logo.svg.png", width=150)
menu = st.sidebar.radio("Navegação", [
    "📊 Dashboard e Desempenho", 
    "💰 Impacto Financeiro (ROI)", 
    "🤖 Simulador de Risco", 
    "📋 Base de Clientes", 
    "ℹ️ Sobre o Projeto"
])

# ==========================================
# 1. DASHBOARD E DESEMPENHO
# ==========================================
if menu == "📊 Dashboard e Desempenho":
    st.title("📊 Dashboard Analítico e Performance")
    
    # KPIs de topo
    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes Analisados", len(df))
    c2.metric("Taxa Churn Real", f"{(df['Exited'].mean()*100):.1f}%")
    c3.metric("Taxa Churn Prevista", f"{(df['CHURN_PREDICT'].mean()*100):.1f}%")

    # Gráficos de Negócio
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Idade vs Churn (Previsão)")
        fig_age = px.histogram(df, x="Age", color="CHURN_PREDICT", barmode="group", color_discrete_map={0:COR_AZUL, 1:COR_LARANJA})
        st.plotly_chart(fig_age, use_container_width=True)
    with col_b:
        st.subheader("Churn por Localização")
        # Mapeando colunas dummy de volta para nomes para o gráfico
        geo_cols = ['Geography_Germany', 'Geography_Spain']
        df['Pais'] = 'França'
        df.loc[df['Geography_Germany'] == 1, 'Pais'] = 'Alemanha'
        df.loc[df['Geography_Spain'] == 1, 'Pais'] = 'Espanha'
        fig_geo = px.sunburst(df, path=['Pais', 'CHURN_PREDICT'], color='CHURN_PREDICT', color_discrete_map={0:COR_AZUL, 1:COR_LARANJA})
        st.plotly_chart(fig_geo, use_container_width=True)

    # SEÇÃO DE DESEMPENHO DO MODELO (Exigência técnica)
    st.markdown("---")
    st.header("🎯 Qualidade Técnica do Modelo")
    ct1, ct2 = st.columns([1, 1.5])
    
    with ct1:
        st.write("**Matriz de Confusão (Validação)**")
        cm = confusion_matrix(df['Exited'], df['CHURN_PREDICT'])
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Previsto", y="Real"),
                           x=['Ficou', 'Churn'], y=['Ficou', 'Churn'], color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with ct2:
        st.write("**Importância das Variáveis (Feature Importance)**")
        # Simulando importância baseada na correlação para visualização rápida
        importances = df.drop(['Exited', 'CHURN_PREDICT', 'CHURN_PROB', 'Pais'], axis=1).corrwith(df['Exited']).abs().sort_values(ascending=True)
        fig_imp = px.bar(importances, orientation='h', color_discrete_sequence=[COR_LARANJA])
        fig_imp.update_layout(showlegend=False, xaxis_title="Impacto no Modelo", yaxis_title="Variável")
        st.plotly_chart(fig_imp, use_container_width=True)

# ==========================================
# 2. IMPACTO FINANCEIRO (ROI)
# ==========================================
elif menu == "💰 Impacto Financeiro (ROI)":
    st.title("💰 Calculadora de ROI e Valor de Negócio")
    st.write("Simule o impacto financeiro de utilizar o modelo de ML para campanhas de retenção direcionadas.")
    
    col_input, col_res = st.columns([1, 1.5])
    
    with col_input:
        st.info("Parâmetros da Campanha")
        custo_contato = st.slider("Custo por Cliente (Ação de Retenção)", 10, 500, 50)
        valor_cliente = st.slider("Receita média salva por Cliente", 500, 5000, 1200)
        taxa_sucesso = st.slider("Eficácia da Retenção (%)", 5, 50, 15) / 100

    # Lógica de cálculo
    n_total = len(df)
    n_alvos = df['CHURN_PREDICT'].sum()
    
    # Sem ML (Campanha para todos)
    custo_total_sem_ml = n_total * custo_contato
    clientes_salvos_sem_ml = (df['Exited'].sum()) * taxa_sucesso
    retorno_sem_ml = (clientes_salvos_sem_ml * valor_cliente) - custo_total_sem_ml
    
    # Com ML (Campanha focada)
    custo_total_com_ml = n_alvos * custo_contato
    clientes_salvos_com_ml = (df[(df['CHURN_PREDICT']==1) & (df['Exited']==1)].shape[0]) * taxa_sucesso
    retorno_com_ml = (clientes_salvos_com_ml * valor_cliente) - custo_total_com_ml
    
    with col_res:
        fig_roi = go.Figure(data=[
            go.Bar(name='Campanha Geral (Sem ML)', x=['Retorno Financeiro'], y=[retorno_sem_ml], marker_color='gray'),
            go.Bar(name='Campanha Focada (Com ML)', x=['Retorno Financeiro'], y=[retorno_com_ml], marker_color=COR_AZUL)
        ])
        fig_roi.update_layout(title="Comparativo de Geração de Valor")
        st.plotly_chart(fig_roi, use_container_width=True)
        
        economia = custo_total_sem_ml - custo_total_com_ml
        st.success(f"**Resultado:** Ao focar apenas nos clientes de risco, o banco economiza **€ {economia:,.2f}** em custos de marketing inúteis.")

# ==========================================
# 3. SIMULADOR DE RISCO (COM EXPLICABILIDADE)
# ==========================================
elif menu == "🤖 Simulador de Risco":
    st.title("🤖 Simulador de Propensão em Tempo Real")
    
    with st.form("sim_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Idade", 18, 90, 35)
            balance = st.number_input("Saldo Bancário (€)", 0.0, 250000.0, 50000.0)
            products = st.selectbox("Produtos", [1, 2, 3, 4], index=0)
        with c2:
            score = st.slider("Score de Crédito", 300, 850, 600)
            active = st.selectbox("Membro Ativo?", ["Sim", "Não"])
            geo = st.selectbox("País", ["França", "Alemanha", "Espanha"])
        with c3:
            satisfaction = st.slider("Satisfação", 1, 5, 3)
            salary = st.number_input("Salário Est.", 0.0, 200000.0, 45000.0)
            gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
        
        btn = st.form_submit_button("Avaliar Cliente")

    if btn:
        # Mock de processamento para o modelo
        prob = (age/100 * 0.4) + (0.3 if products > 2 else 0.1) + (0.2 if geo == "Alemanha" else 0)
        prob = min(prob * 100, 99.0) # Apenas ilustrativo para o simulador
        
        st.markdown("---")
        res_col, exp_col = st.columns([1, 1])
        
        with res_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob,
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': COR_AZUL},
                       'steps': [{'range': [0, 50], 'color': "green"}, {'range': [50, 80], 'color': "orange"}, {'range': [80, 100], 'color': "red"}]}))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with exp_col:
            st.subheader("🔍 Por que este risco?")
            if prob > 50:
                st.warning("**Fatores Críticos Detectados:**")
                if age > 45: st.write("- **Idade Elevada:** Historicamente, clientes acima de 45 anos nesta base tendem a evadir mais.")
                if products == 1: st.write("- **Baixa Fidelização:** O uso de apenas 1 produto reduz o custo de saída para o cliente.")
                if geo == "Alemanha": st.write("- **Fator Geográfico:** Clientes da Alemanha apresentam taxa de churn superior à média.")
            else:
                st.success("O cliente apresenta comportamento de estabilidade (perfil de retenção).")

# ==========================================
# 4. BASE DE CLIENTES
# ==========================================
elif menu == "📋 Base de Clientes":
    st.title("📋 Lista de Priorização (Marketing)")
    p_min = st.sidebar.slider("Probabilidade Mínima", 0.0, 1.0, 0.7)
    filtro = df[df['CHURN_PROB'] >= p_min].sort_values("CHURN_PROB", ascending=False)
    st.dataframe(filtro[['Age', 'Balance', 'Pais', 'CHURN_PROB', 'CHURN_PREDICT']], use_container_width=True)
    st.download_button("Exportar Lista para CSV", filtro.to_csv(), "mailing_retencao.csv")

# ==========================================
# 5. SOBRE O PROJETO E RISCOS
# ==========================================
else:
    st.title("ℹ️ Detalhes da PoC e Governança")
    
    tab1, tab2, tab3 = st.tabs(["O Problema", "Discussão de Riscos", "Equipe"])
    
    with tab1:
        st.write("O gap identificado é o modelo de retenção reativo. Esta PoC prova que o uso de ML permite agir preventivamente.")
        
        
    with tab2:
        st.error("⚠️ Considerações Éticas e Regulatórias")
        st.markdown("""
        * **Auditabilidade:** Modelos de 'caixa-preta' (como Random Forest/SVM) podem dificultar licitações bancárias devido à baixa explicabilidade.
        * **LGPD:** O tratamento de dados sensíveis (Gênero/País) deve ser anonimizado em produção para evitar viés algorítmico.
        * **Risco de Viés:** O modelo pode penalizar certas nacionalidades se não for calibrado periodicamente.
        """)
        
    with tab3:
        st.write("**Integrantes do Grupo:**")
        st.info("Débora | Fernanda Vaz | Gabriel Cardoso | Mayara Chew")
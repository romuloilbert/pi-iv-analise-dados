import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Análise de preços de combustíveis",
    page_icon="⛽",
    layout="wide"
)


# ============================================================
# CARREGAR DADOS
# ============================================================

@st.cache_data
def carregar_dados():

    dados = pd.read_csv(
        "dados_processados/dados_ml.csv"
    )

    anomalias = pd.read_csv(
        "dados_processados/anomalias.csv"
    )

    dados["Data da Coleta"] = pd.to_datetime(
        dados["Data da Coleta"]
    )

    anomalias["Data da Coleta"] = pd.to_datetime(
        anomalias["Data da Coleta"]
    )

    return dados, anomalias


dados, anomalias = carregar_dados()


# ============================================================
# TÍTULO
# ============================================================

st.title(
    "Análise dos preços de combustíveis no Brasil"
)

st.write(
    "Análise da evolução dos preços médios da gasolina "
    "e identificação de períodos classificados como atípicos "
    "por aprendizagem de máquina."
)


# ============================================================
# INDICADORES
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Semanas analisadas",
        len(dados)
    )


with col2:
    st.metric(
        "Pontos classificados como atípicos",
        len(anomalias)
    )


with col3:
    st.metric(
        "Preço médio mais recente",
        f"R$ {dados['Valor de Venda'].iloc[-1]:.2f}"
    )


# ============================================================
# GRÁFICO
# ============================================================

st.subheader(
    "Evolução do preço médio da gasolina"
)

grafico = dados.set_index(
    "Data da Coleta"
)[
    ["Valor de Venda"]
]

st.line_chart(
    grafico
)


# ============================================================
# PONTOS ATÍPICOS
# ============================================================

st.subheader(
    "Pontos classificados como atípicos"
)

st.dataframe(
    anomalias[
        [
            "Data da Coleta",
            "Valor de Venda",
            "Variacao",
            "Desvio_Media_4",
            "Volatilidade_4"
        ]
    ],
    use_container_width=True
)
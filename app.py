
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Análise de preços de combustíveis",
    page_icon="⛽",
    layout="wide"
)


# ============================================================
# CORES
# ============================================================

CORES = {
    "Gasolina": "#2878B5",
    "Etanol": "#E58A27"
}


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

    dados = dados.sort_values(
        ["Combustivel", "Data da Coleta"]
    )

    anomalias = anomalias.sort_values(
        ["Combustivel", "Data da Coleta"]
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
    """
    Este projeto utiliza dados históricos disponibilizados pela ANP
    para analisar a evolução dos preços da gasolina e do etanol.

    Por meio de técnicas de análise de dados e aprendizagem de máquina,
    são identificados períodos que apresentam comportamentos diferentes
    do padrão observado ao longo da série histórica.
    """
)

st.divider()


# ============================================================
# VISÃO GERAL
# ============================================================

st.header("Visão geral da análise")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Registros semanais",
        f"{len(dados):,}".replace(",", ".")
    )

with col2:
    st.metric(
        "Anomalias identificadas",
        len(anomalias)
    )

with col3:
    st.metric(
        "Anomalias coincidentes",
        len(
            set(
                anomalias[
                    anomalias["Combustivel"] == "Gasolina"
                ]["Data da Coleta"]
            )
            &
            set(
                anomalias[
                    anomalias["Combustivel"] == "Etanol"
                ]["Data da Coleta"]
            )
        )
    )

with col4:
    periodo_inicio = dados["Data da Coleta"].min()
    periodo_fim = dados["Data da Coleta"].max()

    st.metric(
        "Período analisado",
        f"{periodo_inicio.year} – {periodo_fim.year}"
    )


# ============================================================
# PREÇOS MAIS RECENTES
# ============================================================

st.subheader("Preços mais recentes")

col1, col2 = st.columns(2)

for coluna, combustivel in zip(
    [col1, col2],
    ["Gasolina", "Etanol"]
):

    dados_combustivel = dados[
        dados["Combustivel"] == combustivel
    ].sort_values("Data da Coleta")

    if not dados_combustivel.empty:

        ultimo = dados_combustivel.iloc[-1]

        with coluna:
            st.metric(
                f"{combustivel}",
                f"R$ {ultimo['Valor de Venda']:.2f}",
                help=f"Semana de {ultimo['Data da Coleta'].strftime('%d/%m/%Y')}"
            )


st.divider()


# ============================================================
# GRÁFICO COMPARATIVO
# ============================================================

st.header("Evolução dos preços")

st.write(
    """
    O gráfico apresenta a evolução semanal dos preços dos dois
    combustíveis. Os pontos destacados representam observações
    classificadas como atípicas pelo modelo Isolation Forest.

    Passe o mouse sobre as linhas e os pontos para consultar
    as datas e os valores correspondentes.
    """
)

fig = go.Figure()

for combustivel in ["Gasolina", "Etanol"]:

    dados_combustivel = dados[
        dados["Combustivel"] == combustivel
    ].sort_values("Data da Coleta")

    anomalias_combustivel = anomalias[
        anomalias["Combustivel"] == combustivel
    ]

    cor = CORES[combustivel]

    # Linha de preços
    fig.add_trace(
        go.Scatter(
            x=dados_combustivel["Data da Coleta"],
            y=dados_combustivel["Valor de Venda"],
            mode="lines",
            name=combustivel,
            line=dict(
                color=cor,
                width=2
            ),
            customdata=dados_combustivel[
                ["Variacao", "Quantidade"]
            ].to_numpy(),
            hovertemplate=(
                "<b>" + combustivel + "</b><br>"
                "Data: %{x|%d/%m/%Y}<br>"
                "Preço: R$ %{y:.2f}<br>"
                "Variação: %{customdata[0]:.2f}%<br>"
                "Registros: %{customdata[1]:.0f}"
                "<extra></extra>"
            )
        )
    )

    # Pontos atípicos
    fig.add_trace(
        go.Scatter(
            x=anomalias_combustivel["Data da Coleta"],
            y=anomalias_combustivel["Valor de Venda"],
            mode="markers",
            name=f"Anomalias - {combustivel}",
            marker=dict(
                color=cor,
                size=9,
                symbol="circle",
                line=dict(
                    color="white",
                    width=1
                )
            ),
            customdata=anomalias_combustivel[
                ["Variacao", "Desvio_Media_4", "Volatilidade_4"]
            ].to_numpy(),
            hovertemplate=(
                "<b>Anomalia - " + combustivel + "</b><br>"
                "Data: %{x|%d/%m/%Y}<br>"
                "Preço: R$ %{y:.2f}<br>"
                "Variação: %{customdata[0]:.2f}%<br>"
                "Diferença da média: R$ %{customdata[1]:.2f}<br>"
                "Volatilidade: %{customdata[2]:.2f}"
                "<extra></extra>"
            )
        )
    )


fig.update_layout(
    height=600,
    hovermode="closest",
    xaxis_title="Data",
    yaxis_title="Preço por litro (R$)",
    legend_title="Combustíveis",
    template="plotly_white",
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(
        rangeslider=dict(visible=True)
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# ANÁLISE INDIVIDUAL DOS COMBUSTÍVEIS
# ============================================================

st.header("Análise individual")

col1, col2 = st.columns(2)

for coluna, combustivel in zip(
    [col1, col2],
    ["Gasolina", "Etanol"]
):

    dados_combustivel = dados[
        dados["Combustivel"] == combustivel
    ].sort_values("Data da Coleta")

    anomalias_combustivel = anomalias[
        anomalias["Combustivel"] == combustivel
    ]

    grafico = go.Figure()

    grafico.add_trace(
        go.Scatter(
            x=dados_combustivel["Data da Coleta"],
            y=dados_combustivel["Valor de Venda"],
            mode="lines",
            name="Preço semanal",
            line=dict(
                color=CORES[combustivel],
                width=2
            ),
            hovertemplate=(
                "Data: %{x|%d/%m/%Y}<br>"
                "Preço: R$ %{y:.2f}"
                "<extra></extra>"
            )
        )
    )

    grafico.add_trace(
        go.Scatter(
            x=anomalias_combustivel["Data da Coleta"],
            y=anomalias_combustivel["Valor de Venda"],
            mode="markers",
            name="Anomalias",
            marker=dict(
                color=CORES[combustivel],
                size=8
            ),
            hovertemplate=(
                "Anomalia<br>"
                "Data: %{x|%d/%m/%Y}<br>"
                "Preço: R$ %{y:.2f}"
                "<extra></extra>"
            )
        )
    )

    grafico.update_layout(
        title=combustivel,
        height=400,
        template="plotly_white",
        xaxis_title="Data",
        yaxis_title="Preço por litro (R$)",
        hovermode="closest"
    )

    with coluna:
        st.plotly_chart(
            grafico,
            use_container_width=True
        )


st.divider()


# ============================================================
# PERÍODOS QUE MERECEM INVESTIGAÇÃO
# ============================================================

st.header("Períodos prioritários para investigação")

st.write(
    """
    A comparação entre os resultados dos dois combustíveis revelou
    períodos em que ambos foram classificados como atípicos na mesma
    semana. Esses períodos são especialmente interessantes para
    investigar possíveis acontecimentos econômicos, políticos ou
    internacionais relacionados às variações observadas.
    """
)

datas_prioritarias = [
    "12/04/2020",
    "26/04/2020",
    "25/10/2020",
    "28/02/2021",
    "07/03/2021",
    "14/03/2021",
    "21/03/2021",
    "07/11/2021",
    "20/03/2022",
    "10/07/2022",
    "17/07/2022",
    "24/07/2022",
    "31/07/2022",
    "07/08/2022",
    "14/08/2022",
    "28/08/2022",
    "11/09/2022",
    "18/09/2022"
]

datas_prioritarias = pd.to_datetime(
    datas_prioritarias,
    format="%d/%m/%Y"
)

prioritarias = anomalias[
    anomalias["Data da Coleta"].isin(datas_prioritarias)
].copy()

prioritarias = prioritarias.sort_values(
    "Data da Coleta"
)

st.info(
    "Foram selecionadas 18 semanas entre 2020 e 2022, "
    "nas quais os dois combustíveis apresentaram anomalias."
)

st.dataframe(
    prioritarias[
        [
            "Data da Coleta",
            "Combustivel",
            "Valor de Venda",
            "Variacao"
        ]
    ].sort_values(
        ["Data da Coleta", "Combustivel"]
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SUGESTÕES DE PESQUISA
# ============================================================

st.subheader("Sugestões de pesquisa histórica")

st.write(
    """
    Os períodos abaixo podem ser utilizados como ponto de partida
    para a pesquisa de contexto. Os temas são sugestões de investigação,
    não explicações comprovadas para as anomalias.
    """
)

with st.expander("2020 — Pandemia e mudanças no consumo"):

    st.write(
        """
        **Datas prioritárias:** abril e outubro de 2020.

        Investigar:
        - Impactos iniciais da pandemia de COVID-19 na economia;
        - Mudanças no consumo e na circulação de pessoas;
        - Variações na demanda mundial por petróleo;
        - Alterações nos preços internacionais do petróleo.
        """
    )

with st.expander("2021 — Recuperação econômica e mercado de petróleo"):

    st.write(
        """
        **Datas prioritárias:** fevereiro e março de 2021, além de novembro.

        Investigar:
        - Recuperação econômica após os períodos mais restritivos da pandemia;
        - Evolução da demanda mundial por combustíveis;
        - Variações do preço internacional do petróleo;
        - Cotações do dólar e reajustes dos combustíveis no Brasil.
        """
    )

with st.expander("2022 — Guerra na Ucrânia, política de combustíveis e eleições presidenciais"):
    st.markdown("""
    **Guerra Rússia-Ucrânia**
    - Investigar os impactos da guerra no mercado internacional de petróleo e nos preços dos combustíveis.

    **Política de combustíveis no Brasil**
    - Pesquisar as mudanças na tributação dos combustíveis, incluindo as alterações no ICMS.
    - Investigar as decisões e mudanças na política de preços dos combustíveis durante o ano.

    **Eleições presidenciais de 2022**
    - Investigar o clima político e eleitoral durante o período de campanha.
    - Pesquisar como o preço da gasolina e do diesel foi abordado pelos candidatos e nos debates eleitorais.
    - Verificar se anúncios, propostas e medidas relacionadas aos combustíveis coincidiram com períodos de anomalias identificados pelo modelo.
    - Observar a repercussão do preço dos combustíveis na opinião pública e no debate político.
    """)


st.divider()


# ============================================================
# TABELA COMPLETA DE ANOMALIAS
# ============================================================

st.header("Consulta de todas as anomalias")

st.write(
    "Utilize os filtros para consultar os pontos identificados pelo modelo."
)

col1, col2 = st.columns(2)

with col1:
    combustiveis_disponiveis = [
        "Todos"
    ] + sorted(anomalias["Combustivel"].dropna().unique().tolist())

    filtro_combustivel = st.selectbox(
        "Combustível",
        combustiveis_disponiveis
    )

with col2:
    anos_disponiveis = sorted(
        anomalias["Data da Coleta"].dt.year.unique().tolist(),
        reverse=True
    )

    filtro_ano = st.selectbox(
        "Ano",
        ["Todos"] + anos_disponiveis
    )


anomalias_filtradas = anomalias.copy()

if filtro_combustivel != "Todos":
    anomalias_filtradas = anomalias_filtradas[
        anomalias_filtradas["Combustivel"] == filtro_combustivel
    ]

if filtro_ano != "Todos":
    anomalias_filtradas = anomalias_filtradas[
        anomalias_filtradas["Data da Coleta"].dt.year == filtro_ano
    ]


st.caption(
    f"{len(anomalias_filtradas)} anomalias encontradas com os filtros selecionados."
)

st.dataframe(
    anomalias_filtradas[
        [
            "Data da Coleta",
            "Combustivel",
            "Valor de Venda",
            "Variacao",
            "Desvio_Media_4",
            "Volatilidade_4"
        ]
    ].sort_values(
        "Data da Coleta",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    """
    Fonte dos dados: Agência Nacional do Petróleo, Gás Natural e
    Biocombustíveis (ANP).

    Identificação de anomalias: algoritmo Isolation Forest.

    Os pontos identificados representam comportamentos estatisticamente
    atípicos em relação às características utilizadas pelo modelo.
    """
)
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

st.set_page_config(
    page_title="Análise histórica dos preços dos combustíveis",
    page_icon="⛽",
    layout="wide",
)

PASTA_DADOS = Path("dados_processados")
ARQUIVO_DADOS = PASTA_DADOS / "dados_ml.csv"
ARQUIVO_ANOMALIAS = PASTA_DADOS / "anomalias.csv"
ARQUIVO_CONTEXTOS = PASTA_DADOS / "contextos_historicos.csv"

CORES = {
    "Gasolina": "#2878B5",
    "Etanol": "#E58A27",
}

NOMES_MESES = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr",
    5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def localizar_coluna(df, possibilidades, obrigatoria=True):
    """Localiza uma coluna aceitando variações simples de nome."""
    mapa = {str(col).strip().lower(): col for col in df.columns}

    for nome in possibilidades:
        encontrado = mapa.get(nome.strip().lower())
        if encontrado is not None:
            return encontrado

    if obrigatoria:
        raise ValueError(
            "Não foi possível localizar nenhuma destas colunas: "
            + ", ".join(possibilidades)
            + f". Colunas encontradas: {list(df.columns)}"
        )
    return None


def padronizar_combustivel(valor):
    texto = str(valor).strip().lower()
    if "gasolina" in texto:
        return "Gasolina"
    if "etanol" in texto:
        return "Etanol"
    return str(valor).strip()


def converter_datas(coluna):
    """Reconhece datas ISO (ano-mês-dia) e brasileiras (dia/mês/ano)."""
    texto = coluna.astype("string").str.strip()
    formato_iso = texto.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", na=False)
    resultado = pd.Series(pd.NaT, index=coluna.index, dtype="datetime64[ns]")
    resultado.loc[formato_iso] = pd.to_datetime(
        texto.loc[formato_iso], yearfirst=True, errors="coerce"
    )
    resultado.loc[~formato_iso] = pd.to_datetime(
        texto.loc[~formato_iso], dayfirst=True, errors="coerce"
    )
    return resultado


def ler_csv(caminho):
    """Lê CSV tentando formatos comuns de separador e codificação."""
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    ultimo_erro = None
    for separador in (",", ";"):
        try:
            df = pd.read_csv(caminho, sep=separador, encoding="utf-8-sig")
            if len(df.columns) > 1:
                return df
        except Exception as erro:
            ultimo_erro = erro

    try:
        return pd.read_csv(caminho, sep=None, engine="python", encoding="latin1")
    except Exception as erro:
        ultimo_erro = erro

    raise ValueError(f"Não foi possível ler {caminho}: {ultimo_erro}")


def preparar_dados():
    dados = ler_csv(ARQUIVO_DADOS)
    anomalias = ler_csv(ARQUIVO_ANOMALIAS)

    # Padroniza nomes de colunas mais comuns nos arquivos do projeto.
    col_data = localizar_coluna(dados, ["Data", "Data da Coleta", "data", "Semana"])
    col_comb = localizar_coluna(dados, ["Combustivel", "Combustível", "Produto"])
    col_preco = localizar_coluna(
        dados, ["Valor de Venda", "Preco", "Preço", "Media", "Mediana"]
    )

    dados = dados.rename(
        columns={col_data: "Data", col_comb: "Combustivel", col_preco: "Preco"}
    )
    dados["Data"] = converter_datas(dados["Data"])
    dados["Combustivel"] = dados["Combustivel"].map(padronizar_combustivel)
    dados["Preco"] = pd.to_numeric(
        dados["Preco"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    # Mantém somente os dois combustíveis do estudo.
    dados = dados[dados["Combustivel"].isin(["Gasolina", "Etanol"])].copy()
    dados = dados.dropna(subset=["Data", "Preco"]).sort_values("Data")

    # Padroniza a tabela de anomalias.
    an_data = localizar_coluna(anomalias, ["Data", "Data da Coleta", "Semana"])
    an_comb = localizar_coluna(anomalias, ["Combustivel", "Combustível", "Produto"])
    anomalias = anomalias.rename(
        columns={an_data: "Data", an_comb: "Combustivel"}
    )
    anomalias["Data"] = converter_datas(anomalias["Data"])
    anomalias["Combustivel"] = anomalias["Combustivel"].map(padronizar_combustivel)
    anomalias = anomalias[
        anomalias["Combustivel"].isin(["Gasolina", "Etanol"])
    ].dropna(subset=["Data"]).copy()

    # Uniformiza as colunas numéricas da tabela de anomalias, quando disponíveis.
    aliases = {
        "Variacao": ["Variacao", "Variação"],
        "Desvio_Media_4": ["Desvio_Media_4", "Desvio da Média 4", "Desvio_Media"],
        "Volatilidade_4": ["Volatilidade_4", "Volatilidade"],
        "Preco": ["Preco", "Preço", "Valor de Venda", "Media", "Mediana"],
    }
    for destino, opcoes in aliases.items():
        origem = localizar_coluna(anomalias, opcoes, obrigatoria=False)
        if origem and origem != destino:
            anomalias = anomalias.rename(columns={origem: destino})

    for coluna in ["Variacao", "Desvio_Media_4", "Volatilidade_4", "Preco"]:
        if coluna in anomalias.columns:
            anomalias[coluna] = pd.to_numeric(
                anomalias[coluna].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    # Se a tabela de anomalias não trouxer o preço, busca no conjunto principal.
    if "Preco" not in anomalias.columns:
        anomalias = anomalias.merge(
            dados[["Data", "Combustivel", "Preco"]],
            on=["Data", "Combustivel"],
            how="left",
        )

    # Quantidade de coletas pode existir no arquivo de dados.
    quantidade_col = localizar_coluna(
        dados, ["Quantidade", "N", "Contagem"], obrigatoria=False
    )
    if quantidade_col and quantidade_col != "Quantidade":
        dados = dados.rename(columns={quantidade_col: "Quantidade"})

    return dados, anomalias


@st.cache_data
def carregar_dados():
    return preparar_dados()


@st.cache_data
def carregar_contextos():
    if not ARQUIVO_CONTEXTOS.exists():
        return pd.DataFrame(
            columns=[
                "data_inicio", "data_fim", "titulo", "categoria",
                "descricao", "relacao", "fonte_1", "fonte_2", "fonte_3",
            ]
        )

    contextos = ler_csv(ARQUIVO_CONTEXTOS)
    obrigatorias = [
        "data_inicio", "data_fim", "titulo", "categoria",
        "descricao", "relacao",
    ]
    for coluna in obrigatorias:
        if coluna not in contextos.columns:
            contextos[coluna] = ""

    contextos["data_inicio"] = pd.to_datetime(
        contextos["data_inicio"], errors="coerce"
    )
    contextos["data_fim"] = pd.to_datetime(
        contextos["data_fim"], errors="coerce"
    )
    for coluna in [
        "titulo", "categoria", "descricao", "relacao",
        "fonte_1", "fonte_2", "fonte_3",
    ]:
        if coluna not in contextos.columns:
            contextos[coluna] = ""
        contextos[coluna] = contextos[coluna].fillna("").astype(str)

    return contextos.dropna(subset=["data_inicio", "data_fim"]).sort_values(
        "data_inicio", ascending=False
    )


def formatar_data(data):
    if pd.isna(data):
        return "Data não disponível"
    return pd.Timestamp(data).strftime("%d/%m/%Y")


def formatar_preco(valor):
    if pd.isna(valor):
        return "—"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_percentual(valor):
    if pd.isna(valor):
        return "—"
    return f"{valor:+.2f}%".replace(".", ",")


def grafico_linha(df, titulo, mostrar_anomalias=False, altura=430):
    fig = go.Figure()

    for combustivel in ["Gasolina", "Etanol"]:
        parte = df[df["Combustivel"] == combustivel].sort_values("Data")
        if parte.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=parte["Data"],
                y=parte["Preco"],
                mode="lines",
                name=combustivel,
                line=dict(color=CORES[combustivel], width=2),
                hovertemplate=(
                    "%{x|%d/%m/%Y}<br>"
                    + combustivel
                    + ": R$ %{y:.2f}<extra></extra>"
                ),
            )
        )

        if mostrar_anomalias:
            datas_anomalias = anomalias_filtradas[
                anomalias_filtradas["Combustivel"] == combustivel
            ]["Data"]
            pontos = parte[parte["Data"].isin(datas_anomalias)]
            if not pontos.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pontos["Data"],
                        y=pontos["Preco"],
                        mode="markers",
                        name=f"Anomalias — {combustivel}",
                        marker=dict(
                            color=CORES[combustivel],
                            size=9,
                            symbol="circle-open",
                            line=dict(width=2),
                        ),
                        hovertemplate=(
                            "%{x|%d/%m/%Y}<br>"
                            + combustivel
                            + " — anomalia: R$ %{y:.2f}<extra></extra>"
                        ),
                    )
                )

    fig.update_layout(
        title=titulo,
        height=altura,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=15, r=15, t=65, b=15),
        xaxis=dict(
            title="Semana",
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(title="Preço mediano semanal (R$/litro)", tickprefix="R$ "),
    )
    return fig


def exibir_metricas(dados, anomalias):
    total_semanas = dados.groupby("Combustivel")["Data"].nunique()
    total_anomalias = len(anomalias)
    datas_por_combustivel = anomalias.groupby("Combustivel")["Data"].apply(set)
    datas_gasolina = datas_por_combustivel.get("Gasolina", set())
    datas_etanol = datas_por_combustivel.get("Etanol", set())
    datas_compartilhadas = datas_gasolina.intersection(datas_etanol)

    inicio = dados["Data"].min()
    fim = dados["Data"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Período analisado", f"{inicio:%Y} – {fim:%Y}")
    c2.metric("Observações semanais", f"{int(total_semanas.sum()):,}".replace(",", "."))
    c3.metric("Anomalias identificadas", f"{total_anomalias}")
    c4.metric("Semanas com anomalia nos dois", f"{len(datas_compartilhadas)}")

    return total_semanas, datas_compartilhadas


# ============================================================
# CARREGAMENTO E VERIFICAÇÃO
# ============================================================

try:
    dados, anomalias = carregar_dados()
    contextos = carregar_contextos()
except Exception as erro:
    st.error(f"Não foi possível carregar os arquivos do projeto: {erro}")
    st.markdown(
        """
        Confira se estes arquivos estão na pasta `dados_processados/`:

        - `dados_ml.csv`
        - `anomalias.csv`
        - `contextos_historicos.csv` (pode estar vazio, mas deve ter os cabeçalhos)

        A estrutura esperada dos arquivos de dados inclui data, combustível e preço.
        """
    )
    st.stop()

if dados.empty:
    st.warning("O arquivo de dados não contém registros válidos de gasolina ou etanol.")
    st.stop()

# Usada pela função de gráficos para desenhar os marcadores de anomalia.
anomalias_filtradas = anomalias.copy()


# ============================================================
# CABEÇALHO / NAVEGAÇÃO
# ============================================================

st.title("Análise histórica dos preços da gasolina e do etanol no Brasil")
st.caption(
    "Projeto Integrador IV · UNIVESP · 2º semestre de 2026"
)

# Menu compacto com links para as seções da página.
st.markdown(
    """
    <style>
      /* Espaçamento do cabeçalho */
      .block-container {
        padding-top: 4.25rem;
      }

      /* Mantém o menu visível durante a rolagem */
      div[data-testid="stElementContainer"]:has(.menu-secoes),
      div.element-container:has(.menu-secoes) {
        position: sticky;
        top: 3.75rem;
        z-index: 1000;
        background-color: light-dark(#FFFFFF, #0E1117);
        padding: 0.5rem 0;
      }

      /* Menu horizontal */
      .menu-secoes {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        padding: 0.45rem 0;
        margin: 0;

        border-bottom: 1px solid light-dark(
            rgba(0, 0, 0, 0.10),
            rgba(255, 255, 255, 0.12)
        );
      }

      /* Aparência das opções */
      .menu-secoes a {
        display: inline-flex;
        align-items: center;
        padding: 0.55rem 0.85rem;
        border-radius: 6px;

        color: var(--text-color) !important;
        font-size: 0.92rem;
        font-weight: 500;
        text-decoration: none !important;
        white-space: nowrap;

        transition: background-color 0.2s ease,
                    color 0.2s ease;
      }

      /* Interação */
      .menu-secoes a:hover {
        background-color: light-dark(
            rgba(0, 0, 0, 0.05),
            rgba(255, 255, 255, 0.08)
        );

        color: var(--st-primary-color, #2878B5) !important;
      }

      .menu-secoes a:focus-visible {
        outline: 2px solid var(--primary-color);
        outline-offset: 4px;
      }

      /* Evita esconder os títulos após navegar */
      html {
        scroll-padding-top: 9rem;
      }

      .secao-ancora {
        scroll-margin-top: 9rem;
      }

      @media (max-width: 640px) {
        .block-container {
          padding-top: 4.5rem;
        }

        html {
          scroll-padding-top: 11.5rem;
        }

        .secao-ancora {
          scroll-margin-top: 11.5rem;
        }
      }
    </style>

    <nav class="menu-secoes" aria-label="Navegação">
      <a href="#apresentacao">Apresentação</a>
      <a href="#visao-geral">Visão geral</a>
      <a href="#comparacao">Comparação</a>
      <a href="#anomalias">Anomalias</a>
      <a href="#contexto-historico">Contexto histórico</a>
    </nav>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# ETAPA 1 — APRESENTAÇÃO DO PROJETO
# ============================================================

st.markdown('<div id="apresentacao" class="secao-ancora"></div>', unsafe_allow_html=True)
st.subheader("Sobre o projeto")
st.write(
    """
    Este projeto analisa o comportamento histórico dos preços da gasolina
    e do etanol no Brasil. A partir de dados públicos da Agência Nacional
    do Petróleo, Gás Natural e Biocombustíveis (ANP), são observadas as
    variações dos preços ao longo do tempo e identificados períodos que
    apresentam comportamento diferente do padrão.
    """
)

st.subheader("Problema de pesquisa")
st.write(
    """
    Como a análise de dados e técnicas de aprendizagem de máquina podem
    ajudar a identificar padrões e períodos de comportamento atípico nos
    preços da gasolina e do etanol no Brasil?
    """
)

st.subheader("Objetivo")
st.write(
    """
    Identificar padrões de variação nos preços dos combustíveis, detectar
    períodos atípicos por meio de aprendizagem de máquina e investigar
    acontecimentos nacionais e internacionais que possam ajudar a
    contextualizar os resultados.
    """
)

st.subheader("Fonte dos dados")
st.write(
    "Os dados utilizados foram disponibilizados pela Agência Nacional do "
    "Petróleo, Gás Natural e Biocombustíveis (ANP)."
)
st.markdown(
    "[Série histórica de preços de combustíveis e de GLP — ANP](https://dados.gov.br/dados/conjuntos-dados/serie-historica-de-precos-de-combustiveis-e-de-glp)"
)


# ============================================================
# ETAPA 2 — VISÃO GERAL DOS DADOS
# ============================================================

st.divider()
st.markdown('<div id="visao-geral" class="secao-ancora"></div>', unsafe_allow_html=True)
st.subheader("Visão geral dos dados")
st.write(
    "Um resumo do conjunto de dados utilizado na análise e dos resultados "
    "gerais da identificação de anomalias."
)

total_semanas, datas_compartilhadas = exibir_metricas(dados, anomalias)

st.markdown("#### Dados por combustível")
col1, col2 = st.columns(2)

for coluna, combustivel in [(col1, "Gasolina"), (col2, "Etanol")]:
    parte = dados[dados["Combustivel"] == combustivel]
    parte_anomalias = anomalias[
        anomalias["Combustivel"] == combustivel
    ]
    with coluna:
        st.markdown(f"**{combustivel}**")
        m1, m2 = st.columns(2)
        m1.metric("Observações semanais", parte["Data"].nunique())
        m2.metric("Anomalias", len(parte_anomalias))

with st.expander("Sobre os dados e a metodologia"):
    st.markdown(
        """
        **Origem e organização**

        Os arquivos originais da ANP contêm registros de preços coletados
        em diferentes revendedores e municípios. Para esta análise, os
        registros foram organizados por combustível e por semana.

        **Preço mediano semanal**

        O valor apresentado representa a mediana dos preços registrados
        naquela semana para o combustível analisado. A mediana é o valor
        central após ordenar os preços e tende a ser menos influenciada
        por valores muito altos ou muito baixos do que a média.

        Portanto, o resultado é um resumo dos preços coletados pela ANP,
        e não o preço praticado em todos os postos do Brasil.

        **Preparação para aprendizagem de máquina**

        Para identificar comportamentos atípicos, foram calculadas três
        características para cada combustível:

        - **Variação semanal:** mudança percentual em relação à semana anterior.
        - **Desvio da média móvel:** diferença entre o preço da semana e a
          média das quatro semanas consideradas.
        - **Volatilidade:** dispersão das variações observadas no período
          móvel de quatro semanas.

        Essas características são utilizadas pelo algoritmo de detecção
        de anomalias apresentado na sessão de Anomalias.
        """
    )


# ============================================================
# ETAPA 3 — COMPARAÇÃO DOS COMBUSTÍVEIS
# ============================================================

st.divider()
st.markdown('<div id="comparacao" class="secao-ancora"></div>', unsafe_allow_html=True)
st.subheader("Comparação dos preços")
st.write(
    "Explore a evolução dos preços medianos semanais da gasolina e do "
    "etanol. Use a barra inferior do gráfico principal para selecionar um período."
)

st.plotly_chart(
    grafico_linha(
        dados,
        "Evolução dos preços da gasolina e do etanol",
        mostrar_anomalias=True,
        altura=470,
    ),
    use_container_width=True,
)

st.markdown("#### Evolução individual")
col_gasolina, col_etanol = st.columns(2, gap="medium")
for coluna, combustivel in [
    (col_gasolina, "Gasolina"),
    (col_etanol, "Etanol"),
]:
    parte = dados[dados["Combustivel"] == combustivel]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=parte["Data"],
            y=parte["Preco"],
            mode="lines",
            name=combustivel,
            line=dict(color=CORES[combustivel], width=1.8),
            hovertemplate=(
                "%{x|%d/%m/%Y}<br>Preço mediano: R$ %{y:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(text=f"{combustivel}", font=dict(size=15)),
        height=270,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=38, b=10),
        xaxis=dict(title=None, type="date", rangeslider=dict(visible=False)),
        yaxis=dict(title="R$/litro", tickprefix="R$ "),
        showlegend=False,
    )
    with coluna:
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Como interpretar os gráficos?"):
    st.markdown(
        """
        - Cada ponto da linha representa a mediana dos preços registrados
          naquela semana.
        - As linhas mostram a evolução dos preços ao longo do período.
        - Os círculos destacados indicam semanas classificadas como
          anomalias pelo modelo Isolation Forest.
        - Uma anomalia é um comportamento estatisticamente diferente do
          padrão identificado pelo modelo. Isso não significa,
          necessariamente, erro nos dados ou aumento do preço.
        """
    )


# ============================================================
# ETAPA 4 — DETECÇÃO DE ANOMALIAS
# ============================================================

st.divider()
st.markdown('<div id="anomalias" class="secao-ancora"></div>', unsafe_allow_html=True)
st.subheader("Detecção de anomalias")
st.write(
    """
    O projeto utiliza o algoritmo **Isolation Forest**, uma técnica de
    aprendizagem de máquina não supervisionada. Ele procura observações
    que se diferenciam do comportamento mais comum dos dados, sem precisar
    receber previamente uma lista de exemplos classificados.
    """
)

with st.expander("Como o modelo identifica um comportamento atípico?"):
    st.markdown(
        """
        A análise é executada separadamente para gasolina e etanol, pois
        cada combustível pode apresentar padrões e amplitudes de variação
        diferentes.

        O modelo considera três características calculadas para cada
        semana:

        - **Variação percentual:** quanto o preço mudou em relação à semana anterior.
        - **Desvio da média móvel de quatro semanas:** quanto o preço se
          afastou da média recente.
        - **Volatilidade de quatro semanas:** o quanto as variações recentes
          oscilaram.

        A configuração `contamination=0.05` foi utilizada como estimativa
        de 5% de observações atípicas entre os registros entregues ao
        modelo para cada combustível. Como algumas semanas iniciais não
        possuem todas as características móveis calculadas, essa
        proporção não deve ser interpretada como exatamente 5% de todas
        as semanas do arquivo original.
        """
    )

st.markdown("#### Explorar anomalias")
opcoes_combustivel = ["Ambos", "Gasolina", "Etanol"]
filtro_combustivel = st.selectbox(
    "Combustível",
    opcoes_combustivel,
    key="filtro_anomalia_combustivel",
)

anos_disponiveis = sorted(anomalias["Data"].dt.year.dropna().unique().tolist())
opcoes_ano = ["Todos"] + [str(int(ano)) for ano in anos_disponiveis]
filtro_ano = st.selectbox(
    "Ano",
    opcoes_ano,
    key="filtro_anomalia_ano",
)

apenas_compartilhadas = st.checkbox(
    "Mostrar somente semanas em que os dois combustíveis foram classificados como anomalia"
)

anomalias_exibidas = anomalias.copy()
if filtro_combustivel != "Ambos":
    anomalias_exibidas = anomalias_exibidas[
        anomalias_exibidas["Combustivel"] == filtro_combustivel
    ]
if filtro_ano != "Todos":
    anomalias_exibidas = anomalias_exibidas[
        anomalias_exibidas["Data"].dt.year == int(filtro_ano)
    ]
if apenas_compartilhadas:
    datas_duplas = (
        anomalias.groupby("Data")["Combustivel"].nunique()
    )
    datas_duplas = datas_duplas[datas_duplas == 2].index
    anomalias_exibidas = anomalias_exibidas[
        anomalias_exibidas["Data"].isin(datas_duplas)
    ]

colunas_tabela = [
    coluna for coluna in [
        "Data", "Combustivel", "Preco", "Variacao",
        "Desvio_Media_4", "Volatilidade_4",
    ]
    if coluna in anomalias_exibidas.columns
]
tabela = anomalias_exibidas[colunas_tabela].copy()
if "Data" in tabela.columns:
    tabela["Data"] = tabela["Data"].dt.strftime("%d/%m/%Y")
tabela = tabela.rename(
    columns={
        "Data": "Semana",
        "Combustivel": "Combustível",
        "Preco": "Preço mediano (R$/L)",
        "Variacao": "Variação semanal (%)",
        "Desvio_Media_4": "Desvio da média móvel",
        "Volatilidade_4": "Volatilidade (4 semanas)",
    }
)
st.caption(f"{len(anomalias_exibidas)} ocorrência(s) encontrada(s).")
st.dataframe(tabela, use_container_width=True, hide_index=True)

st.divider()
st.markdown("#### Semanas com anomalias nos dois combustíveis")
st.write(
    """
    Esta comparação destaca as semanas em que gasolina e etanol foram
    classificados como atípicos ao mesmo tempo. A coincidência pode ajudar
    a direcionar a pesquisa de acontecimentos do período, mas não
    demonstra que os combustíveis tiveram uma causa comum.
    """
)

gas = anomalias[anomalias["Combustivel"] == "Gasolina"].copy()
eta = anomalias[anomalias["Combustivel"] == "Etanol"].copy()
compartilhadas = sorted(set(gas["Data"]).intersection(set(eta["Data"])))

if not compartilhadas:
    st.info("Não foram encontradas semanas simultâneas nos arquivos atuais.")
else:
    comparacao = pd.DataFrame({"Data": compartilhadas})
    colunas_gas = ["Data"] + [
        c for c in ["Preco", "Variacao"] if c in gas.columns
    ]
    colunas_eta = ["Data"] + [
        c for c in ["Preco", "Variacao"] if c in eta.columns
    ]
    gas = gas[colunas_gas].rename(
        columns={
            "Preco": "Preço gasolina (R$/L)",
            "Variacao": "Variação gasolina (%)",
        }
    )
    eta = eta[colunas_eta].rename(
        columns={
            "Preco": "Preço etanol (R$/L)",
            "Variacao": "Variação etanol (%)",
        }
    )
    comparacao = comparacao.merge(gas, on="Data", how="left").merge(
        eta, on="Data", how="left"
    )

    anos_compartilhados = sorted(
        pd.Series(compartilhadas).dt.year.unique().tolist()
    )
    anos_comp_opcoes = ["Todos"] + [str(int(a)) for a in anos_compartilhados]
    ano_comp = st.selectbox(
        "Filtrar período das coincidências",
        anos_comp_opcoes,
        key="ano_compartilhadas",
    )
    if ano_comp != "Todos":
        comparacao = comparacao[
            comparacao["Data"].dt.year == int(ano_comp)
        ]

    tabela_comp = comparacao.copy()
    tabela_comp["Data"] = tabela_comp["Data"].dt.strftime("%d/%m/%Y")
    st.dataframe(tabela_comp, use_container_width=True, hide_index=True)

    # Gráfico dos preços somente nas semanas compartilhadas.
    dados_comp = dados[dados["Data"].isin(comparacao["Data"].apply(
        lambda x: pd.to_datetime(x, dayfirst=True) if isinstance(x, str) else x
    ))]
    if not dados_comp.empty:
        fig_comp = go.Figure()
        for combustivel in ["Gasolina", "Etanol"]:
            parte = dados_comp[
                dados_comp["Combustivel"] == combustivel
            ].sort_values("Data")
            fig_comp.add_trace(
                go.Scatter(
                    x=parte["Data"],
                    y=parte["Preco"],
                    mode="lines+markers",
                    name=combustivel,
                    line=dict(color=CORES[combustivel], width=2),
                    marker=dict(size=8),
                    hovertemplate=(
                        "%{x|%d/%m/%Y}<br>"
                        + combustivel
                        + ": R$ %{y:.2f}<extra></extra>"
                    ),
                )
            )
        fig_comp.update_layout(
            title="Preços nas semanas com anomalias simultâneas",
            height=400,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Semana",
            yaxis_title="Preço mediano semanal (R$/litro)",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

with st.expander("Limitações da análise"):
    st.markdown(
        """
        - O modelo identifica padrões estatisticamente incomuns, mas não
          explica por que eles aconteceram.
        - Uma anomalia não representa necessariamente um erro, uma crise
          ou uma variação negativa.
        - O modelo não realiza previsão de preços.
        - As coincidências entre combustíveis indicam ocorrência na mesma
          semana, não uma relação de causa e efeito.
        """
    )



# ============================================================
# ETAPA 5 — CONTEXTO HISTÓRICO
# ============================================================

st.divider()
st.markdown(
    '<div id="contexto-historico" class="secao-ancora"></div>',
    unsafe_allow_html=True
)

st.subheader("Contexto histórico")

st.write(
    """
    Esta seção reúne acontecimentos nacionais e internacionais
    que podem ajudar a compreender a evolução dos preços da
    gasolina e do etanol no Brasil. A associação com as anomalias
    é feita pela proximidade das datas, sem comprovar causalidade.
    """
)

# ------------------------------------------------------------
# EXPLICAÇÃO SOBRE AS LIMITAÇÕES
# ------------------------------------------------------------

with st.expander(
    "Por que alguns acontecimentos não apresentam anomalias?"
):
    st.write(
        """
        Nem todo acontecimento importante para o mercado de
        combustíveis provoca uma mudança considerada atípica
        pelo modelo.

        Isso pode acontecer porque:

        - **Efeitos graduais:** algumas mudanças nos preços
          acontecem ao longo de várias semanas, sem grandes
          variações isoladas.

        - **Atraso nos repasses:** alterações nos preços do
          petróleo, nas refinarias ou nos impostos podem
          levar algum tempo para chegar aos postos.

        - **Comportamento estatístico:** uma variação pode ser
          economicamente relevante, mas ainda estar dentro
          dos padrões identificados pelo Isolation Forest.

        - **Limitações da análise:** o modelo utiliza dados
          semanais agregados e não identifica diretamente
          as causas das mudanças.

        Da mesma forma, uma anomalia pode ser identificada
        mesmo quando não encontramos um acontecimento
        histórico que ajude a explicá-la.

        Portanto, a ausência de uma anomalia não significa
        que o acontecimento não teve impacto nos preços.
        """
    )


# ------------------------------------------------------------
# FILTROS
# ------------------------------------------------------------

if contextos.empty:
    st.info(
        "A pesquisa de contextos históricos ainda está em andamento. "
        "Os acontecimentos aparecerão aqui quando forem adicionados "
        "ao arquivo `contextos_historicos.csv`."
    )

else:
    categorias = sorted(
        c for c in contextos["categoria"].unique()
        if c.strip()
    )

    anos_contexto = sorted(
        set(contextos["data_inicio"].dt.year.dropna())
        | set(contextos["data_fim"].dt.year.dropna()),
        reverse=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        ano_contexto = st.selectbox(
            "Ano",
            ["Todos"] + [str(int(a)) for a in anos_contexto],
            key="contexto_ano"
        )

    with f2:
        categoria_contexto = st.selectbox(
            "Categoria",
            ["Todas"] + categorias,
            key="contexto_categoria"
        )

    with f3:
        combustivel_contexto = st.selectbox(
            "Combustível relacionado",
            ["Ambos", "Gasolina", "Etanol"],
            key="contexto_combustivel"
        )

    tipo_contexto = st.radio(
        "Exibir acontecimentos",
        ["Todos", "Com anomalias", "Sem anomalias"],
        horizontal=True,
        key="contexto_tipo"
    )

    # --------------------------------------------------------
    # APLICAÇÃO DOS FILTROS
    # --------------------------------------------------------

    contextos_exibidos = contextos.copy()

    if ano_contexto != "Todos":
        ano_escolhido = int(ano_contexto)

        contextos_exibidos = contextos_exibidos[
            (
                contextos_exibidos["data_inicio"].dt.year
                <= ano_escolhido
            )
            &
            (
                contextos_exibidos["data_fim"].dt.year
                >= ano_escolhido
            )
        ]

    if categoria_contexto != "Todas":
        contextos_exibidos = contextos_exibidos[
            contextos_exibidos["categoria"]
            == categoria_contexto
        ]

    # --------------------------------------------------------
    # CLASSIFICAÇÃO DOS ACONTECIMENTOS
    # --------------------------------------------------------

    eventos_com_anomalias = []
    eventos_sem_anomalias = []

    for _, evento in contextos_exibidos.iterrows():
        inicio = evento["data_inicio"]
        fim = evento["data_fim"]

        # Procura anomalias no intervalo do acontecimento.
        relacionadas = anomalias[
            (anomalias["Data"] >= inicio)
            & (anomalias["Data"] <= fim)
        ].copy()

        # Considera o combustível selecionado.
        if combustivel_contexto != "Ambos":
            relacionadas = relacionadas[
                relacionadas["Combustivel"]
                == combustivel_contexto
            ]

        if relacionadas.empty:
            eventos_sem_anomalias.append(
                (evento, relacionadas)
            )
        else:
            eventos_com_anomalias.append(
                (evento, relacionadas)
            )

    # --------------------------------------------------------
    # FUNÇÃO PARA EXIBIR CADA ACONTECIMENTO
    # --------------------------------------------------------

    def exibir_evento(evento, relacionadas):
        inicio = evento["data_inicio"]
        fim = evento["data_fim"]
        titulo = evento["titulo"].strip() or "Contexto sem título"

        with st.container(border=True):
            st.markdown(f"#### {titulo}")

            meta = (
                f"{formatar_data(inicio)} "
                f"a {formatar_data(fim)}"
            )

            if evento["categoria"].strip():
                meta += f" · {evento['categoria'].strip()}"

            st.caption(meta)

            # Descrição do acontecimento
            if evento["descricao"].strip():
                st.write(evento["descricao"].replace("$", r"\$"))

            # Possível relação com os preços
            if evento["relacao"].strip():
                st.markdown("**Possível relação com os preços**")
                st.write(evento["relacao"].replace("$", r"\$"))

            # Anomalias identificadas
            if not relacionadas.empty:
                st.markdown("**Anomalias no período**")

                por_combustivel = (
                    relacionadas.groupby("Combustivel")["Data"]
                    .apply(
                        lambda datas: sorted(set(datas))
                    )
                    .to_dict()
                )

                for combustivel in ["Gasolina", "Etanol"]:
                    datas = por_combustivel.get(
                        combustivel, []
                    )

                    if datas:
                        datas_texto = ", ".join(
                            formatar_data(data)
                            for data in datas
                        )

                        st.markdown(
                            f"- **{combustivel}:** {datas_texto}"
                        )

            else:
                st.caption(
                    "Nenhuma anomalia identificada neste "
                    "intervalo para o combustível selecionado."
                )

            # Fontes consultadas
            fontes = [
                evento.get("fonte_1", ""),
                evento.get("fonte_2", ""),
                evento.get("fonte_3", "")
            ]

            fontes = [
                fonte.strip()
                for fonte in fontes
                if str(fonte).strip()
            ]

            if fontes:
                st.markdown("**Fontes consultadas**")

                for indice, fonte in enumerate(
                    fontes, start=1
                ):
                    st.markdown(
                        f"- [Fonte {indice}]({fonte})"
                    )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    total_com = len(eventos_com_anomalias)
    total_sem = len(eventos_sem_anomalias)

    st.caption(
        f"{total_com} acontecimento(s) com anomalias · "
        f"{total_sem} sem anomalias no intervalo selecionado"
    )

    # A classificação considera o combustível filtrado.
    if combustivel_contexto != "Ambos":
        st.caption(
            "A classificação considera apenas as anomalias "
            f"de {combustivel_contexto.lower()}."
        )

    # --------------------------------------------------------
    # GRUPO 1 — COM ANOMALIAS
    # --------------------------------------------------------

    if tipo_contexto in ["Todos", "Com anomalias"]:
        st.markdown(
            "### Acontecimentos com anomalias identificadas"
        )

        st.caption(
            "Acontecimentos que coincidem temporalmente "
            "com uma ou mais anomalias detectadas pelo modelo."
        )

        if not eventos_com_anomalias:
            st.info(
                "Nenhum acontecimento com anomalias foi "
                "encontrado para os filtros selecionados."
            )
        else:
            for evento, relacionadas in eventos_com_anomalias:
                exibir_evento(evento, relacionadas)

    # --------------------------------------------------------
    # GRUPO 2 — SEM ANOMALIAS
    # --------------------------------------------------------

    if tipo_contexto == "Todos":
        st.markdown("")

        # Mantém os acontecimentos secundários recolhidos.
        with st.expander(
            f"Outros acontecimentos relevantes ({total_sem})",
            expanded=False
        ):
            st.caption(
                "Acontecimentos sem anomalias identificadas "
                "no intervalo e no combustível selecionado. "
                "Isso não significa ausência de impacto "
                "sobre os preços."
            )

            if not eventos_sem_anomalias:
                st.info(
                    "Todos os acontecimentos encontrados "
                    "coincidem com pelo menos uma anomalia."
                )
            else:
                for evento, relacionadas in eventos_sem_anomalias:
                    exibir_evento(evento, relacionadas)

    elif tipo_contexto == "Sem anomalias":
        st.markdown(
            "### Acontecimentos sem anomalias identificadas"
        )

        st.caption(
            "Acontecimentos relevantes que não coincidiram "
            "com anomalias no intervalo e no combustível "
            "selecionado."
        )

        if not eventos_sem_anomalias:
            st.info(
                "Nenhum acontecimento sem anomalias foi "
                "encontrado para os filtros selecionados."
            )
        else:
            for evento, relacionadas in eventos_sem_anomalias:
                exibir_evento(evento, relacionadas)


# ------------------------------------------------------------
# OBSERVAÇÃO FINAL
# ------------------------------------------------------------

st.divider()

st.caption(
    "As associações apresentadas são baseadas na proximidade "
    "temporal entre acontecimentos históricos e anomalias. "
    "Não representam comprovação de causalidade. "
    "A ausência de anomalias também não significa "
    "ausência de efeitos econômicos."
)



# ============================================================
# RODAPÉ
# ============================================================

st.divider()
st.markdown(
    """
    **UNIVESP — Universidade Virtual do Estado de São Paulo**  
    Engenharia da Computação  
    Disciplina: Projeto Integrador IV  
    2º semestre de 2026
    """
)
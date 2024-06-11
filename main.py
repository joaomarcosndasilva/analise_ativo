import streamlit as st
import yfinance as yf
from time import sleep
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

icone_info = "ℹ️"
icone_warning = "⚠️"
icone_erro = "🚨"
icone_sucess = "✅"
titulo1 = 'Análise de ativos. V 1.0'
titulo2 = 'Esta aplicação analisa e preve o preço de fechamento do dia de qualquer ativo da B3'
comentario = 'As atualizações ocorrem frequentemente, então sempre entre para chegar o que há de mais atual :)'

acoes_ibov = ['RRRP3','ALOS3','ALPA4','ABEV3','ARZZ3','ASAI3', 'AZUL4','B3SA3','BBSE3','BBDC3','BBDC4','BRAP4','BBAS3',
'BRKM5', 'BRFS3', 'BPAC11','CRFB3','CCRO3','CMIG4','CIEL3','COGN3', 'CPLE6','CSAN3','CPFE3','CMIN3','CVCB3', 'CYRE3',
'DXCO3','ELET3','ELET6','EMBR3','ENGI11','ENEV3','EGIE3','EQTL3','EZTC3','FLRY3','GGBR4','GOAU4','NTCO3','SOMA3','HAPV3',
'HYPE3','IGTI11','IRBR3','ITSA4','ITUB4','JBSS3','KLBN11','RENT3','LREN3','LWSA3','MGLU3','MRFG3','BEEF3','MRVE3','MULT3',
'PCAR3','PETR3','PETR4','RECV3', 'PRIO3','PETZ3','RADL3','RAIZ4','RDOR3','RAIL3','SBSP3','SANB11','SMTO3','CSNA3','SLCE3',
'SUZB3','TAEE11','VIVT3','TIMS3','TOTS3','TRPL4','UGPA3','USIM5','VALE3','VAMO3','VBBR3','VIVA3','WEGE3','YDUQ3',]

acoes_ibov = sorted(acoes_ibov)

def graficos_analises():
    global df
    ticket = yf.Ticker(acao)
    df = ticket.history(period='5y')
    fig = px.line(df, df.index, df.Close)

    st.plotly_chart(fig)

    st.sidebar.info('GRÁFICOS', icon=icone_sucess)

    cb_volume = st.sidebar.checkbox('Gráfico de Volume')
    if cb_volume:
        with st.spinner('Aguarde...'):
            sleep(0.5)
        st.info('Gráfico de Volume', icon=icone_sucess)
        st.line_chart(df.Volume)

    cb_dividendos = st.sidebar.checkbox('Gráfico de dividendos')
    if cb_dividendos:
        with st.spinner('Aguarde...'):
            sleep(0.5)
        st.info('Gráfico de dividendos!', icon=icone_sucess)
        st.line_chart(df.Dividends)
    st.sidebar.warning('PREVER PREÇOS FUTUROS', icon=icone_warning)

    lr = st.sidebar.checkbox('Regressão Linear (LR)')
    if lr:
        with st.spinner('Aguarde...'):
            sleep(1)
        analisar_ativo(codigo_ativo=acao)
            #except:
             #   st.error('A Regressão Linear ainda não está funcionando, por favor, aguarde + alguns dias', icon=icone_erro)

    cb_neural = st.sidebar.checkbox('Rede Neural Artificial (RNA)')
    if cb_neural:
        with st.spinner('Aguarde...'):
            sleep(0.5)
        st.error('A (RN) Rede Neural, ainda não funciona. Aguarde o próximo commit :).', icon=icone_erro)

    pycaret = st.sidebar.checkbox('Vários Algorítimos (PYCARET)')
    if pycaret:
        with st.spinner('Aguarde...'):
            sleep(0.5)
        st.error('Ainda estou ajustando, talvez no pŕoximo commit já estará funcionando :).', icon=icone_erro)

    cb_fbprophet = st.sidebar.checkbox('Previsor da COVID (PROPHET)')
    if cb_fbprophet:
        with st.spinner('Aguarde...'):
            sleep(0.5)
        st.error('O PROPHET ainda não estão funcionando, por favor, aguarde + alguns dias', icon=icone_erro)

def analisar_ativo(codigo_ativo='CPLE6', periodo_analisado='9'):
    global figura, df
    import pandas as pd
    global df, lr, y_de_amanha, df_inicial, x_features, scaler, total, teste, treino, validacao, coeficiente, df2, ativo

    ativo, periodo = codigo_ativo, periodo_analisado,

    df_inicial = df[:]
    df = df_inicial[:]
    df = df.drop(['Dividends', 'Stock Splits'], axis=1)
    df['mm9'] = df['Close'].rolling(9).mean().round(2)
    df['mm21'] = df['Close'].rolling(21).mean().round(2)
    df_inicial = df[:]
    df['Close'] = df['Close'].shift(-1)
    df = df.dropna()

    total = len(df)
    total_inicial = len(df_inicial)

    treino = total - 700
    treino_inicial = total_inicial - 700

    teste = total - 15
    teste_inicial = total_inicial - 15

    ########################################################################################################################
    st.subheader('A SEPARAÇÃO DOS DADOS SEGUE A SEGUINTE DIVISÃO:')
    st.write(f'\nTreino das linhas 0 até {treino} - Teste da linha {treino} até {teste} - Validação da linha {teste} até {total}')
    #st.write(f'Treino 0:{treino_inicial} - Teste {treino_inicial}:{teste_inicial} - Validação {teste_inicial}:{total_inicial}')
    ########################################################################################################################

    df = df.reset_index()
    df_inicial = df_inicial.reset_index()

    x_features = df.drop(['Date', 'Close'], axis=1)
    x_features_inicial = df_inicial.drop(['Date', 'Close'], axis=1)

    x_features_list = list(x_features)

    y_labels = df['Close']
    y_labels_inicial = df_inicial['Close']

    from sklearn.feature_selection import SelectKBest

    k_best_features = SelectKBest(k='all')
    k_best_features.fit_transform(x_features, y_labels)
    k_best_features_score = k_best_features.scores_
    melhores = zip(x_features_list, k_best_features_score)
    melhores_ordenados = list(reversed(sorted(melhores, key=lambda x: x[1])))

    melhores_variaveis = dict(melhores_ordenados[:15])
    melhores_selecionadas = melhores_variaveis.keys()

    x_features = x_features.drop('Volume', axis=1)
    x_features_inicial = df_inicial.drop(['Date', 'Close', 'Volume'], axis=1)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_features)
    x_features_normalizado = scaler.fit_transform(x_features)

    x_features_inicial = x_features_inicial.dropna()
    scaler.fit(x_features_inicial)
    x_features_inicial_normalizado = scaler.fit_transform(x_features_inicial)

    x_features_normal = pd.DataFrame(x_features_normalizado, columns=list(x_features.columns))
    x_features_normal_inicial = pd.DataFrame(x_features_inicial_normalizado, columns=list(x_features_inicial.columns))

    x_train = x_features_normal[0:treino]
    x_train_inicial = x_features_inicial_normalizado[0:treino_inicial]

    x_test = x_features_normal[treino:teste]
    x_test_inicial = x_features_inicial_normalizado[treino_inicial:teste_inicial]

    y_train = y_labels[0:treino]
    y_train_inicial = y_labels[0:treino_inicial]

    y_test = y_labels[treino:teste]
    y_test_inicial = y_labels[treino_inicial:teste_inicial]

    st.subheader('Explicação do modelo de Regressão Linear:')
    st.write(f'\nO modelo aprenderá com os dados da linha 0 a {treino} das variáveis {list(x_features.columns)}')
    st.write(f'O modelo testará com os dados da linha {treino} a {teste} da variável Close')
    st.write('\nNa Segunda Parte: ')
    st.write(f'O modelo aprenderá com os dados da linha 0 a {treino_inicial} das variáveis {list(x_features_inicial.columns)}')
    st.write(f'O modelo testará com os dados da linha {treino_inicial} a {teste_inicial} da variável Close')

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_predito = lr.predict(x_test)

    # lr.fit()
    coeficiente = r2_score(y_test, y_predito)

    st.write(f'''O coeficiente é de {coeficiente * 100:.2f}%, isto é,  {coeficiente * 100:.2f}% das variações no valor dopreço futuro de
    Fechamento (Close) é explicada pela variação nas variávies {list(x_features.columns)} do dia anterior''')

    previsao = x_features_normal[teste:total]
    dia = df['Date'][teste:total]
    real = df['Close'][teste:total]

    previsao_hoje = x_features_normal_inicial[teste:total]
    dia_hoje = df_inicial['Date'][teste_inicial:total_inicial]
    real_hoje = df_inicial['Close'][teste_inicial:total_inicial]

    y_pred = lr.predict(previsao)
    y_de_amanha = lr.predict(x_features_inicial_normalizado)

    df2 = pd.DataFrame({'Data': dia, 'Cotacao': real, 'Previsto': y_pred})
    df2['Cotacao'] = df2['Cotacao'].shift(+1)


    figura, ax = plt.subplots(figsize=(16, 8))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_tick_params(rotation=30)

    ax.set_title(f'Teste de Previsão dos ultimos 15 pregões do ativo {ativo.replace(".SA", "")}\nCoeficiente R2 de {round(coeficiente * 100, 2)}% - By J. Brutus', fontsize=24)
    ax.set_ylabel('Preço do ativo em R$', fontsize=14)
    ax.plot(df2['Data'], df2['Cotacao'], marker='o', label='Cotação Real', color='blue')
    ax.plot(df2['Data'], df2['Previsto'], marker='o', label='Cotação Prevista', color='red')

    plt.grid()
    plt.legend()
    #st.pyplot(figura)

    rodar_nova()


def rodar_nova():
  global x_norm
  import pandas as pd
  import datetime

  df = df_inicial

  df = df.drop(['Date','Volume' ], axis=1) # vai até o 2237
  df = df.dropna()
  y = df['Close']
  x = df.drop('Close', axis=1)

  scaler.fit(x)
  x_norm = scaler.fit_transform(x)

  y_previsto = lr.predict(x_norm[-1:])
  hoje = datetime.date.today()

  previsao_hoje = pd.DataFrame({'Data':hoje, 'Preco Previsto': y_previsto})

  ###############################################################################



  figura, ax = plt.subplots(figsize=(16, 8))

  ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
  ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
  ax.xaxis.set_tick_params(rotation=30)

  ax.set_title(f'\n\nPrevisão do preço de fechamento para hoje, {hoje.strftime("%d/%m/%Y")} (EM VERDE), do ativo {ativo.replace(".SA", "")}\nCoeficiente R2 de {round(coeficiente * 100, 2)}% - By J. Brutus', fontsize=24)
  ax.set_ylabel('Preço do ativo em R$', fontsize=14)
  ax.plot(df2['Data'], df2['Cotacao'], marker='o', label='Cotação Real', color='blue')
  ax.plot(df2['Data'], df2['Previsto'], marker='o', label='Cotação Prevista', color='red')
  ax.plot(previsao_hoje['Data'], previsao_hoje['Preco Previsto'], marker='o', color='green', label='Previsão do fechamento')

  plt.grid()
  plt.legend()
  st.pyplot(figura)


  return

st.title(titulo1)
st.subheader(titulo2)
st.write(comentario)
st.write('by J. Brutus')
st.sidebar.success('ANÁLISE/ PREVISÃO DE ATIVOS', icon=icone_info)

select_modo = st.sidebar.radio("Selecione como você quer ver a análise", ("Lista de ativos", "Digitar o código"))

if select_modo == "Digitar o código":
    acao = st.sidebar.text_input('Digite o código do Ativo e selecione as datas!', 'VALE3', help='Digite o código do ativo sem o ".SA" e pressione ENTER. ')
    acao = f'{acao}.SA'
    if acao:
        try:
            graficos_analises()
        except:
            st.warning(f'Você digitou o ativo {acao}. e selecionou os períodos ')
            st.error("Alguma coisa não está certa. Tente alterar o período de datas")

elif select_modo == "Lista de ativos":
    papeis = acoes_ibov
    acao = st.sidebar.selectbox(f'Selecione um dos {len(papeis)} que compões o IBOV:', papeis, help=f"Está Lista contém {len(papeis)} ativos de ações que compõpes o índice do Ibovespa. Os ativos são açoes preferenciais,"
                                                                   f"ações ordinárias e unitis...")
    acao = f'{acao}.SA'
    graficos_analises()

else:
    st.info('Marque como você vai querer a análise')

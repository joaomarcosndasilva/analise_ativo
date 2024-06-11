import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def pagina_web():
    st.title('Análise de Ativos da B3')
    st.sidebar.title('Digite o código do ativo:')
    ativo = st.sidebar.text_input('Digite o código e aperte enter...', 'VALE3')
    ticket = yf.Ticker(f'{ativo}.SA')
    df = ticket.history(period='5y')

    fig = px.line(data_frame=df, x=df.index, y=df.Close)
    fig.update_layout()
    st.plotly_chart(fig)
    ############################################3
    dre = ticket.get_financials(freq='quarterly')
    dre = pd.DataFrame(dre)
    dre = dre / 1000
    dre = dre[dre.columns[::-1]]

    bal = ticket.get_balance_sheet(freq='yearly')
    bal = pd.DataFrame(bal)
    bal = bal/1000
    bal = bal[bal.columns[::-1]]


    fig2 = make_subplots(rows=2, cols=2, row_heights=[5,5], column_widths=[3,3], subplot_titles=('EBITIDA', 'Lucro Líquido', 'Dívida Líquida', 'Dívida Total'), shared_xaxes=False )
    #st.dataframe(dre)
    fig2.add_trace(go.Bar(x=dre.columns, y=dre.loc['EBITDA'], name='EBTIDA'), row=1, col=1)
    fig2.add_trace(go.Bar(x=dre.columns, y=dre.loc['NetIncome'], name='Lucro Líquido'), row=1, col=2)
    fig2.add_trace(go.Bar(x=bal.columns, y=bal.loc['NetDebt'], name='Dívida Liquida'), row=2, col=1)
    fig2.add_trace(go.Bar(x=bal.columns, y=bal.loc['TotalDebt'],name='Dívida Total'), row=2, col=2)
    fig2.update_layout(title_text=f'<b>Avaliação Fundamentalista de {ativo}', template='plotly_dark', showlegend=False, height=500, width=1000)

    st.plotly_chart(fig2)




pagina_web()

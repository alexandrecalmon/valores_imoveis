import pandas as pd
import streamlit as st
from pycaret.regression import predict_model, load_model
import plotly.express as px
from sklearn.datasets import load_boston

def Pegar_Dados():
    boston = load_boston()
    dados = pd.DataFrame(boston.data, columns=boston.feature_names)
    dados['MEDV'] = boston.target
    return dados

data = Pegar_Dados()

# Titulo
st.title('Prevendo Valores de Imoveis de Boston')

# Subtitulo
st.markdown('Este App utilizado para exibir a solução de Machine Learning para prever valores os imóveis em Boston.')

# Lista de atributos a serem exibidos por padrão
cols = ['RM', 'PTRATIO', 'LSTAT', 'MEDV']
multselect = st.multiselect('Atributos', data.columns.tolist(), default=cols)

# Exibindo os top10 registros do dataframe
st.dataframe(data[multselect].head(10))
st.subheader('Distribuição de imóveis por preço')

# Definingo a faixa de valores
faixa_valores = st.slider('Faixa de Preço', float(data.MEDV.min()), 150., (10.0, 100.0))

# Filtrando os dados
dados = data[data['MEDV'].between(left=faixa_valores[0], right=faixa_valores[1])]

# Plotando a distribuicao dos dados
fig = px.histogram(dados, x='MEDV', nbins=100, title='Distribuição de Preços')
fig.update_xaxes(title='MEDV')
fig.update_yaxes(title="Total Imóveis")
st.plotly_chart(fig)


# mapeando dados do usuário para cada atributo
crim = st.sidebar.number_input('Taxa de Criminalidade', value=data.CRIM.mean())
indus = st.sidebar.number_input('Proporção de Hectares de Negócio', value=data.INDUS.mean())
chas = st.sidebar.selectbox('Faz limite com o rio?' , ('Sim', 'Não'))
zn = st.sidebar.number_input('Proporção de terreno em lotes', value=data.ZN.mean())
chas = 1 if chas =='Sim' else 0 # Transforando o ddo de entrada em binario
nox = st.sidebar.number_input('Concentração de óxido nítrico', value=data.NOX.mean())
rm = st.sidebar.number_input('Número de Quartos', value=1)
ptratio = st.sidebar.number_input('Índice de alunos para professores', value=data.PTRATIO.mean())
b = st.sidebar.number_input('Proporção de pessoas com descendencia afro-americana', value=data.B.mean())
lstat = st.sidebar.number_input('Porcentagem de status baixo', value=data.LSTAT.mean())

# Tansformando os atributos mapeados em dataframe
dados = {'CRIM': crim, 'INDUS': indus, 'CHAS': chas, 'NOX': nox, 'RM': rm, 'PTRATIO': ptratio, 'B': b, 'ZN': zn, 'LSTAT': lstat}
dados = pd.DataFrame([dados])

# Executando a predição e mostrando o resultado
btn_predict = st.sidebar.button('Efetuar Predição')

# Carregando o modelo treinado
model = load_model('previsor_vendas_imoveis')

if btn_predict:
    previsao = predict_model(model, data=dados)
    st.subheader("O valor previsto para o imóvel é: ")
    result = f"US $ {previsao['Label'][0]}"
    st.write(result)



# PREVIS-O_PRE-O_LEITE
O script tem como objetivo realizar uma prevosão estimada do preço de venda do leite, usando informções obtidas do milkpoint 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

#Pacotes para modelagem 
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import kpss
from scipy.stats import friedmanchisquare


#Dados

DADOS_LPI=pd.read_excel(r"C:\Users\tecno\.spyder-py3\modelagem de preço\dados.xlsx")

plt.figure(figsize=(12, 6))
plt.plot(DADOS_LPI["MÊS"], DADOS_LPI["LPI"], label="Preço do Leite")
plt.title("Série Temporal: Preço do Leite (LPI)")
plt.xlabel("Data")
plt.ylabel("Preço")
plt.grid(True)
plt.legend()
plt.show()

"""
Apartir de agora sem feitos testes para identificar se a série é estacionaria, se possui tendencia,
sazonalidade e se possui homocedasticidade. 
"""
# Seleciona apenas a coluna numérica da série
serie = DADOS_LPI['LPI']  # Substitua 'LPI' pelo nome da coluna numérica desejada

# teste de ADF(Augmented dickey fuller)
result = adfuller(serie)

# Exibe os resultados
print(f"Estatística ADF: {result[0]:.4f}")
print(f"p-valor: {result[1]:.4f}")
print(f"Número de lags usados: {result[2]}")
print(f"Número de observações usadas: {result[3]}")
print("Valores críticos:")
for key, value in result[4].items():
    print(f"   {key}: {value:.4f}")

# Interpretação do p-valor
if result[1] < 0.05:
    print("Os dados são estacionários.")
else:
    print("Os dados não são estacionários. Diferenciação pode ser necessária.")


# teste de KPSS(Kwiatkowski Phillips-Schmidt- shim)
result = kpss(serie, regression='c', nlags="auto")  # 'c' é para estacionaridade em torno da média

# Exibe os resultados
print(f"Estatística KPSS: {result[0]:.4f}")
print(f"p-valor: {result[1]:.4f}")
print(f"Número de lags usados: {result[2]}")
print("Valores críticos:")
for key, value in result[3].items():
    print(f"   {key}: {value:.4f}")

# Interpretação do p-valor
if result[1] < 0.05:
    print("Rejeitamos a hipótese nula: A série não é estacionária.")
else:
    print("Não rejeitamos a hipótese nula: A série é estacionária.")
 
    

 # Teste ADF para identificar raízes unitárias
"""
Para descobrirmos se a série é estacionária foi criado um teste para identificar.

H₀: A série tem uma raiz unitária, ou seja, não é estacionária.
H₁: A série não tem uma raiz unitária, ou seja, é estacionária.
"""

from statsmodels.tsa.stattools import adfuller

def adfuller_test(y):
    adf_result = adfuller(y)
    print("ADF Statistic:", adf_result[0])
    print("P-Value:", adf_result[1])
    print("Critical Values:", adf_result[4])
    if adf_result[1] < 0.05:
        print("A série é estacionária (rejeitamos H₀).")
    else:
        print("A série não é estacionária (não rejeitamos H₀).")

# Exemplo de uso
adfuller_test(DADOS_LPI['LPI'])
    
"""
Para transformar a série estacionaria é necessario retirar a tendência e a sazonalidade

"""


#Transformando em um série logarítmica 
DADOS_LPI['LPI_log'] = np.log(DADOS_LPI['LPI'])
plt.plot(DADOS_LPI['LPI_log'])
plt.title('Série com Transformação Logarítmica')
plt.show()

#Remoção da tendência 
DADOS_LPI['LPI_diff'] = DADOS_LPI['LPI_log'].diff()  # Use a série logarítmica ou original
DADOS_LPI = DADOS_LPI.dropna(subset=['LPI_diff'])
plt.plot(DADOS_LPI['LPI_diff'])
plt.title('Série Diferenciada')
plt.show()

#Remoção da sazonalidade
DADOS_LPI['LPI_seasonal_diff'] = DADOS_LPI['LPI'] - DADOS_LPI['LPI'].shift(12)  # Exemplo para periodicidade anual
DADOS_LPI = DADOS_LPI.dropna(subset=['LPI_seasonal_diff'])
plt.plot(DADOS_LPI['LPI_seasonal_diff'])
plt.title('Série com Sazonalidade Removida')
plt.show()

#aplicação do teste de estacionalidade novamente 
from statsmodels.tsa.stattools import adfuller
result = adfuller(DADOS_LPI['LPI_diff'])
print(f"Estatística ADF: {result[0]:.4f}, p-valor: {result[1]:.4f}")
if result[1] < 0.05:
    print("Série é estacionária.")
else:
    print("Série ainda não é estacionária. Pode ser necessário aplicar mais transformações.")
    
"""
SAZONALIDADE
"""
    
from statsmodels.tsa.seasonal import seasonal_decompose

decomposicao = seasonal_decompose(DADOS_LPI["LPI"], model='additive', period=12)  # Ajuste o 'period' conforme a sazonalidade esperada

# Plotar os componentes
decomposicao.plot()
plt.show()

#A série tem comportamento diferentes quando é feita a remoção da tendência.
#Quando tiramos a tendência a série que não era estacionaria tem com comportamento diferente. 


"""
Gráfico de ACF (Autocorrelação) e PACF(Autocorrelação parcial)

"""
n1 = len(DADOS_LPI['LPI_diff'])
print(f"Tamanho da série: {n1}")
max_lags = (n1 // 2 - 1)


fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Plote ACF
sm.graphics.tsa.plot_acf(DADOS_LPI['LPI_diff'], lags=max_lags, ax=ax[0])

# Plote PACF
sm.graphics.tsa.plot_pacf(DADOS_LPI['LPI_diff'], lags=max_lags, ax=ax[1])

plt.show()



#Teste de Friedman

tamanho_truncado = (len(DADOS_LPI["LPI"]) // 12) * 12
dados_truncados = DADOS_LPI["LPI"].iloc[:tamanho_truncado]

# Criar grupos sazonais com tamanho igual
dados_sazonais = [dados_truncados[i::12] for i in range(12)]

# Teste de Friedman
from scipy.stats import friedmanchisquare

stat, p = friedmanchisquare(*dados_sazonais)
print(f"Estatística de Teste: {stat}, p-valor: {p}")

# Interpretação
if p < 0.05:
    print("A série apresenta sazonalidade significativa.")
else:
    print("Não foi detectada sazonalidade significativa.")
    

#Modelagem: Modelo Prophet

dados_prophet=DADOS_LPI.rename(columns={"MÊS":"ds", "LPI":"y"})

print(dados_prophet.head(124))
model = Prophet()
model.fit(dados_prophet)
future= model.make_future_dataframe(periods=6, freq='M')  # 'M' indica frequência mensal

# Fazer previsões
forecast = model.predict(future)

# Visualizar previsões
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Gráfico das previsões
model.plot(forecast)
model.plot_components(forecast)
m=prophet(dados_prophet)
m=prophet()
m=Prophet()
m.fit(dados_prophet)
future= model.make_future_dataframe(periods=6, freq='M')
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
future= model.make_future_dataframe(periods=12, freq='M')
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)
plot_components_plotly(m, forecast)
!pip install plotly
!pip install notebook
!pip install ipywidgets
from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from prophet.plot import plot_plotly

plot_plotly(m, forecast)
from IPython.display import display
from prophet.plot import plot_plotly
fig = plot_plotly(m, forecast)
display(fig)
fig.show()
from prophet.plot import plot_components

# Gráfico simples usando Matplotlib
m.plot(forecast)
plt.show()

# Componentes da previsão
plot_components(m, forecast)
plt.show()
forecast
forecast.head(120)
forecast.head(165)
pd.set_option('display.max_columns', None)
forecast.head()
forecast.to_csv("forecast.csv", index=False)
pd.set_option('display.max_columns', None)
forecast.head()
forecast.to_csv("forecast.csv", index=False)
fore=forecast.to_csv("forecast.csv", index=False)
print(fore)
forecast.info()
print(forecast.tail(15))

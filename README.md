# Machine Learning from Scratch

Tutorial apresentado na Python Brasil 2019 em Ribeirão Preto - SP

## Configuração
- crie uma virtualenv com python 3.7
- pip install pipenv
- pipenv install

## Regressão Linear
Metodo para aproximar duas variaveis linearmente (localização de imovel vs
preço, idade do carro vs preço). Dizemos que duas variaveis tem relação
linear se plotarmos seus valores num grafico e eles "parecerem" uma linha.
Os passos para gerar uma regressão linear são 5:


1. randomizar os inputs da função de hipotese
2. computar o Mean Squared Error
3. Calcular as derivativas parciais
4. Atualiazar os parametros baseados nas derivativas e na taxa de apredizado
5. repetir do 2 ao 4 até o error ser o menor possivel.

### Função de Hipotese (Hypoteses Function)

h0(x) = theta1(x) + theta0 == y = m(x) + b

é a função linear que vimos no primeiro grau da escola, o parametro theta1
(m) define a angulação da linha e o theta0 (b) define onde a linha cruza o
eixo y.
# Sistemas Lineares
## Objetivos da aula
* Definir o que são sistemas lineares
* Apresentar uma forma de solução desses sistemas
* Mostrar graficamente um sistema linear

## Intuição
Vamos considerar que temos um problema de negócio no qual precisamos saber qual o preço de uma residência baseado em suas características como quantidade de quartos, área total, número de banheiros e etc. Podemos representar esse problema como uma equação do tipo:
$$preço= preço\_medio\_região + a \cdot número\_quarto + b \cdot número\_banheiros + \cdots + n \cdot area\_total$$
ou de forma mais genérica:
$$y = a x_0+b x_1+ \cdots+nx_{n}$$
No qual nossa varíável independente $$y$$ representa o preço da residência, $$x_0$$ até $$x_n$$ são as features do nosso problema (quantidade de quartos, área total, número de banheiros) e $$a, b, ...,  n$$ são os parâmetros cujos valores solucionam nossa equação.
Cada informação de preço de residência que coletamos gera uma equação que possui valores de $$y$$ e $$x_{i...n}$$ distintos. Se tivermos $$m$$ preços de residências podemos criar o que chamados um sistema de $$m$$ equações:

$$y_0 = a x_{0,0}+b x_{0,1}+ \cdots+n x_{0,n}$$
$$y_1 = a x_{1,0}+b x_{1,1}+ \cdots+n x_{1,n}$$
$$\qquad\qquad\qquad\quad\vdots$$
$$y_m = a x_{m,0}+b x_{m,1}+ \cdots+n x_{m,n}$$

Nesse caso, nossa intenção é encontrar quais são os parâmetros de $$a$$ até $$m$$ que melhor solucionam esse conjunto de equações.

## Definição de sistemas lineares
Sistema linear é um **conjunto de equações lineares** que possui m equações e n incógnitas e assume uma forma genérica do tipo:

$$a_{0,0} x_0+a_{0,1} x_1+ \cdots+a_{0,n}x_{n}=y_0$$
$$a_{1,0}x_0+a_{1,1}x_1+…+a_{1,n}x_n=y_1$$
$$\qquad \qquad \qquad \qquad \quad \vdots$$
$$a_{m,0}x_0+a_{m,1}x_1+ \cdots +a_{m,n}x_n=y_m$$

 Nos casos que iremos trabalhar na aula de hoje, as incógnitas serão representadas pelos valores de $$x$$.
Esse sistema linear pode ser modificado para ser escrito da seguinte forma matricial:

$$\begin{bmatrix}
a_{0,0} x_0 + a_{0,1} x_1 + \ldots + a_{0,n} x_n \\
a_{1,0} x_0 + a_{1,1} x_1 + \ldots + a_{1,n} x_n \\ 
\vdots \qquad \qquad \ddots \qquad \qquad \vdots \\ 
a_{m,0} x_0 + a_{m,1} x_1 + \ldots + a_{m,n} x_n  
\end{bmatrix} = 
\begin{bmatrix}
 y_0 \\
 y_1 \\
 \vdots \\
 y_m
\end{bmatrix}$$

Como os valores de $$x_0$$ à $$x_n$$ são comuns a todas as linhas, podemos separá-los para reestruturar nossa equação matricial da forma:

$$\begin{bmatrix}
a_{0,0} + a_{0,1} + \ldots + a_{0,n} \\ 
a_{1,0} + a_{1,1} + \ldots +  a_{1,n} \\ 
\vdots \qquad \qquad \ddots \qquad \qquad \vdots  \\ 
a_{m,0} + a_{m,1} + \ldots + a_{m,n}  
\end{bmatrix}
\begin{bmatrix}
x_0 \\ x_1 \\ \vdots \\ x_n
\end{bmatrix} 
= 
\begin{bmatrix}
 y_0 \\ y_1 \\ \vdots \\ y_m
\end{bmatrix}$$

Também podemos representar resumidamente como: $$Ax = y$$, onde $$A$$ é a matriz dos coeficientes, $$x$$ é o vetor de $$n$$ incógnitas que soluciona o sistema e $$y$$ o vetor dos termos independentes.

##### Exemplo
Como exemplo de um sistema linear, vamos considerar o sistema de equações:

$$x_0+3x_2=y_0$$
$$3x_0+2x_1+x_2=y_1$$
$$x_0+x_2=y_2$$

em notação matricial podemos representá-las por:

$$\begin{bmatrix}
1 & 0 & 3 \\
3 & 2 & 1 \\
1 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x_0 \\ x_1 \\ x_2
\end{bmatrix}
= \begin{bmatrix}
y_0 \\ y_1 \\ y_2
 \end{bmatrix}$$

## Solução de sistemas lineares
Para solucionar um sistema linear é preciso encontrar o valor das incógnitas de modo que as todas as equações matemáticas do sistema sejam verdadeiras.
Existem algumas formas de solucionarmos sistemas lineares como o Método de Eliminação de Gauss, a Regra de Cramer e Método da Matriz Inversa. Aqui vamos nos ater a essa última. Partindo da nossa equação (1), multiplicamos os dois lados da equação pela matriz inversa de $$A$$ para desaparecer com ela do lado esquerdo (2). Multiplicando a inversa de $$A$$ por $$b$$ obteremos a solução $$X$$ do nosso problema (3). Seguindo esses passos, conseguiremos encontrar o conjunto solução desse sistema linear, ou pelo menos uma das soluções.

$$AX=y \qquad\qquad\qquad\qquad\!\!\qquad(1)$$
$$A \cdot A^{-1} \cdot X = A^{-1} \cdot y \quad \! \qquad(2)$$
$$X=A^{-1} \cdot y \qquad \qquad \qquad \!\! \qquad(3)$$

Esses passos só poderão ser realizados se a matriz $$A$$ possuir uma inversa e isso só ocorre se o seu determinante for diferente de zero. 

#### Numpy
Com o `numpy`, a solução do sistema utilizando os passos mencionados é feito da seguinte forma:
```python
import numpy as np

# Cria matriz A
>>> A = np.array([[1, 0, 3],
                  [3, 2, 1],
                  [1, 0, 1]])

# Cria matrix y
>>> y = np.matrix([[1], [1], [1]])

# Para encontrar a matrix inversa de A utilizamos o método `np.linalg.inv()`
>>> A_I = np.linalg.inv(A)

# Utilizando a equação (3) acima podemos encontrar o conjunto solução desse sistema X multiplicando a Inversa de A por y com o método np.dot()
>>> X = np.dot(A_I, y)
>>> X
matrix([[ 1.],
        [-1.],
        [ 0.]])
```
Também podemos utilizar a o método `numpy.linalg.solve` que resolve sistemas lineares de solução única nos quais a matriz A é quadrada e o determinane é diferente de zero).
```
>>> X = np.linalg.solve(A,B)
>>> X
matrix([[ 1.],
        [-1.],
        [ 0.]])
```
## Visualizando sistemas lineares
Suponha que acabou de ocorrer um assalto a banco e o ladrão fugiu a velocidade de 2,5km/s. Após 5 segundos, um policial iniciou a perseguição com uma velocidade de 3km/s. Vamos desconsiderar a aceleração, trânsito e o atrito e considerar que as velocidades se manterão constantes. Com essas informações, como determinar qual a distância e o tempo no qual o policial irá alcançar o ladrão?
Para resolver esse problema precisamos escrever as equações de movimento de ambos:
Ladrão: $$d_l = 2.5 * t$$
Policial: $$d_p = 3 * (t-5)$$

O encontro ocorrerá quando ambos estiverem na mesma posição no espaço, ou seja,  $$d_l=d_p=d$$
Podemos escrever essas informações em forma matricial:

$$\begin{bmatrix}
1 & -2.5 \\
1 & -3
\end{bmatrix}
\begin{bmatrix}
d \\ t
\end{bmatrix}
= \begin{bmatrix}
0 \\ -15
 \end{bmatrix}$$
 
 Para encontrar a solução desse sistema linear em python:
```python
>>> A = np.array([[1, -2.5], [1, -3]])
>>> y = np.array([0, -15])
>>> np.linalg.solve(A, y)
array([75., 30.])
```
Da forma como montamos o problema, o primeiro elemento representa a distância e o segundo o tempo $$\begin{bmatrix} d \\ t \end{bmatrix}$$. Assim, eles irão se encontrar na distância $$d=75m$$ no tempo $$t=30min$$.

Com o auxílio da biblioteca `matplotlib` podemos visualizar graficamente o ponto no qual a curva do ladrão irá se encontrar com a do policial. Para isso precisamos graficar tanto a equação do ladrão quanto o do policial e encontrar um ponto no qual elas se cruzam.
```python
import numpy as np
import matplotlib.pyplot as plt

# Cria eixo x com 1000 valores de tempo espaçados linearmente entre 0 e 40 minutos
t = np.linspace(0, 40, 1000) # início, fim, # pontos

# Equação da distância percorrida pelo ladrão:
>>> d_l = 2.5 * t 

# Equação da distância percorrida pela polícia:
>>> d_p = 3 * (t-5)

# Vamos plotar esses duas curvas e ver onde elas se encontram
>>> fig, ax = plt.subplots()
>>> plt.title('Momento de captura')
>>> plt.xlabel('Tempo (em minutos)')
>>> plt.ylabel('Distância (em km)')
>>> ax.set_xlim([0, 40])
>>> ax.set_ylim([0, 100])
>>> ax.plot(t, d_l, c='green')
>>> ax.plot(t, d_p, c='brown')
>>> plt.axvline(x=30, color='purple', linestyle='--')
>>> _ = plt.axhline(y=75, color='purple', linestyle='--')
```
![image](https://user-images.githubusercontent.com/62657143/147373473-e9624e0d-fd91-4261-a82f-098e489d6709.png)
Esse é um exemplo de sistema linear em que teremos apenas uma solução.

## Principais usos em ML
* Em modelos de ML como deep learning
* Redução de dimensionalidade (por exemplo, PCA)
* Criar rankings (utilizando os autovetores)
* Sistemas de recomendação (exemplo: singular value decomposition - SVD)
* Processamento de Linguagem Natural (exemplo: SVD e matrix factorization)

## Exercícios
(1) Utilizando o numpy, resolva o sistema linear:
$$\begin{bmatrix}
         1 & 1 & 1 & 1 & 1 & 1\\
         10 & 2 & 3 & 4 & 5 & 1\\
         1 & 1 & 2 & 4 & 1 & 1\\
         2 & 7 & 1 & 1 & 10 & 2\\
         3 & 1 & 1 & 1 & 20 & 1\\
         2 & 6 & 1 & 4 & 5 & 1\\
         \end{bmatrix}.\begin{bmatrix}
                               a_{1} \\
                               b_{1} \\
                               c_{1} \\
                               d_{1} \\
                               e_{1} \\
                               f_{1} \\
                               \end{bmatrix} = \begin{bmatrix}
                                                 21 \\
                                                 70 \\
                                                 36 \\
                                                 85 \\
                                                 118 \\
                                                 64 \\
                                                 \end{bmatrix}$$
                                                 
Você de encontrar como solução o vetor $$[1 \ 2 \ 3 \ 4 \ 5 \ 6]$$

(2) (URCA 2015/2)
Uma transportadora possui depósitos em três cidades, Juazeiro do Norte, Iguatu e Sobral. Márcio, José e Pedro são funcionários desta transportadora e transportam mercadorias apenas entre esses depósitos. Ontem Márcio saiu de Sobral, entregou parte da carga em Iguatu e o restante em Juazeiro do Norte, percorrendo ao todo 538 Km. Dias antes, José saiu de Iguatu, fez entregas em Sobral e depois seguiu para Juazeiro do Norte, percorrendo 905 Km. Por fim, semana passada, Pedro saiu de Iguatu, descarregou parte da carga em Juazeiro do Norte e o restante em Sobral, percorrendo ao todo 681 Km. Sabendo que os três motoristas cumpriram rigorosamente o percurso imposto pela transportadora, quanto percorreria um motorista que saísse de Juazeiro do Norte, passasse por Iguatu, depois por Sobral e retornasse para Juazeiro do Norte?
Resposta: 1062km


# Algebra Linear: Vetores, Espaço Vetorial e Matrizes

## Objetivos da aula
Entender os conceitos de:
- Vetores
- Espaço vetorial
- Matrizes

## Álgebra Linear

Álgebra linear é um ramo da Matemática dedicado à estudar as propriedades de vetores, matrizes, espaços vetoriais, transformações lineares e sistemas de equações lineares.  A Álgebra Linear inicia-se com o estudo de equações lineares a uma variável que são equações da forma $$ax+b=0$$, evoluindo, posteriormente para o estudo de sistemas de equações lineares os quais resolvemos com matrizes. 

## Vetores
Vetores são um conjunto de números posicionais e unidimensionais, tal que cada elemento pode ser acessado por um índice. A representação de um vetor pode ser feita com uma seta sobre a letra que representa nosso vetor, $$\overrightarrow{v}$$, ou em caixa baixa, em itálico e negrito, $$\mathbf{v}$$.
No Python podemos criar um vetor $$\mathbf{v}=(67, 250, 2)$$ utilizando a biblioteca NumPy:

```python
>>> import numpy as np
>>> vetor = np.array([[67, 250, 2]])
>>> vetor
array([[ 67, 250, 2]])
```

Nesse caso dizemos que esse objeto é um vetor linha com tamanho 3 (1,3).
```python
>>> vetor.shape
(1, 3)
```
Damos preferência para criar os vetores utilizando listas de listas para manter o tamanho da forma (1,3). Tente criar um vetor da forma np.array([67, 250, 2]) e compare o shape dele com o que obtivemos anteriormente.
Como dito anteriormente, para acessar elementos do nosso vetor precisamos informar qual a posição desse elemento, lembrando que os índices em python começam com zero:
```python
>>> vetor[0]
67
>>> vetor[1]
250
```

## Vetores no $$\mathbb{R}^n$$
Vamos considerar o plano cartesiano com seus eixos x e y. O par ordenado (x,y) pode ser um ponto ou vetor nesse plano cartesiano que, por sua vez, é uma interpretação geométrica do conjunto $$\mathbb{R}^2=\{(x,y)/ x,y \in \mathbb{R}\}$$.
![image](https://user-images.githubusercontent.com/62657143/144061489-fe1c064d-9393-481f-bc90-d168da7b93e6.png)
Da mesma forma, podemos definir um vetor de tamanho três que representa uma localização em um espaço tridimensional com eixos x, y, z.
Se n = 1 , 2 ou 3, temos os casos conhecidos de  $$\mathbb{R}$$ , $$\mathbb{R}^2$$ e $$\mathbb{R}^3$$ que correspondem, respectivamente, à reta, ao plano e ao espaço tridimensional.
Embora não tenhamos mais a visão geométrica, essa mesma ideia se estende aos espaços $$\mathbb{R}^4$$, $$\mathbb{R}^5 \ldots \mathbb{R}^n$$ os quais podemos descrever como:
$$\mathbb{R}^4 \rightarrow (x_1,x_2,x_3,x_4)$$
$$\mathbb{R}^5 \rightarrow (x_1,x_2,x_3,x_4,x_5)$$
$$\mathbb{R}^n = \{(x_1,x_2, \ldots ,x_n)/ x_i \in \mathbb{R}\}$$

A Álgebra Linear irá nos ajudar a expandir nossos cálculos para essas outras dimensões.

## Vetores no $$\mathbb{R}^n$$
Considere dois vetores $$\overrightarrow{u}=(u_1, u_2, \ldots , u_n)$$ e $$\overrightarrow{v}=(v_1, v_2, \ldots,  v_n)$$ em $$\mathbb{R}^n$$ e $$\alpha$$ um escalar qualquer pertencente aos $$\mathbb{R}$$, define-se:
a) Eles são ditos iguais se $$u_1=v_1,u_2=v_2,…,u_n=v_n$$:
```python
>>> v = np.array([67, 250, 2])
>>> u = np.array([67, 250, 2])
>>> v == u
array([True,  True,  True])
```
b) A soma $$\mathbf{u}+\mathbf{v}$$ é definida por $$\mathbf{u}+\mathbf{v} = (u_1 + v_1, u_2 + v_2, \ldots, u_n + v_n)$$ 
```python
>>> u + v
array([134, 500, 4])
```
c) O múltiplo escalar $$\alpha$$ de $$\mathbf{v}$$ é $$\alpha \mathbf{v}=(\alpha v_1, \alpha v_2, \ldots, \alpha v_n)$$
```python
>>> 3*v
array([201, 750, 6])
```
d) O produto escalar $$\overrightarrow{u} \cdot \overrightarrow{v} = u_1 v_1 + u_2 v_2 + \ldots + u_n v_nu$$
```python
>>> u * v
array([4489, 62500, 4])
```
e) Podemos calcular o módulo (Euclidiando) do vetor: $$| \overrightarrow{u} |= \sqrt{u_1^2 + u_2^2 + \ldots + u_n^2}$$
Na próxima seção apresentaremos outras funções para determinar a maginitude do vetor.

Em resumo, vetores são objetos especiais que obedecem esses axiomas de soma e multiplicação por escalares. Qualquer objeto que satisfaça essas propriedades pode ser considerado um vetor. Exemplos de vetores são: polinômios, matrizes, sinais de aúdio e uma tupla com elementos de $$\mathbb{R}^n$$. 

## Norma de vetores
Vetores também podem representar uma determinada magnitude e direção. Essa magnitude do vetor pode ser obtida através de uma classe de funções denominadas normas que calculam a distância que o vetor encontra-se da origem. Em Machine Learning, as normas são muito utilizadas para calcular o erro da predição dos modelos. Aqui vamos apresentar como calcular as normas mais utilizadas para isso.

### Norma $$L^2$$
Ela também é conhecida como norma Euclidiana por calcular a distância simples da origem. Como mencionado anteriormente, essa é a norma mais conhecida e é descrita por:
$$||\mathbf{v}||_2  = \sqrt{\sum v_i^2} = \sqrt{v_1^2 + v_2^2 + \ldots + v_n^2}$$ 

Muitas vezes ela é representada sem o subscrito $$||\mathbf{v}||$$. Em Python podemos calcular a norma por vias normais:

```python
>>> np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
258.53006007
```
ou utilizar o método [np.linalg.norm()](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) do numpy, no qual o primeiro parâmetro é o vetor que estamos tratando e o segundo é a ordem da norma:
```python
>>> np.linalg.norm(u, 2)
258.53006007
```
Esse módulo é conhecido como Norma $$L^2$$ e é representado matematicamente por $$||\mathbf{v}||_2$$. Essa á norma mais utilizada em Machine Learning.

### Norma $$L^1$$
Na Norma $$L^1$$, somamos o valor absoluto de cada um dos elementos do vetor:
$$||\mathbf{v}||_1  = \sum |v_i| = |v_1| + |v_2| + \ldots + |v_n|$$ 

```python
>>> np.abs(u[0]) + np.abs(u[1]) + np.abs(u[2])
32
# Com o método np.linalg.norm()
>>> np.linalg.norm(u, 1)
32
```
Como esperado, a norma $$L^1$$ resulta em um tamanho de vetor diferente da $$L^2$$.
Outro ponto importante é que, ao contrário da norma $$L^2$$, essa função varia linearmente em todos os pontos do espaço, esteja ele próximo ou distante da origem.

### Norma $$L^2_2$$
Ela é muito parecida com a norma $$L^2$$, mas nesse caso não extraimos a raiz quadrada após a soma:
$$||\mathbf{v}||_2^2  = {\sum v_i^2} = {v_1^2 + v_2^2 + \ldots + v_n^2}$$ 
```python
>>> u[0]**2 + u[1]**2 + u[2]**2
654
# Com o método np.linalg.norm()
>>> np.linalg.norm(u, 2)**2
654
```
A vantagem de usar a norma $$L^2_2$$ é que ela é computacionalmente mais rápida de se calcular, em compensação, ela cresce muito devagar para valores próximos a origem enquanto para valores distantes ela cresce rapidamente.

### Norma infinito ou norma do máximo
Retorna o valor absoluto máximo :
$$||\mathbf{v}||_{\infty}  = max_i |v_i| = max(|v_1|, |v_2|, \ldots, |v_n|)$$ 
```python
>>> np.max(np.abs(u))
25
# Com o método np.linalg.norm()
>>> np.linalg.norm(u,  np.inf)
25
```

## Espaço vetorial (ou espaço linear)
Um espaço vetorial é uma estrutura matemática constituída por um conjunto $$V$$ não vazio, cujos elementos são chamados vetores. Nesse espaço, estão definidas as operações de soma e de multiplicação por escalar (números reais) que devem obedecer  oito regras (axiomas) para ser considerado um espaço vetorial. 
A partir de agora, podemos definir essas operações de tal forma que não tenham semelhança com as operações usuais dos espaços $$\mathbb{R}^2$$.

A **soma entre vetores** deve satisfazer os axiomas: 
1. Associatividade: 
$$(\mathbf{u}+\mathbf{v}) + \mathbf{w} = \mathbf{u} +(\mathbf{v}+\mathbf{w})$$, para quaisquer vetores $$\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$$;
2. Elemento neutro: 
existe o vetor $$\mathbf{0} \in V$$ que satisfaz $$\mathbf{0}+\mathbf{v} = \mathbf{v}+\mathbf{0} = \mathbf{v}$$, para qualquer $$v \in V$$;
3. Inverso aditivo: 
para cada $$v \in V$$, existe o inverso $$\mathbf{u} = -\mathbf{v} \in V$$, que satisfaz $$\mathbf{v} + \mathbf{u} = 0$$;
4. Comutatividade: 
$$\mathbf{u}+\mathbf{v} = \mathbf{v}+\mathbf{u}$$, para quaisquer  $$u, v \in V$$;

A **multiplicação de vetor por escalar** deve satisfazer os axiomas:
1. Associatividade da multiplicação por escalar: 
$$\alpha \cdot {(\beta \cdot \mathbf{v})} = (\alpha \cdot \beta) \cdot \mathbf{v}$$, para quaisquer $$\alpha, \beta \in \mathbb{R}$$ e qualquer $$v \in V$$; 
2. Vale que $$1 \cdot \mathbf{v} = \mathbf{v}$$, ou seja, a unidade dos números reais não altera os vetores de $$V$$; 
3. Distributiva de um escalar em relação à soma de vetores:
$$\alpha \cdot (\mathbf{u}+\mathbf{v}) = \alpha \cdot \mathbf{u} + \alpha \cdot \mathbf{v}$$, para qualquer $$\alpha \in \mathbb{R}$$ e quaisquer $$u, v \in V$$; 
4. Distributiva da soma de escalares em relação a um vetor:
$$(\alpha+\beta) \cdot \mathbf{v} = \alpha \cdot \mathbf{v} + \beta \cdot \mathbf{v}$$, para quaisquer $$\alpha, \beta \in \mathbb{R}$$ e qualquer $$v \in V$$.

#### Exemplo de espaço vetorial
Vamos provar que $$V = \mathbb{R}^2 = \{( x_1 , x_2 )| x_1 , x_2 \in \mathbb{R}^2\}$$  com as operações usuais de adição de vetores e de multiplicação de vetores por escalares reais definidos em $$\mathbb{R}^2$$  é um espaço vetorial. 
Adição de vetores: $$(x_1 , x_2 ) + (y_1 , y_2 ) = ( x_1 + y_1 , x_2 + y_2 )$$
Multiplicação de vetores por escalar: $$k ( x_1 , x_2 ) = ( kx_1 , kx_2 )$$

Sejam $$\mathbf{u} = ( u_1 , u_2 ), \mathbf{v} = ( v_1 , v_2 )$$ e $$\mathbf{w} = ( w_1 , w_2 )$$ vetores quaisquer do $$\mathbb{R}^2$$ e $$k$$ e $$l$$ escalares quaisquer em $$\mathbb{R}$$: 
a) Comutativa da adição:
$$\mathbf{u} + \mathbf{v} = ( u_1 , u_2 ) + ( v_1 , v_2 ) = ( u_1 + v_1 , u_2 + v_2 ) = ( v_1 + u_1 , v_2 + u_2 ) = ( v_1 , v_2 ) + ( u_1 , u_2 )
= \mathbf{v} + \mathbf{u}$$
b) Associativa da adição:
$$\mathbf{u} + ( \mathbf{v} + \mathbf{w} ) = ( u_1 , u_2 ) + (( v_1 , v_2 ) + ( w_1 , w_2 )) = ( u_1 , u_2 ) + ( v_1 + w_1 , v_2 + w_2 )
= ( u_1 + ( v_1 + w_1 ), u_2 + ( v_2 + w_2 )) = (( u_1 + v_1 ) + w_1 ,( u_2 + v_2 ) + w_2 )
= ( u_1 + v_1 , u_2 + v_2 ) + ( w_1 , w_2 ) = (( u_1 , u_2 ) + ( v_1 , v_2 )) + ( w_1 , w_2 )
= ( \mathbf{u} + \mathbf{v} ) + \mathbf{w}$$
c) Elemento neutro da adição:
Devemos encontrar um vetor \mathbf{0} em $$\mathbb{R}^2$$ tal que $$\mathbf{u} + \mathbf{0} = \mathbf{u}$$ . Para tanto, basta  tomar o vetor neutro como $$\mathbf{0} = (0,0)$$. Desse modo, teremos:
$$\mathbf{u} + \mathbf{0} = ( u_1 , u_2 ) + (0,0) = ( u_1 + 0, u_2 + 0) = ( u_1 , u_2 ) = \mathbf{u}$$.
d) Elemento simétrico da adição:
Devemos mostrar que cada vetor $$u$$ em $$\mathbb{R}^2$$ tem um simétrico $$-\mathbf{u}$$ tal que $$\mathbf{u} + ( - \mathbf{u} ) = \mathbf{0}$$. Para isso, basta definir $$- \mathbf{u} = ( - u_1 , - u_2 )$$. Assim, teremos,
$$\mathbf{u} + ( - \mathbf{u} ) = ( u_1 , u_2 ) + ( - u_1 , - u_2 ) = ( u_1 + ( - u_1 ), u_2 + ( - u_2 )) = (0,0) = \mathbf{0}$$.
e) $$k ( \mathbf{u} + \mathbf{v} ) = k (( u_1 , u_2 ) + ( v_1 , v_2 )) = k ( u_1 + v_1 , u_2 + v_2 ) = ( k ( u_1 + v_1 ), k ( u_2 + v_2 ))
= ( ku_1 + kv_1 , ku_2 + kv_2 ) = ( ku_1 , ku_2 ) + ( kv_1 , kv_2 ) = k ( u_1 , u_2 ) + k ( v_1 , v_2 )
= k \mathbf{v} + k \mathbf{u}$$
f) $$( k + l ) \mathbf{u} = ( k + l )( u_1 , u_2 ) = (( k + l ) u_1 ,( k + l ) u_2 ) = ( ku_1 + lu_1 , ku_2 + lu_2 )
= ( ku_1 , ku_2 ) + ( lu_1 , lu_2 ) = k ( u_1 , u_2 ) + l ( u_1 , u_2 )
= k\mathbf{u} + l\mathbf{u}$$
g) $$k ( l \mathbf{u} ) = k ( l ( u_1 , u_2 )) = k ( lu_1 , lu_2 ) = ( k ( lu_1 ), k ( lu_2 )) = (( kl ) u_1 ,( kl ) u_2 ) = ( kl )( u_1 , u_2 )
= ( kl ) \mathbf{u}$$
h) $$1 \mathbf{u} = 1( u_1 , u_2 ) = (1 u_1 ,1 u_2 ) = ( u_1 , u_2 ) = \mathbf{u}$$

Como conseguimos provar a validade dos oito axiomas podemos concluir que a prova de que a estrutura matemática dada é um espaço vetorial real.

## Matrizes
As matrizes são objetos bidimensionais formados por um conjunto de vetores. Na figura abaixo temos uma ilustração da diferença entre um escalar, um vetor de linha, um de coluna e uma matríx: 
![image](https://user-images.githubusercontent.com/62657143/144272110-97ed9e41-0f4e-4d44-8332-4f5fc24310e2.png)
Representamos uma matriz por $$\mathbf{A[m,n]}$$ onde **m** é a **quantidade de linhas** e **n** de **colunas**. Cada elemento $$x$$ da matriz terá um índice associado a sua posição de linha e outro para a coluna, de tal forma que podemos escrever a matrix $$A[m,n]$$ como:

$$\begin{bmatrix}
   x_{0,0} & x_{0,1} & \dotsb & x_{0,n-1} \\
   x_{1,0} & x_{1,1} & \dotsb & x_{1,n-1} \\
   \vdots  & \vdots  & \ddots & \vdots  \\
   x_{m-1,0} & x_{m1,1} & \dotsb & x_{m-1,n-1}
    \end{bmatrix}$$

Repare que o último elemento da matriz é o $$x_{m-1,n-1}$$ porque os índices iniciam em zero.

#### Numpy
Para representarmos a matriz $$A[2, 3]$$:

$$A =\begin{bmatrix} 4 & 19 & 8 \\ 16 & 3 & 5 \end{bmatrix}$$ 

com o `numpy` podemos reaizar a concatenação de vetores ou utilizar o método `np.matrix`:
```python
# Concatenação de vetores
A = np.array([[4, 19, 8],
              [16, 3, 5]])

# Método np.matrix
A = np.matrix([[4, 19, 8],
               [16, 3, 5]])
```

Para **acessar uma posição específica da matriz** precisamos informar tanto o valor da linha quanto o da coluna da seguinte forma:
```python
>>> A[1, 2]
5
```
Com isso conseguimos extrair o elemento que está na linha de índice 1 e coluna de índice 2, já que o índice no `numpy` inicia-se no zero.

### Adição de matrizes
Suponha as matrizes $$A \in \mathbb{R}^{mxn}$$ e $$B \in \mathbb{R}^{mxn}$$, ambas com _m_ linhas e _n_ colunas. A soma dessas matrizes é realizada elemento a elemento da forma: 

$$A + B = \begin{bmatrix} a & b & c\\ d & e & f \end{bmatrix} + \begin{bmatrix} g & h & i \\ j & k & l \end{bmatrix} = \begin{bmatrix} a + g & b + h & c + i \\ d + j & e + k & f + l \end{bmatrix}$$

Note a necessidade das matrizes terem exatamente o mesmo tamanho.
**Exemplo:** Vamos supor que queremos realizar a seguinte soma entre matrizes:

![image](https://user-images.githubusercontent.com/62657143/144277255-97376b51-3de9-4145-8794-6524c436ff42.png)

utilizando a biblioteca `numpy` precisamos apenas utilizar o símbolo **+** entre os vetores:
```python
import numpy as np
>>> a = np.array([[1, 3, 2], [4, 5, 3]])
>>> b = np.array([[2, 1, 4], [9, 5, 2]])
>>> a + b
array([[3, 4, 6], [13, 10, 5]])
```
### Multiplicação por escalar
A multiplicação por escalar também acontece elemento a elemento:

![image](https://user-images.githubusercontent.com/62657143/144278608-137a24a7-7316-4c02-be6b-c1e46a0bf1f0.png)

Com o `numpy` fazemos uma multiplicação simples (__*__) entre o escalar e a matriz:
```python
>>> import numpy as np
>>> E = np.matrix([[3, 4],
                   [2, 7]])
>>> 10*E
matrix([[30, 40],
        [20, 70]])
```
### Multiplicação de matrizes
Na multiplicação de uma matriz $$A[n, k]$$ por uma matriz $$C[k, m]$$, obtemos como resultado uma matriz _n_ x _m_. Notem que nesse caso precisamos que a dimensão das colunas (_k_) de uma das matrizes seja igual a dimensão das linhas da outra (_k_) e que o resultado terá a mesma quantidade (_n_) de linhas da primeira e de colunas (_m_) que a segunda. 
A multiplicação de matrizes segue a regra na qual soma-se a multiplicação elemento a elemento entre as linha da primeira matriz e as colunas da segunda:
![image](https://user-images.githubusercontent.com/62657143/144279232-53b926b9-15d9-47bf-86fd-268c7a8c4775.png)

No `numpy`, essa multiplicação é representada pelo método `np.dot()`:
```python
>>> import numpy as np
>>> a = np.array([[1, 3, 2], [4, 5, 3]])
>>> c = np.array([[1, 1], [2, 3], [5, 4]])
>>> np.dot(a, c)
array([[17, 18],
       [29, 31]])
>>> np.dot(c, a)
array([[ 5,  8,  5],
       [14, 21, 13],
       [21, 35, 22]])
```
Como podemos observar do exemplo acima, a ordem de multiplicação entre as matrizes altera o resultado. 

**Obs:** Se realizarmos uma multiplicação utilizando o `*` teremos uma multiplicação simples elemento a elemento:
```python
>>> import numpy as np
>>> a = np.array([[1, 3, 2], [4, 5, 3]])
>>> b = np.array([[2, 1, 4], [9, 5, 2]])
>>> a*b
array([[ 2,  3,  8],
      [36, 25,  6]])
```

## Exercícios
(UFMT) Uma empresa fabrica três produtos. Suas despesas de produção estão divididas em três categorias. Em cada uma dessas categorias, faz-se uma estimativa do custo de produção de um único exemplar de cada produto. Faz-se, também, uma estimativa da quantidade de cada produto a ser fabricado por trimestre. Essas estimativas são dadas na **tabela 1** e na **tabela 2**. A empresa gostaria de apresentar a seus acionistas uma única tabela mostrando o custo total por **trimestre** de cada uma das três categorias: **matéria- prima**, **pessoal** e **despesas gerais**.

*Tabela 3. Custo de Produção por Item (em reais)*
|**Gastos**     |**Produto A**|**Produto B**|**Produto C**|
|:-------------:|:-----------:|:-----------:|:-----------:|
|Matéria-prima  |0.10         |0.30         |0.15         |
|Pessoal        |0.30         |0.40         |0.25         |
|Despesas-gerais|0.10         |0.20         |0.15         |

*Tabela 4. Quantidade Produzida por Trimestre*
|**Produto**|**Verão**|**Outono**|**Inverno**|**Primavera**|
|:---------:|:-------:|:--------:|:---------:|:-----------:|
|A          |4000     |4500      |4500       |4000         |
|B          |2000     |2600      |2400       |2200         |
|C          |5800     |6200      |6000       |6000         |

*Resposta*
Você deve obter uma matriz cujas linhas representam as categorias e as colunas os trimestres e que tenham os valores abaixo:

| **Custos por Categoria por Trimeste**|**Verão** | **Outono** | **Inverno**| **Primavera** |
|:---------------------------------:|:-------:|:--------:|:---------:|:-----------:|
| Matéria-prima                      |1870     |2160      |2070       |1960    |
| Pessoal                            |3450     |3940      |3810       |3580    |
| Despesas Gerais                    |1670     |1900      |1830       |1740    |






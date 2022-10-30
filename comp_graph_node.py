import numpy as np

'''
    Classe Node:
    - Armazena informação sobre um nó dentro do grafo computacional
    - Cada nó corresponde a um valor escalar, que pode ser obtido por meio de uma operação arbitrariamente complexa
    - Guarda informações acerca dos gradientes para que seja feito o processo de backpropagation

    - Cada nó possui um valor escalar e ele pode conter filhos que representam as operações que realiza
    - Além disso, cada nó possui um identificador de operação que foi feito para resultar nele
'''
class Node:
    
    def __init__(self, scalar, _children_components=(), _math_operator=''):

        self.scalar = scalar                 # Valor escalar
        self._math_operator = _math_operator # operação matemática fica para produzir o nó atual
        
        self.gradient = 0                    # Gradiente que é utilizado no backprop  
        self._backward = lambda: None        # Define a derivada para ser computada no backpropagation
        
        self._previous = set(_children_components) # armazena os nós subsequentes

    # Realiza a operação de adição 'a + b'
    def __add__(self, b):
        b = b if isinstance(b, Node) else Node(b)
        res = Node(self.scalar + b.scalar, (self, b), '+')

        # Define a operação de backward propagation este nó
        # r = a + b
        # dr/da e dr/db = 1
        # utilizando a regra da cadeia para L: dL / da = dL/dr x dr/da, assim dL/da = dr/da
        # considerando r como o output dessa operação, teremos gradient = dr/da
        def _backward():   
            self.gradient += res.gradient
            b.gradient    += res.gradient
        
        res._backward = _backward

        return res
    
    # Realiza a operação de multiplicação
    def __mul__(self, b):
        b = b if isinstance(b, Node) else Node(b)
        res = Node(self.scalar * b.scalar, (self, b), '*')

        # Define a operação de backward propagation para este nó
        # r = a * b
        # dr/da = b e dr/db = a (cruzada)
        # Utilizando a regra da cadeia para L: dL/da = dL/dr x dr/da, assim dL/da = dL/dr x b
        # Portanto, multiplicamos o upstream gradiente (gradiente propagado pela camada mais acima)
        # pelo gradiente atual resultante da multiplicação (derivadas cruzadas)
        def _backward():
            self.gradient += b.scalar * res.gradient
            b.gradient    += self.scalar * res.gradient
        
        res._backward = _backward

        return res
    
    # Realiza a exponenciação 
    # r = a**b - considerando b como um valor inteiro ou float
    # dr/da = ba^{b-1}
    # Da mesma forma, teremos que realizar a regra da cadeia, multiplicando pelo upstream gradient
    def _pow__(self, b):
        assert isinstance(b, (int, float)), "Apenas exponenciação de valores inteiros ou flutuantes"
        res = Node(self.scalar ** b, (self, ), f'**{b}')

        def _backward():
            self.gradient += (b * self.scalar ** (b - 1)) * res.gradient
        
        res._backward = _backward

        return res

    ### Operações derivadas da soma e multiplicação
    def __neg__(self): # Negação
        return self * (-1)
    
    def __radd__(self, b): # Alternativa para soma b + a
        return self + b
    
    def __rmul__(self, b): # Alternativa para a multiplicação b * a
        return self * b
    
    def __sub__(self, b): # subtração a - b
        return self + (- b)
    
    def __rsub__(self, b): # subtração b - a
        return b + (-self)
    
    def __truediv__(self, b): # divisão a / b
        return self * b **( -1)
    
    def __rtruediv__(self, b): # divisão b / a
        return b * self **( -1)
    
    def __gt__(self, other):
        return self.scalar > other.scalar
    
    def __lt__(self, other):
        return self.scalar < other.scalar
    
    def __ge__(self, other):
        return self.scalar >= other.scalar
    
    def __le__(self, other):
        return self.scalar <= other.scalar

    ### Representação da estrutura/objeto Node
    def __repr__(self):
        return f"Node(scalar={self.scalar}, gradient={self.gradient})"
    
    ### Operador/Função de ativação

    # Função de ativação ReLU - max(0, x)
    def relu(self):
        res = Node(0 if self.scalar < 0 else self.scalar, (self, ), 'ReLU')

        def _backward():
            self.gradient += (res.scalar > 0) * res.gradient
        
        res._backward = _backward

        return res

    def sigmoid(self):
        res = Node(1/(1 + np.exp(-self.scalar)), (self,), 'Sigmoid')

        def _backward():
            self.gradient += res.scalar * (1 - res.scalar)
        
        res._backward = _backward
        
        return res

    # Função de softmax - Recebe um valor e uma lista de outros nós
    def softmax(self, x):
        sum_softmax = 0
        for xi in x:
            sum_softmax = np.exp(xi.scalar)

        softmax_score = np.exp(self.scalar) / sum_softmax

        return softmax_score


    # Baseando-se em um nó podemos realizar o backpropagation dele para os outros elementos
    # Para isso, iremos seguir a ordem contrária com que o grafo computacional é construído
    def backward_propagation(self):
        topological_sorted = []
        visited = set()
        
        def build_topological_sort(i):
            if i not in visited:
                visited.add(i)
                for child in i._previous:
                    build_topological_sort(child)
                topological_sorted.append(i)
        
        build_topological_sort(self)

        self.gradient = 1 # Gradiente incia-se como 1 (dL/dL = 1)
        for node in reversed(topological_sorted):
            node._backward() # Calcula o gradiente para cada nó de maneira reversa

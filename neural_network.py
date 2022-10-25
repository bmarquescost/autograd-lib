import random
from comp_graph_node import Node

'''
    Classe Module
    - Possui códigos base para a rede neural
        - Zerar os gradientes dos parêmetros para próximos passos do backpropagation
'''
class Module:
    def clear_gradient(self):
        for param in self.parameters():
            param.gradient = 0
    
    def parameters(self):
        return []

'''
    Classe Neuron
    - Corresponde a um elemento de neurônio da rede neural
    - Para cada valor de entrada, multiplica por um peso
    - Todos os valores são somados em conjunto com um viés 
'''
class Neuron(Module):
    def __init__(self, n_inputs, non_linear=True):
        self.w = [Node(random.uniform(-1,1)) for _ in range(n_inputs)] # learnable parameters do neurônio
        self.b = Node(0) # viés b
        self.non_linear = non_linear

    # Chamada Neuron(valor)
    # Realiza função de ativação em cima da somatória ponderada dos inputs pelos pesos
    def __call__(self, x):
        activation = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return activation.relu() if self.non_linear else activation # aplica a função de ativação
    
    def parameters(self):
        return self.w + [self.b] # retorna a listagem com todos os parâmetros
    
    def __repr__(self):
        return f"ReLU - Neurônio {len(self.w)}"


'''
    Classe Layer
    - Classe que concentra um stacking de neurônios
'''
class Layer(Module):
    def __init__(self, n_inputs, n_output, **kwargs):
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in range(n_output)]

    def __call__(self, x):
        res = [neuron(x) for neuron in self.neurons]
        return res[0] if len(res) == 1 else res 
    
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer ({','.join(str(neuron) for neuron in self.neurons)})"


'''
    Classe MLP - Multilayer Perceptron
'''
class MLP(Module):
    # n_outputs corresponde a todas as camadas intermediárias e a camada final
    def __init__(self, n_inputs, n_outputs): 
        net_dim = [n_inputs] + n_outputs
        self.layers = [Layer(net_dim[i], net_dim[i+1], non_linear = i != len(n_outputs) - 1) for i in range(len(n_outputs))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self):
        return f"MLP ({', '.join(str(layer) for layer in self.layers)})"
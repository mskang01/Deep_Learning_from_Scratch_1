# Ch.2 Perceptrons (퍼셉트론)
import numpy as np

# AND Gate
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5]) # weight (가중치): determines the importance of each neurons
    b = -0.7 # bias: determines how easily the y neuron is activated
    tmp1 = np.sum(x * w) + b
    
    if tmp1 <= 0:
        return 0
    else:
        return 1
    
print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))


print("------------------------------")

# NAND Gate (AND와 결과값이 반대)

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    
    if tmp <= 0:
        return 0
    else:
        return 1
    
print(NAND(0,0))
print(NAND(1,0))
print(NAND(0,1))
print(NAND(1,1))

print("------------------------------")

# OR Gate
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])    # Note: AND,NAND,OR are determined by how w(weight), b(bias) are chosen
    b = - 0.2
    tmp2 = np.sum(x*w) + b
    
    if tmp2 <= 0:
        return 0
    else:
        return 1
    
print(OR(0,0))
print(OR(1,0))
print(OR(0,1))
print(OR(1,1))

print("------------------------------")

# XOR Gate (x1,x2 하나만 1일 때 1 출력)
# XOR is nonlinear -> multi-layer perceptron is needed

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND (s1,s2)
    return y

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))
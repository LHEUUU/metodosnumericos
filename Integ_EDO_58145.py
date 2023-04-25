import math
import matplotlib.pyplot as plt
import numpy as np

#Exercício 1a)

def V(h):
    return 32 - h**2 * (6 -h)

def trapezio(a,b,n):
    h = (b-a)/n
    soma_h = V(a)+V(b)
    
    for i in range(1,n):
        soma_h += 2*V(a+(i*h))
    
    I = (h/2)*soma_h
    
    return I

def simpson(a,b,n):
    h = (b-a)/n
    soma_h = V(a)+V(b)
    soma_2h = 0
    
    for i in range(1,n,2):
        soma_h += 4*V(a+i*h)
        
    for i in range(2,n-1,2):
        soma_h += 2*V(a+i*h)
        
    soma_h *= h/3
    
    soma_2h *= 2*h/3
    
    erro = abs(soma_h - soma_2h)/15
    
    return soma_h

print("Exercício 1a):")
print("Valor do trabalho (método do Trapézio), em J:", ((-9810*math.pi)/3)*trapezio(4,2,40))
print("Valor do trabalho (método de Simpson), em J:", ((-9810*math.pi)/3)*simpson(4,2,40))


#Exercício 1b)

xaxis_t=[]
desvio_t=[]
xaxis_s=[]
desvio_s=[]

for i in range (1,41):
    xaxis_t.append(i)
    desvio_t.append(abs(trapezio(4,2,i)+12))
    
for i in range (2,41,2):
    xaxis_s.append(i)
    desvio_s.append(abs(simpson(4,2,i)+12))
        
plt.plot(xaxis_t, desvio_t,'.-')
plt.plot(xaxis_s, desvio_s,'.-')
plt.title("Desvio do integral numérico ao valor real em função do nº de divisões", fontsize=13)
plt.xlabel("Nº de divisões", fontsize=12)
plt.ylabel("Desvio do integral numérico ao valor real", fontsize=12)
plt.legend(["Método do Trapézio", "Método de Simpson"], loc="best", fontsize=12)
plt.grid()
plt.show()


#Exercício 2

dados = [0, 37, 71, 104, 134, 161, 185, 207, 225, 239, 250]

def trapezio_dados(a,b,data):
    n = len(data)
    h = (b-a)/(n-1)
    soma_h = data[0] + data[n-1]
    
    for i in range(1,n-1):
        soma_h += 2*data[i]
        
    I = (h/2)*soma_h
    
    return I

print("\nExercício 2):")
print("O valor do trabalho é, em J:", trapezio_dados(0,0.5,dados))
print("A velocidade de saída da flecha é, em m/s:", math.sqrt((2/0.075)*trapezio_dados(0,0.5,dados)))

#Exercício 3a)  

def f(x,y):
    return -2.3*y

def euler(f, x0, xmax, y0, h):
    n = int(math.ceil((xmax-x0)/h))
    y = [y0]
    x = [x0]
    
    for i in range(n):
        y.append(y[i] + f(x[i],y[i])*h)
        x0 += h
        x.append(x0)
    
    return x,y

graf1 = euler(f,0,5,1,0.5)
graf2 = euler(f,0,5,1,0.7)
graf3 = euler(f,0,5,1,1)

t=np.linspace(0,5,1000)

plt.plot(graf1[0], graf1[1],'.-')
plt.plot(graf2[0], graf2[1],'.-')
plt.plot(graf3[0], graf3[1],'.-')
plt.plot(t, np.exp(-2.3*t))
plt.title("Soluções de N(t) em função de t horas para diferentes h", fontsize=13)
plt.xlabel("Tempo (t) em horas", fontsize=12)
plt.ylabel("Densidade de núcleos radioativos N(t)", fontsize=12)
plt.legend(["h = 0.5", "h = 0.7", "h = 1", "Curva analítica"], loc="best", fontsize=12)
plt.xticks(np.arange(0, 6, 0.5))
plt.grid()
plt.show()


#Exercício 3b)

def RK4(f, x0, xmax, y0, h):
    n = int(math.ceil((xmax-x0)/h))
    y = [y0]
    x = [x0]
    
    for i in range(n):
        k1 = f(x0, y[i])
        k2 = f(x0+h/2, y[i]+k1*h/2)
        k3 = f(x0+h/2, y[i]+k2*h/2)
        k4 = f(x0+h, y[i]+k3*h)
        
        y.append(y[i] + (h/6)*(k1+2*k2+2*k3+k4))
        x0 += h
        x.append(x0)
        
    return x,y

h = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
desvio_RK4 = []
desvio_euler = []

for h0 in h:
    desvio_euler.append(abs(math.exp(-2.3*5) - euler(f,0,5,1,h0)[1][-1]))
    desvio_RK4.append(abs(math.exp(-2.3*5) - RK4(f,0,5,1,h0)[1][-1]))
    
plt.plot(h, desvio_euler, '.-')   
plt.plot(h, desvio_RK4, '.-')
plt.title("Desvio ao valor analítico em função do tamanho do passo", fontsize=13)
plt.xlabel("Tamanho do passo (h)", fontsize=12)
plt.ylabel("Desvio ao valor analítico", fontsize=12)
plt.legend(["Euler", "Runge-Kutta 4ºOrdem"], loc="best", fontsize=12)
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.show()


#Exercício 4

def fx(x,y):
    return 20*math.cos(math.pi/4)

def fy(x,y):
    return 20*math.sin(math.pi/4) - 9.81*x

t=np.linspace(0,5,1000)

plt.plot(euler(fx,0,5,0,0.5)[1], (euler(fy,0,5,0,0.5)[1]), '.-')
plt.plot(RK4(fx,0,5,0,0.5)[1], (RK4(fy,0,5,0,0.5)[1]), '.-')
plt.plot(20*math.cos(math.pi/4)*t, 20*math.sin(math.pi/4)*t -(9.81/2)*t**2)
plt.title("Trajetória do projétil", fontsize=13)
plt.xlabel("Posição x(m)", fontsize=12)
plt.ylabel("Posição y(m)", fontsize=12)
plt.legend(["Euler","Runge-Kutta 4ºOrdem","Curva analítica"], loc="best", fontsize=12)
plt.ylim(0,20)
plt.xlim(0,50)
plt.xticks(np.arange(0,50,5))
plt.grid()
plt.show()
import math
import matplotlib.pyplot as plt
import numpy as np

#Exercício 1a)

def A1(R3,R2): 
    return [[1, -1, -1],
            [0, R2, -1],
            [-R3, 0, -1]]
    
B1 = [0, -2, -7]


def subs_inv(A, B):
    n = len(B)
    a = [[A[i][j] for j in range(n)] for i in range(n)] 
    b = [B[i] for i in range(n)]
    
    x=[0 for i in range(n)]
    x[n-1]= b[n-1]/a[n-1][n-1]
    
    for i in range(n-2, -1, -1):
        soma = 0

        for j in range(i,n):
            soma += a[i][j]*x[j]
        
        x[i] = (b[i] - soma)/a[i][i]
        
    return x

print("Exercício 1a):")
print ("Para R3 = 0Ω, temos:", subs_inv(A1(0,2),B1))
print ("Para R3 = 2Ω, temos:", subs_inv(A1(2,2),B1))


#Exercício 1b)

def gauss_sem_pivot(A,B):
    n = len(B)
    a = [[A[i][j] for j in range(n)] for i in range(n)] 
    b = [B[i] for i in range(n)]

    for k in range(0,n-1):
        for i in range(k+1,n):
            m = a[i][k]/a[k][k]
            
            for j in range(k,n):
                a[i][j] -= m*a[k][j]
                
            b[i] -= m*b[k]
    
    return subs_inv(a,b)

print("\nExercício 1b):")
print("Método de eliminação de Gauss sem pivot:")
print ("Para R3 = 0Ω, temos:", gauss_sem_pivot(A1(0,2),B1))
print ("Para R3 = 2Ω, temos:", gauss_sem_pivot(A1(2,2),B1))
#print ("Para R3 = 2Ω e R2 = 0Ω, temos:", gauss_sem_pivot(A1(2,0),B1))  #erro valor dividido por 0

def gauss_com_pivot(A,B):
    n = len(B)
    a = [[A[i][j] for j in range(n)] for i in range(n)] 
    b = [B[i] for i in range(n)]
    
    for k in range(0,n-1):
        max = k
        
        for i in range(k+1,n): 
            if abs(a[i][k]) > abs(a[max][k]):
                max = i
                
        if max != k:
            for i in range(k,n):
                temp = a[k][i]
                a[k][i] = a[max][i]
                a[max][i] = temp
                
            temp = b[k]
            b[k] = b[max]
            b[max] = temp
    
        for i in range(k+1,n):
            m = a[i][k]/a[k][k]
            
            for j in range(k,n):
                a[i][j] -= m*a[k][j]
                    
            b[i] -= m*b[k]
        
    return subs_inv(a,b)

print("Método de eliminação de Gauss com pivot:")
print ("Para R3 = 2Ω e R2 = 0Ω, temos:", gauss_com_pivot(A1(2,0),B1))


#Exercício 1d)

v2=[]
I1=[]
I2=[]
I3=[]
c1=[]
c2=[]
c3=[]

for v in range(-10,11):
    v2.append(v)
    I1.append(gauss_com_pivot(A1(2,2),[0,-v,-v-5])[0])
    I2.append(gauss_com_pivot(A1(2,2),[0,-v,-v-5])[1])
    I3.append(gauss_com_pivot(A1(2,2),[0,-v,-v-5])[2])
    
    c1.append((1/8) * (15 + 2 * v))
    c2.append((1/8) * (5 - 2 * v))
    c3.append((1/4) * (5 + 2 * v))

plt.plot(v2,I1)
plt.plot(v2,I2)
plt.plot(v2,I3)
plt.scatter(v2,c1)
plt.scatter(v2,c2)
plt.scatter(v2,c3)
plt.xlabel("Valores de V2(V)", fontsize=12)
plt.ylabel("Valores de I(A)", fontsize=12)
plt.title("I1, I2 e I3 em função de V2", fontsize=13)
plt.legend(["I1 (Python)", "I2 (Python)", "I3 (Python)", "I1 (Mathematica)", "I2 (Mathematica)", "I3 (Mathematica)"], loc="best", ncol= 2, fontsize=11)
plt.grid()
plt.show()


#Exercício 2a)

#k_max = 200 e e= 10^-4

A2 = [[-5, 3, 0, 0, 0],
      [1, -2, 1, 0, 0],
      [0, 1, -2, 1, 0],
      [0, 0, 3, -6, 3],
      [0, 0, 0, 3, -5]]
      
    
B2 = [-80, 0, 0, 60, 0]

def gauss_seidel_sr(a,b,e,k_max):
    n = len(b)
    x=[0 for i in range(n)]
    k = 0
    erro_max = 1

    while abs(erro_max) > e and k < k_max:
        for i in range (0,n):
            x_anterior = x[i]
            soma_antes = sum([a[i][j]*x[j] for j in range (0,i)])
            soma_depois = sum([a[i][j]*x[j] for j in range (i+1,n)])
            x[i] = (b[i] - soma_antes - soma_depois)/a[i][i]
            erro_max = abs((x[i]-x_anterior)/x[i])

        k += 1

    return x

print("\nExercício 2a):")
print ("Valores de x:", gauss_seidel_sr(A2,B2,math.pow(10,-4),200))


#Exercício 2b)

def gauss_seidel_cr(a,b,lamb,e,k_max):
    n = len(b)
    x=[0 for i in range(n)]
    k = 0
    erro_max = 1
    
    lista_k = []
    lista_erro = []
    
    while abs(erro_max) > e and k < k_max:
        for i in range (0,n):
            x_anterior = x[i]
            soma_antes = sum([a[i][j]*x[j] for j in range (0,i)])
            soma_depois = sum([a[i][j]*x[j] for j in range (i+1,n)])
            x[i] = lamb*(b[i] - soma_antes - soma_depois)/a[i][i] + (1-lamb)*x[i]
            erro_max = abs((x[i]-x_anterior)/x[i])
        
        k += 1
        lista_k += [k]
        lista_erro += [erro_max]
    
    return x, lista_k, lista_erro

graf1 = gauss_seidel_cr(A2,B2,0.5,math.pow(10,-4),200)
graf2 = gauss_seidel_cr(A2,B2,1,math.pow(10,-4),200)
graf3 = gauss_seidel_cr(A2,B2,1.2,math.pow(10,-4),200)
graf4 = gauss_seidel_cr(A2,B2,2,math.pow(10,-4),200)

plt.plot(graf1[1], graf1[2])
plt.plot(graf2[1], graf2[2])
plt.plot(graf3[1], graf3[2])
plt.plot(graf4[1], graf4[2])
plt.yscale("log")
plt.title("Precisão do método em função do nº de interações", fontsize=13)
plt.xlabel("Número de iterações", fontsize=12)
plt.ylabel("Valor da precisão", fontsize=12)
plt.legend(["λ = 0.5", "λ = 1", "λ = 1.2", "λ = 2"], loc="best", fontsize=12)
plt.grid()
plt.show()


#Exercício 3a)

#a 1 função é um círculo e a 2 é uma parabola

angulo = np.linspace(0,2*np.pi,1000)
raio = math.sqrt(5)

x1 = raio * np.cos(angulo)
y1 = raio * np.sin(angulo)

x2 = np.linspace(-3,3,1000)

def y(x):
    return x**2 - 1

plt.plot(x2, y(x2))
plt.plot(x1, y1)
plt.title("Representação gráfica do sistema", fontsize=13)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.legend(["y + 1 = x^2", "x^2 = 5 - y^2"], loc="best", fontsize=12)
plt.ylim(-4, 4)
plt.xlim(-5, 5)
plt.grid()
plt.show()


#Exercício 3b)

def newton(p,e,k_max):
    [x,y] = p
    n = len(p)
    erro_max = 1
    k = 0
    
    while abs(erro_max) > e and k < k_max:
        [x,y] = p
        J = [[2*x, 2*y], [-2*x, 1]]
        F = [(x**2 - 5 + y**2), (y + 1 - x**2)]
        
        d = gauss_com_pivot(J,F)
        
        for i in range(0,n):
            p[i] -= d[i]
            erro_max = abs(d[i]/p[i])
        
        k += 1
        
    return p

#valores iniciais [x0 =1,y0=1],[x0 =-2,y0=1]

print("\nExercício 3b):")
print("Para o valor inicial [1,1], obteve-se:", newton([1,1],math.pow(10,-4),200))
print("Para o valor inicial [-2,1], obteve-se:", newton([-2,1],math.pow(10,-4),200))
import math
import matplotlib.pyplot as plt

#Exercício 1a)

def f(x):
    return math.pow(x,2) - 4

def bis(a,b,e):
    fa = f(a)
    fb = f(b)
    
    d = (b-a)/2
    x = a + d
    
    nint = 0
    lista_nit = []   #guarda o nº de iterações
    listax = []      #guarda o valor de x para certa iteração
    listad = []      #guarda o valor de d para cada iteração - exercício 1d)
    
    while d > e:
        fx = f(x)
        
        if fx * fa > 0:
            a = x
            fa = fx
        else:
            b = x
            fb = fx
            
        d = (b - a)/2
        x = a + d
        nint += 1
        
        listax += [x]
        lista_nit += [nint]
        listad += [d]

    return [lista_nit, listax, listad, x]

print("x I1 =", bis(0.7, 2.6, math.pow(10, -5))[3])
print("x I2 =", bis(0.4, 1.7, math.pow(10, -5))[3])
print("x I3 =", bis(-3, 0.6, math.pow(10, -5))[3])

#Exercício 1b)

grafico1 = bis(0.7, 2.6, math.pow(10, -5))
grafico2 = bis(0.4, 1.7, math.pow(10, -5))
grafico3 = bis(-3, 0.6, math.pow(10, -5))

plt.scatter(grafico1[0], grafico1[1])
plt.scatter(grafico2[0], grafico2[1])
plt.scatter(grafico3[0], grafico3[1])
plt.plot(grafico1[0], grafico1[1])
plt.plot(grafico2[0], grafico2[1])
plt.plot(grafico3[0], grafico3[1])
plt.title("Valor estimado de x em função do nº de interações")
plt.xlabel("Número de iterações", fontsize=12)
plt.ylabel("Valor de x", fontsize=12)
plt.xticks(range(1, 19))
plt.legend(["Int1 = [0.7, 2.6]", "Int2 = [0.4, 1.7]", "Int3 = [-3, 0.6]"], loc="center right", fontsize=12)
plt.show()

#Exercício 1c)

#x0 = 0.4
#x1 = 0.5

def df(x):
    return 2*x

def new(x0, e, k_max):
    f0 = f(x0)
    k = 0
    d = f0 / df(x0)
    
    lista_nit = []   #guarda o nº de iterações = k
    listad = []      #guarda o valor de d para cada iteração
    
    while abs(d) > e and k < k_max:
        x0 -= d
        f0 = f(x0)
        d = f0 / df(x0)
        k += 1
        
        listad += [abs(d)]
        lista_nit += [k]
    
    return [lista_nit, listad]

def sec(x0, x1, e, k_max):
    f0 = f(x0)
    f1 = f(x1)
    k = 0
    d = f1 * (x1 - x0) / (f1 -f0)
    
    lista_nit = []   #guarda o nº de iterações = k
    listad = []      #guarda o valor de d para cada iteração
    
    while abs(d) > e and k < k_max:
        x2 = x1 - d
        x0 = x1
        x1 = x2
        f0 = f1
        f1 = f(x1)
        d = f1 * (x1 -x0) / (f1 - f0)
        k += 1
        
        listad += [abs(d)]
        lista_nit += [k]
    
    return [lista_nit, listad]

#Exercício 1d)

grafico_bis = bis(0.4, 0.5, math.pow(10, -5))
grafico_new = new(0.4, math.pow(10, -5), 20)       #assumi k_max = 20
grafico_sec = sec(0.4, 0.5, math.pow(10, -5), 20)  #assumi k_max = 20

plt.scatter(grafico_bis[0], grafico_bis[2])
plt.scatter(grafico_new[0], grafico_new[1])
plt.scatter(grafico_sec[0], grafico_sec[1])
plt.plot(grafico_bis[0], grafico_bis[2])
plt.plot(grafico_new[0], grafico_new[1])
plt.plot(grafico_sec[0], grafico_sec[1])
plt.yscale("log")
plt.title("Erro em função do nº de interações")
plt.xlabel("Número de iterações", fontsize=12)
plt.ylabel("Valor do erro", fontsize=12)
plt.legend(["M. Bissecção", "M. Newton", "M. Secante"], loc="upper right", fontsize=12)
plt.show()


#Exercício 2

def i(t):
    return 9*math.exp(-t)*math.sin(2*math.pi*t) - 1.50

def di(t):
    return 18*math.pi*math.exp(-t)*math.cos(2*math.pi*t) - 9*math.exp(-t)*math.sin(2*math.pi*t)

def New(x0, e, k_max):
    i0 = i(x0)
    k = 0
    d = i0 / di(x0)
    
    while abs(d) > e and k < k_max:
        x0 -= d
        i0 = i(x0)
        d = i0 / di(x0)
        k += 1
    
    return x0     #x0 = t

print("Valor 1 de t:", New(0.6, math.pow(10,-6), 20))     #assumi k_max = 20
print("Valor 2 de t:", New(0.7, math.pow(10,-6), 20))
#print("Valor 3 de t:", New(0.75, math.pow(10,-6), 20))   #esta linha de código resulta num erro de Overflow, pq chega a um ponto o declive é 0
print("Valor 4 de t:", New(0.8, math.pow(10,-6), 20))
print("Valor 5 de t:", New(0.9, math.pow(10,-6), 20))


#Exercício 3a)

def p(x):
    return 0.5*math.pow((x-2),2)

def num_ouro(a, b, e):
    phi = (1 + math.sqrt(5)) / 2
    x0 = b - (b-a)/phi
    x1 = a + (b-a)/phi
    
    while abs(b-a) / abs(x0+x1) > e:
        if p(x0) < p(x1):
            b = x1
        else:
            a = x0
        
        x0 = b - (b-a)/phi
        x1 = a + (b-a)/phi
        
    return (b+a)/2

print("Posição de equilíbrio do corpo:", num_ouro(-0.7, 2.6, 0.00001))  #e = 0,001% =0,00001
print("Posição de equilíbrio do corpo:", num_ouro(0.4, 1.7, 0.00001))


#Exercício 3b)
        
def dp(x):
    return (x-2)

def grad(x0, k_max, lamb, e):
    d = dp(x0)
    x1 = x0 - lamb * d
    k = 0
    
    lista_nit = []   #guarda o nº de iterações = k
    lista_x1 = []    #guarda os valores do minimo da função para cada iteração
    
    while abs(lamb * d) > e and k < k_max:
        x0 = x1
        d = dp(x0)
        x1 = x0 - lamb * d
        k += 1
        
        lista_nit += [k]
        lista_x1 += [x1]
        
    return [lista_nit, lista_x1]


#Exercício 3c)

graf1 = grad(0, 10, 0.1, math.pow(10, -5))
graf2 = grad(0, 10, 0.5, math.pow(10, -5))
graf3 = grad(0, 10, 1, math.pow(10, -5))
graf4 = grad(0, 10, 2, math.pow(10, -5))
graf5 = grad(0, 10, 2.1, math.pow(10, -5))

plt.scatter(graf1[0], graf1[1])
plt.scatter(graf2[0], graf2[1])
plt.scatter(graf3[0], graf3[1])
plt.scatter(graf4[0], graf4[1])
plt.scatter(graf5[0], graf5[1])
plt.plot(graf1[0], graf1[1])
plt.plot(graf2[0], graf2[1])
plt.plot(graf3[0], graf3[1])
plt.plot(graf4[0], graf4[1])
plt.plot(graf5[0], graf5[1])
plt.title("Valores do mínimo em função do número de iterações para cada λ")
plt.xlabel("Número de iterações", fontsize=12)
plt.ylabel("Valor do mínimo", fontsize=12)
plt.legend(["λ = 0.1", "λ = 0.5", "λ = 1", "λ = 2", "λ = 2.1"], loc="best", ncol= 3)
plt.show()


#Exercício 4a)

#x0 = 2 e lambda = 0.5

def u(r):
    return 80*math.exp(-2*r)-10/r

def du(r):
    return -160*math.exp(-2*r) + 10/math.pow(r,2)

def gradiente(x0, k_max, lamb, e):
    d = du(x0)
    x1 = x0 - lamb * d
    k = 0
    
    while abs(lamb * d) > e and k < k_max:
        x0 = x1
        d = du(x0)
        x1 = x0 - lamb * d
        k += 1
    
    return x1

print("Distância de equilíbrio:", gradiente(2, 10, 0.5, math.pow(10,-5)))


#Exercício 4b) 

#lambda = 0.25

def u2d(x,y):
    return 80*math.exp(-2*(math.sqrt(x*x + y*y)))-10/(math.sqrt(x*x + y*y))

def dx_u2d(x,y):
    return -160*x*math.exp(-2*math.sqrt(x*x + y*y))/(math.sqrt(x*x + y*y)) + 10*x/(math.sqrt(math.pow(x*x + y*y, 3)))

def dy_u2d(x,y):
    return -160*y*math.exp(-2*(math.sqrt(x*x + y*y)))/(math.sqrt(x*x + y*y)) + 10*y/(math.sqrt(math.pow(x*x + y*y, 3)))

def gradiente2D(x0, y0, k_max, lamb, e):
    dx = dx_u2d(x0,y0)
    dy = dy_u2d(x0,y0)
    x1 = x0 - lamb*dx
    y1 = y0 - lamb*dy
    k = 0
    
    lista_x1 = []   #guarda os valores de x1 para cada iteração
    lista_y1 = []   #guarda os valores de y1 para cada iteração
    lista_nit = []  #guarda o nº de iterações = k
    
    while abs(lamb*math.sqrt(dx*dx + dy*dy)) > e and k < k_max:
        x0 = x1
        y0 = y1
        dx = dx_u2d(x0,y0)
        dy = dy_u2d(x0,y0)
        x1 = x0 - lamb*dx
        y1 = y0 - lamb*dy
        k += 1
        
        lista_x1 += [x1]   
        lista_y1 += [y1]
        lista_nit += [k]
    
    return [lista_nit, lista_x1, lista_y1, x1, y1]

print("Mínimo (x,y):", gradiente2D(5, -5, 200, 0.25, math.pow(10,-5))[3], gradiente2D(5, -5, 200, 0.25, math.pow(10,-5))[4])  #assumi k_max = 200 de modo a obter valores mais precisos

graf2d = gradiente2D(5, -5, 200, 0.25, math.pow(10,-5))  #assumi k_max = 200 de modo a obter valores mais precisos

fig, ax1 = plt.subplots()

#gráfico maior:
ax1.plot(graf2d[0], graf2d[1])
ax1.plot(graf2d[0], graf2d[2])
ax1.set_title("Valores de x e de y em função do número de iterações")
ax1.set_xlabel("Número de iterações", fontsize=12)
ax1.set_ylabel("Valores de x e y", fontsize=12)
ax1.legend(["x", "y"], loc="upper right", fontsize=12)

#gráfico pequeno inserido:
left, bottom, width, height = [0.25, 0.35, 0.3, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])

ax2.plot(graf2d[1], graf2d[2])
ax2.set_title("Trajetória y(x)")
ax2.set_xlabel("Valores de x", fontsize=12)
ax2.set_ylabel("Valores de y", fontsize=12)

plt.show()
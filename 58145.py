import math
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

#Exercício 1a)

f = open('C:\\Users\\leoma\\OneDrive\\Ambiente de Trabalho\\Uni\\2º Ano\Métodos Numéricos\\Aulas Práticas\\E4\\allwithoutnames.txt', 'r')

Obama = []
MacCain = []
votos = []

for x in f:
    votos.append(x.split())
    
for i in votos:
    Obama.append(float(i[0]))
    MacCain.append(float(i[1]))
    
def media(lista):
    return sum(lista)/len(lista)

def mediana(lista):
    n = len(lista)
    a = lista.copy()
    a.sort()
    
    if (n % 2) == 0:
        mediana1 = a[n//2]
        mediana2 = a[n//2 - 1]
        Mediana = (mediana1 + mediana2) / 2
        
    else:
        Mediana = a[n//2]
        
    return Mediana

def variancia(lista):
    soma = 0
    
    for i in lista:
        soma += (i- media(lista))**2
        
    return soma/(len(lista))

print("Exercício 1a):")
print("Votos Obama: \n Média: %f \n Mediana: %f \n Variância: %f" %(media(Obama), mediana(Obama), variancia(Obama)))
print("Votos MacCain: \n Média: %f \n Mediana: %f \n Variância: %f" %(media(MacCain), mediana(MacCain), variancia(MacCain)))


#Exercício 1b)

A_pos = []
A_neg = []
B_pos = []
B_neg = []

for i in range(len(Obama)):
    total = Obama[i] + MacCain[i]
    vo = Obama[i]/total
    vm = MacCain[i]/total
    
    if Obama[i] > MacCain[i]:
        A_pos.append(vo-vm)
    else:
        A_neg.append(vo-vm)
    
    if total > 20000:
        if Obama[i] > MacCain[i]:
            B_pos.append(vo-vm)
        else:
            B_neg.append(vo-vm)
    
fig, axs = plt.subplots(1, 2, tight_layout=True)
axs[0].hist(A_pos,bins=8,edgecolor='black',label='Obama',color='green')
axs[0].hist(A_neg,bins=9,edgecolor='black',label='McCain',color='darkblue')
axs[1].hist(B_pos,bins=[0, 0.1,0.2,0.3,0.4,0.5,0.6],edgecolor='black',color='green')
axs[1].hist(B_neg,[ -0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0],edgecolor='black',color='darkblue')
axs[0].set_title('Todos os condados')
axs[1].set_title('Condados com mais de 20 mil eleitores')
axs[0].set_xlim(-1,1)
axs[1].set_xlim(-1,1)
axs[0].set_ylim(0,500)
axs[1].set_ylim(0,190)
axs[1].set_yticks(np.arange(0,176,50))
axs[0].set_yticks(np.arange(0,501,50), minor=True)
axs[1].set_yticks(np.arange(0,176,25), minor=True)
axs[0].set_xticks(np.arange(-1,1.1,0.25), minor=True)
axs[1].set_xticks(np.arange(-1,1.1,0.25), minor=True)
axs[0].tick_params(left=True, right=True,which='both',direction='in')
axs[1].tick_params(left=True, right=True,which='both',direction='in')
fig.supylabel('Frequência absoluta', fontsize=12)
fig.supxlabel('Diferença relativa de votos', fontsize=12)
fig.legend(fontsize=10,bbox_to_anchor=(0.96,0.9))
plt.show()


#Exercício 2a)

f = open('C:\\Users\\leoma\\OneDrive\\Ambiente de Trabalho\\Uni\\2º Ano\Métodos Numéricos\\Aulas Práticas\\E4\\spots.txt', 'r')

ano = []
mes = []
num_Wolf = []
spots = []
auto_corr = []
eixo_x = []

for x in f:
    spots.append(x.split())
    
for i in spots:
    ano.append(float(i[0]))
    mes.append(float(i[1]))
    num_Wolf.append(float(i[2]))
    
N = len(num_Wolf)
    
for k in range(N):
    cov = sum([(num_Wolf[t] - media(num_Wolf))*(num_Wolf[t+k] - media(num_Wolf)) for t in range(N - k)])
    auto_corr.append((1/(N-k) * cov)/ variancia(num_Wolf))
    eixo_x.append(k)
    
plt.plot(eixo_x, auto_corr)
plt.xticks(np.arange(0,3100,250))
plt.title("Função de autocorrelação do número de Wolf", fontsize=13)
plt.ylabel("Valor de autocorrelação", fontsize=12)
plt.xlabel("Nº de meses", fontsize=12)
plt.minorticks_on()
plt.grid()
plt.show()


#Exercício 2b)

print("\nExercício 2b):\nPeriodicidade da série:", ((auto_corr.index(max(auto_corr[t] for t in range(2750,2820))))-auto_corr.index((max(auto_corr[t] for t in range(-1,5)))))/22)

#Exercício 2c)

dif_saz = []

for t in range(N):
    dif_saz.append(num_Wolf[t]-num_Wolf[t-127])

fig, axs = plt.subplots(3, 1, tight_layout=True)
axs[0].plot(eixo_x, num_Wolf, color='royalblue')
axs[0].set_title("Medição mensal do nº de Wolf")
axs[0].set_ylabel('Nº de Wolf')
axs[0].set_yticks(np.arange(0,201,100))
axs[0].grid()
axs[1].plot(eixo_x, dif_saz, color='darkorchid')
axs[1].set_title("Medição mensal do nº de Wolf com diferenças sazonais")
axs[1].set_ylabel('Nº de Wolf')
axs[1].set_yticks(np.arange(-200,201,100))
axs[1].grid()
axs[2].plot(eixo_x, auto_corr, color='red')
axs[2].set_title("Função de autocorrelação do nº de Wolf")
axs[2].set_ylabel('Autocorrelação')
axs[2].set_yticks(np.arange(-1,1.1,0.5))
axs[2].grid()
fig.supxlabel('Nº de meses decorridos')
plt.show()




#Exercício 3a)

metabolismo = [270, 82, 50, 4.8, 1.45, 0.97]
massa = [400, 70, 45, 2, 0.3, 0.16]

k1=sum([(np.log(metabolismo[i])-media(np.log(metabolismo)))*(np.log(massa[i])-media(np.log(massa))) for i in range(len(metabolismo))])
k2=sum([(np.log(massa[i])-media(np.log(massa)))**2 for i in range(len(metabolismo))])
k=k1/k2
a=media(np.log(metabolismo))-k*media(np.log(massa))  #a --> log(a)

print("\nExercício 3a):")
print("k = %f e log(a) = %f " %(k,a))


#Exercício 3b)

f=[]
for i in massa:
    f.append(np.exp(a)*i**k)

plt.plot(massa,metabolismo,".",label="Pontos dados")
plt.plot(massa,f,label="Função calculada")
plt.title('Gráfico de pontos dados e da função calculada',fontsize=13)
plt.xlabel("Massa (Kg)",fontsize=12)
plt.ylabel("Metabolismo (W)",fontsize=12)
plt.legend(loc='best',fontsize=12)
plt.grid()
plt.minorticks_on()
plt.show()

#Exercício 3c)

def deriv_dm(m0,b0,y,x):
    soma = 0
    for i in range(len(massa)):
        soma+=(y[i]-m0*x[i]-b0)*x[i]

    return -2*soma

def deriv_db(m0,b0,y,x):
    soma = 0
    for i in range(len(massa)):
        soma+=(y[i]-m0*x[i]-b0)

    return -2*soma

def gradiente(m0,b0,y,x,e,k_max,lamb):
    dm = deriv_dm(m0,b0,y,x)
    db = deriv_db(m0,b0,y,x)
    m1=m0-lamb*dm
    b1=b0-lamb*db
    k=0

    while abs(lamb*math.sqrt(dm*dm+db*db)) > e and k < k_max:
        m0=m1
        b0=b1
        dm=deriv_dm(m0,b0,y,x)
        db=deriv_db(m0,b0,y,x)
        m1=m0-lamb*dm
        b1=b0-lamb*db
        k=k+1

    return m1, b1

print("\nExercício 3c):\n Valores de k e log(a):")

for l in [0.01,0.05,0.1]:
    print(gradiente(0,0,np.log(metabolismo),np.log(massa),math.pow(10,-6),100,l)[0],gradiente(0,0,np.log(metabolismo),np.log(massa),math.pow(10,-6),100,l)[1])


#Exercício 4a)

rgb = io.imread('C:\\Users\\leoma\\OneDrive\\Ambiente de Trabalho\\Uni\\2º Ano\Métodos Numéricos\\Aulas Práticas\\E4\\rocks.jpg')
n_rgb = len(rgb)

cinzento = [[0 for i in range(len(rgb[0]))] for i in range(n_rgb)]

sum = 0

for i in range(n_rgb):
    for j in range(len(rgb[0])):
        cinzento[i][j] = (0.2126*rgb[i][j][0] + 0.7152*rgb[i][j][1] + 0.0722*rgb[i][j][2]) / 255
        sum += cinzento[i][j]
        
fig, axs = plt.subplots(1, 2, tight_layout=True)
axs[0].imshow(rgb)
axs[0].axis("off")
axs[1].imshow(cinzento, cmap=plt.cm.gray)
axs[1].axis("off")
axs[0].set_title('RGB',fontsize=13)
axs[1].set_title('Escala de cizentos',fontsize=13)
plt.show()


#Exercício 4b)

cinzento_original = []
thers = 0.228

for i in range(n_rgb):
    for j in range(len(rgb[0])):
        cinzento_original.append(cinzento[i][j])
        
        if cinzento[i][j] > thers:
            cinzento[i][j] = 1
        
        else:
            cinzento[i][j] = 0

fig, axs = plt.subplots(1, 2, tight_layout=True)
axs[0].hist(cinzento_original, 256)
axs[0].set_title('Histograma da intensidade da imagem em escala de cinzentos', fontsize=9.5)
axs[0].set_ylabel('Nº de pixéis', fontsize=11)
axs[0].set_xlabel('Valor da intensidade', fontsize=11)
axs[0].axvline(0.228, color="crimson")
axs[1].imshow(cinzento, cmap=plt.cm.gray)
axs[1].set_title('Imagem binária', fontsize=13)
axs[1].axis("off")
plt.show()


#Exercício 4c)

vazios = 0
total = 0

for i in cinzento:
    for j in i:
        if j == 0:
            vazios += 1
        total += 1
        
print("\nExercício 4c):\nPorosidade do material, n = %f" %(vazios/total * 100))
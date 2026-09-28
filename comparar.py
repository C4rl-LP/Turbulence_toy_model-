import time
import campos 
import os

# Comparar tempo dos campos 
xp=0.13
yp=0.41

nmax=9
t=0.0

Nrep=100

t0=time.perf_counter()

for _ in range(Nrep):
    campos.campo_total_podado(
        xp,yp,t,nmax
    )

t1=time.perf_counter()

tempo_podado=t1-t0

# -------- campo completo ----------
t0=time.perf_counter()

for _ in range(Nrep):
    campos.campo_total_correto(
        xp,yp,t,nmax
    )

t1=time.perf_counter()

tempo_full=t1-t0


# -------- campo podado ----------



print("Completo :",tempo_full)
print("Podado   :",tempo_podado)

print(
 "speedup =",
 tempo_full/tempo_podado
)

pasta = os.path.dirname(os.path.abspath(__file__))
arquivo = os.path.join(pasta,'Teste_temporal.txt')

with open(arquivo, "a") as f:

    f.write("Teste \n")
    f.write(f"ponto: ({xp},{yp})    N = {nmax}   t = {t}     Repetições = {Nrep} \n")
    f.write("=-"*10)
    f.write("\n")
    f.write(f"Completo : {tempo_full}\n" )
    f.write(f"Podado : {tempo_podado} \n")
    f.write(f"speedup = {tempo_full/tempo_podado}\n")
    f.write("=-"*10)
    f.write("\n")
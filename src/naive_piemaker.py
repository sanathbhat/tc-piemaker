import sys

T = int(input())
S = int(input())
C = int(input())
N = int(input())
P = float(input())

targets = [[0 for x in range(N)] for y in range(T)]  
targetCounts = [0] * T

sources = [[0 for x in range(N)] for y in range(S)]  
for i in range(S):
  s=input()
  for k in range(N):
    sources[i][k]=int(s[k])



numMoves=30
moves=0
i=0

while moves<numMoves:
  idS=i%S
  idT=-1

  for t in range(T):
    canPlace=True
    for k in range(N):
      if not(sources[idS][k]==0) and not(targets[t][k]==0):
        canPlace=False
    if canPlace:
      idT=t
      break

  if idT>=0:
    #place plate
    for k in range(N):
      if not(sources[idS][k]==0):
        targets[idT][k]=sources[idS][k]
        targetCounts[idT]+=1

    #clear target plate
    if targetCounts[idT]==N:
      targetCounts[idT]=0
      for k in range(N):
        targets[idT][k]=0


    print(str(idS)+" "+str(idT))
    sys.stdout.flush()       

  else:
    #discard plate
    print(str(idS)+" D")
    sys.stdout.flush()  

  moves+=1

  #read new plate and elapsed time
  s=input()
  for k in range(N):
    sources[idS][k]=int(s[k])

  elapsedTime=int(input())
  i+=1

print("-1")
sys.stdout.flush()
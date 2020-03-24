from model import Model
import numpy as np 
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import copy

m = Model(nodePerStimuli=8)
stimuli = ortho_group.rvs(8)
Ntrials = 10

inSigUS = stimuli[0]
inSigCS = stimuli[1]
inSigContext = stimuli[2]
inSigNoUS = stimuli[7]
inSigUnpairedCS = stimuli[3]
inSigUnpairedContext = stimuli[4]

def fearAcq():
	global Ntrials, inSigUS, inSigContext, inSigNoUS, inSigUnpairedCS, inSigUnpairedContext
	global m
	fearAcqCS = []
	fearAcqUnpairedCS = []
	fearAcqContext = []
	for t in range(Ntrials):
		
		m.forward(inSigUS, inSigCS, inSigContext, USpresent=1)
		fearAcqCS.append( m.evaluate(inSigNoUS, inSigCS, inSigUnpairedContext) )
		fearAcqUnpairedCS.append( m.evaluate(inSigNoUS, inSigUnpairedCS, inSigUnpairedContext) )
		fearAcqContext.append( m.evaluate(inSigNoUS, inSigUnpairedCS, inSigContext))

	return np.array(fearAcqCS), np.array(fearAcqUnpairedCS), np.array(fearAcqContext)


fearAcqCS, fearAcqUnpairedCS, fearAcqContext = fearAcq()

plt.plot(range(Ntrials), fearAcqCS, marker='o', label='CS+')
plt.plot(range(Ntrials), fearAcqUnpairedCS , marker='o', label='CS-' )
plt.plot(range(Ntrials), fearAcqContext , marker='o', label='context' )
plt.legend()
plt.xlabel('Number of Trials')
plt.ylabel('Fear Response')
plt.title('Fear Acquisition')
plt.savefig('fearAcq.png')
plt.clf()



fearExt = []
fearExtDiffContext = []

m1 = copy.copy(m)
m2 = copy.copy(m)


Ntrials2 = 100

for t in range(Ntrials2):
	
	m1.forward(inSigNoUS, inSigCS, inSigContext, USpresent=0)
	fearExt.append( m1.evaluate(inSigNoUS, inSigCS, inSigContext))

	m2.forward(inSigNoUS, inSigCS, inSigUnpairedContext, USpresent=0)
	fearExtDiffContext.append( m2.evaluate(inSigNoUS, inSigCS, inSigUnpairedContext))

print(fearExt)
plt.plot(range(Ntrials2), fearExt, marker='o', label='same context')
plt.xlabel('Number of Trials')
plt.ylabel('Fear Response')
plt.title('Fear Extinction')
plt.savefig('fearExt.png')



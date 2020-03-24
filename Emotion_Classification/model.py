import numpy as np 
from multiprocessing import Pool

def hardlim(x):
	if x>0:
		return 1
	else:
		return 0

class BrainPart:
	def __init__(self, idx,inputDim, alpha, beta, gamma):
		'''
		Args:
			idx(int)
			inputDim(int)
			alpha(float)
			beta(float)
			gamma(float)
		'''
		self.idx = idx #the index of the brain part in BrainModel
		self.OFCWeight = (np.random.random_sample(inputDim) - 0.5 )
		self.amygWeight = (np.random.random_sample(inputDim+1) -0.5)
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

	def responseNlearn(self, stimuliTotal, reinforce):
		'''
		Get output and update weights
		Args:
			stimuliTotal(np.array): dimension [inputDim+1, ], the last element is the signal from thalamus
			reinforce(int): 1 or 0
		'''
		self.response(stimuliTotal)
		self.learn(stimuliTotal, reinforce)
		return self.E

	def response(self,stimuliTotal):
		'''
		Return the output of this brain part
		Recognize the image and response to it
		Args:
			stimuliTotal(np.array): dimension [inputDim+1, ]
		'''
		self.Ea = hardlim(np.dot(stimuliTotal, self.amygWeight)) 
		self.Eo = hardlim(np.dot(stimuliTotal[:-1],self.OFCWeight))
		self.Ea_prime = hardlim(np.dot(stimuliTotal[:-1], self.amygWeight[:-1]))
		self.E = self.Ea - self.Eo
		#print('response E: {}, Ea: {}'.format(self.E, self.Ea))

	def learn(self, stimuliTotal, reinforce):
		'''
		After recognizing the image, learn if the response is right
		Update the weights

		Args:
			stimuliTotal(np.array): dimension [inputDim+1, ]
			reinforce(int): 1 if this image is goal image, others are 0
		'''
		amygWeight_old = self.amygWeight
		self.amygWeight = (1-self.gamma)*self.amygWeight + self.alpha*max(reinforce-self.Ea, 0)*stimuliTotal

		if reinforce==1:
			R0 = max(self.Ea_prime-reinforce, 0) - self.Eo
		elif reinforce ==0 :
			R0 = max(self.Ea_prime-self.Eo, 0)
		else:
			print('Error!!!!')

		self.OFCWeight = (1-self.gamma)*self.OFCWeight + self.beta*R0*stimuliTotal[:-1]

class BinaryBrainModel:
	def __init__(self,inputDim, alpha, beta, gamma):
		'''
		Only one output node, output 1 if the image is emotional
		input_num(int): number of nodes for the stimuli
		'''
		self.inputDim = inputDim
		
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

		self.brain_part = BrainPart(idx=0, inputDim=inputDim, alpha=alpha, beta=beta, gamma=gamma)


	def trainPerEpoch(self, images, labels, epoch=None):
		'''
		Train the model with the images

		Args:
			images(np.array): dimension [image_num, imag_dimension]
			labels(np.array): dimension [image_num, ], the content are integers
			epoch(int): to record which epoch is this
		'''
		#assert np.max(labels)+1 == self.outputDim
		assert len(images) == len(labels)
		predict = []
		for i in range(len(images)):
			image_aug = np.append(images[i], np.max(images[i]))
			output = self.brain_part.responseNlearn(image_aug, labels[i])

			predict.append(output)

			#print('epoch: {}, percent: {}'.format(epoch, i/len(images)), end='\r')
		predict = np.array(predict)
		return ((predict == labels).sum())/len(labels)

	def save(self, pathAmy, pathOFC):
		amygWeights = self.brain_part.amygWeight.reshape(1,-1)
		OFCWeights = self.brain_part.OFCWeight.reshape(1,-1)

		print(amygWeights)
		print('saving amygWeights...')
		np.save(pathAmy, amygWeights)

		print(OFCWeights)
		print('saving OFCWeights...')
		np.save(pathOFC, OFCWeights)

		print('Done!')


class BrainModel:
	def __init__(self,inputDim, outputDim, alpha, beta, gamma):
		'''
		input_num(int): number of nodes for the stimuli
		output_num(int): number of labels
		'''
		self.inputDim = inputDim
		self.outputDim = outputDim
		
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

		self.brainPartList = [BrainPart(idx=i, inputDim=inputDim, alpha=alpha, beta=beta, gamma=gamma) for i in range(outputDim)] 


	def _trainImage(self, image_aug, label_vector):
		'''
		Train all the neurons for one image
		Args:
			image_aug(np.array): dimension [image_dimension+1]
			label_vector(np.array): dimension [outputDim]
		'''
		output = []
		for i in range(self.outputDim):
			output.append(self.brainPartList[i].responseNlearn(image_aug, label_vector[i]))

		output = np.array(output)
		return output

	def trainPerEpoch(self, images, labels, epoch=None):
		'''
		Train the model with the images

		Args:
			images(np.array): dimension [image_num, imag_dimension]
			labels(np.array): dimension [image_num, ], the content are integers
			epoch(int): to record which epoch is this
		'''
		correct = 0
		for i in range(len(images)):
			image_aug = np.append(images[i], np.max(images[i]))

			label = np.zeros(self.outputDim)
			label[labels[i]] = 1
			output = self._trainImage(image_aug,label)

			if np.array_equal(label, output) == 1:
				correct +=1

			#print('epoch: {}, percent: {}'.format(epoch, i/len(images)), end='\r')

		return correct / len(images)

	def save(self, pathAmy, pathOFC):
		amygWeights = self.brainPartList[0].amygWeight.reshape(1,-1)
		OFCWeights = self.brainPartList[0].OFCWeight.reshape(1,-1)

		for brain_part in self.brainPartList[1:]:
			amygWeights = np.append(amygWeights, brain_part.amygWeight.reshape(1,-1), axis=0)
			OFCWeights = np.append(OFCWeights, brain_part.OFCWeight.reshape(1,-1), axis=0)

		print(amygWeights)
		print('saving amygWeights...')
		np.save(pathAmy, amygWeights)

		print(OFCWeights)
		print('saving OFCWeights...')
		np.save(pathOFC, OFCWeights)

		print('Done!')









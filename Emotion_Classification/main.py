from model import BrainModel, BinaryBrainModel
import numpy as np 
import argparse

def process_command():
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--alpha', type=float, default=0.8)
	parser.add_argument('-b', '--beta', type=float, default=0.2)
	parser.add_argument('-g', '--gamma', type=float, default=0.02)
	parser.add_argument('-epo', '--epochs', type=int, default=10)
	parser.add_argument('-num', '--image_num', type=int, default=200)
	parser.add_argument('-emo', '--binary_emotion', action='store_true', default=False)
	args = parser.parse_args()
	return args

def readNProcess(pathx, pathy):
	'''
	read train and trainY and preprocess trainX
	'''
	global args

	trainX = np.load(pathx)
	trainY = np.load(pathy)

	mask = (np.max(trainX,axis=1).reshape(-1,1) != 0).reshape(-1)

	trainX = trainX[mask]/(np.max(trainX[mask],axis=1).reshape(-1,1))
	trainY = trainY[mask]

	

	if args.binary_emotion:
		mask = np.ma.mask_or((trainY==0), (trainY==6))
		trainX = trainX[mask]
		trainY = trainY[mask]

	trainX = trainX[:args.image_num]
	trainY = trainY[:args.image_num]

	if args.binary_emotion:
		trainY[trainY==0] = 1 #angry has label 1 (emotional images)
		trainY[trainY==6] = 0 #neutral has label 0 (normal images)

	for i in np.unique(trainY):
		print('Emotion type {}: {}'.format(i,(trainY==i).sum()))


	return trainX, trainY


if __name__ == '__main__':
	args = process_command()
	for key, item in vars(args).items():
		print('{}: {}'.format(key, item))


	trainX, trainY = readNProcess('data/trainX.npy', 'data/trainY.npy')
	print(trainX)
	print(trainX.shape)
	print(trainY)
	print(trainY.shape)

	print('='*20+'Start Training'+'='*20)

	if args.binary_emotion:
		brain = BinaryBrainModel(inputDim=2304, alpha=args.alpha, beta=args.beta, gamma=args.gamma)

	else:
		brain = BrainModel(inputDim=2304, outputDim=7, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
	
	print('total images: {}'.format(args.image_num))

	accLi = []
	for epoch in range(args.epochs):
		acc = brain.trainPerEpoch(images=trainX, labels=trainY, epoch=epoch)
		accLi.append(acc)
		if epoch%10 == 0:
			print('Epoch: {}, accuracy:{}'.format(epoch, acc))
			print('=============')
		#brain.save('weights/amyg.npy','weights/OFC.npy')
	brain.save('weights/amyg.npy','weights/OFC.npy')
	np.save('acc.npy', np.array(accLi))









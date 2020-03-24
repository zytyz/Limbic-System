import numpy as np 
import pandas as pd 
from multiprocessing import Pool

def decodeStr(string):
	return [int(x) for x in string.split(' ')]


def getTrain(path):
	data = pd.read_csv(path)
	print(data)
	with Pool() as p:
		trainX = p.map(decodeStr, data['feature'].values)
	trainY = data['label'].values
	return np.array(trainX), np.array(trainY)

if __name__ == '__main__':
	trainX, trainY = getTrain('data/train.csv')
	print(trainX)
	print(trainX.shape)
	print(trainY)
	print(trainY.shape)
	np.save('data/trainX.npy', trainX)
	np.save('data/trainY.npy', trainY)
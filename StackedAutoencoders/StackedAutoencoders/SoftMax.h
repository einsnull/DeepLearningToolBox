#pragma once
#include "FunctionBase.h"
#include <ctime>
#include <fstream>
#include <iostream>
using namespace std;

//trainingData  ndim * numOfExamples
//labels numOfExamples * 1

class SoftMax : public FunctionBase
{
private:
	MatrixXd theta;
	int inputSize;
	int numClasses;
public:
	SoftMax(int inputSize,int numClasses);
	MatrixXi predict(MatrixXd &data);
	void train(MatrixXd &data,MatrixXi &labels,
		double lambda,double alpha,
		int maxIter,int miniBatchSize);
	bool saveModel(char *szFileName);
	bool loadModel(char *szFileName);
	MatrixXd getTheta();
private:
	MatrixXd randomInitialize(int lIn,int lOut);
	double computeCost(double lambda,MatrixXd &data,
		MatrixXi &labels,MatrixXd &thetaGrad);
	void SoftMax::miniBatchSGD(MatrixXd &trainData,MatrixXi &labels,
		double lambda,double alpha,int maxIter,int batchSize);
};

//initialize the model
SoftMax::SoftMax(int inputSize,int numClasses)
{
	this ->inputSize = inputSize;
	this ->numClasses = numClasses;
	theta = randomInitialize(numClasses,inputSize);
}

MatrixXd SoftMax::getTheta()
{
	return theta;
}

//random initialize the weight
MatrixXd SoftMax::randomInitialize(int lIn,int lOut)
{
	//Random initialize the weight in a specific range
	int i,j;
	double epsilon = sqrt(6.0/(lIn + lOut + 1));
	MatrixXd result(lIn,lOut);
	srand((unsigned int)time(NULL));
	for(i = 0;i < lOut;i++)
	{
		for(j = 0;j < lIn;j++)
		{
			result(j,i) = ((double)rand() / (double)RAND_MAX) * 2 * epsilon - epsilon; 
		}
	}
	return result;
}

//Predict
MatrixXi SoftMax::predict(MatrixXd &data)
{
	//cout << theta.rows() << " " << theta.cols() << endl;
	//cout << data.rows() << " " << data.cols() << endl;
	MatrixXd m = theta * data;
	MatrixXi pred(m.cols(),1);
	for(int i = 0; i < m.cols(); i++)
	{
		double max = 0;
		int idx = 0;
		for(int j = 0; j < m.rows();j++)
		{
			if(m(j,i) > max)
			{
				max = m(j,i);
				idx = j;
			}
		}
		pred(i,0) = idx;
	}
	return pred;
}

//cost function
double SoftMax::computeCost(double lambda,MatrixXd &data,
							MatrixXi &labels,MatrixXd & thetaGrad)
{
	int numCases = data.cols();
	MatrixXd groundTruth = binaryCols(labels,numClasses);
	//
	MatrixXd M = theta * data;
	MatrixXd maxM = M.colwise().maxCoeff();

	M = bsxfunMinus(M,maxM);
	MatrixXd expM = expMat(M);
	
	MatrixXd tmp1 = (expM.colwise().sum()).replicate(numClasses,1);

	MatrixXd p = expM.cwiseQuotient(tmp1);
	//compute cost
	double cost = (groundTruth.cwiseProduct(logMat(p))).sum() * (-1.0 / numCases)
		+ (lambda / 2.0) * theta.array().square().sum();
	//compute the gradient of theta
	thetaGrad = (groundTruth - p) * data.transpose() * (-1.0 / numCases)
		+ theta * lambda;
	return cost;
}

//mini batch stochastic gradient descent
void SoftMax::miniBatchSGD(MatrixXd &trainingData,MatrixXi &labels,double lambda,
						   double alpha,int maxIter,int batchSize)
{
	//get the binary code of labels
	MatrixXd thetaGrad(theta.rows(),theta.cols());
	MatrixXd miniTrainingData(trainingData.rows(),batchSize);
	MatrixXi miniLabels(batchSize,1);
	int iter = 1;
	int numBatches = trainingData.cols() / batchSize;
	
	//mini batch stochastic gradient decent
	for(int i = 0; i < maxIter;i++)
	{
		double J = 0;
		// compute the cost
		for(int j = 0;j < numBatches; j++)
		{
			miniTrainingData = trainingData.middleCols(j * batchSize,batchSize);
			miniLabels = labels.middleRows(j * batchSize,batchSize);
			J += computeCost(lambda,miniTrainingData,miniLabels,thetaGrad);
#ifdef _IOSTREAM_
			if(miniTrainingData.cols() < 1 || miniTrainingData.rows() < 1)
			{
				cout << "Too few training examples!"  << endl; 
			}
#endif

			if(fabs(J) < 0.001)
			{
				break;
			}
			//update theta
			theta -= thetaGrad * alpha;
		}
		J = J / numBatches;
#ifdef _IOSTREAM_
		cout << "iter: " << iter++ << "  cost: " << J << endl;
#endif
	}
}

//train the model
void SoftMax::train(MatrixXd &data,MatrixXi &labels,
					double lambda,double alpha,
					int maxIter,int miniBatchSize)
{
	miniBatchSGD(data,labels,lambda,alpha,maxIter,miniBatchSize);
}

//save model to file
bool SoftMax::saveModel(char *szFileName)
{
	ofstream ofs(szFileName);
	if(!ofs)
	{
		return false;
	}
	int i,j;
	ofs << inputSize << " " << numClasses << endl;
	for(i = 0; i < theta.rows(); i++)
	{
		for(j = 0;j < theta.cols(); j++)
		{
			ofs << theta(i,j) << " ";
		}
	}
	ofs.close();
	return true;
}

//load model from file
bool SoftMax::loadModel(char *szFileName)
{
	ifstream ifs(szFileName);
	if(!ifs)
	{
		return false;
	}
	ifs >> this -> inputSize >> this -> numClasses;
	int i,j;
	theta.resize(numClasses,inputSize);

	for(i = 0; i < theta.rows(); i++)
	{
		for(j = 0;j < theta.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> theta(i,j);
		}
	}
	ifs.close();
	return true;
}
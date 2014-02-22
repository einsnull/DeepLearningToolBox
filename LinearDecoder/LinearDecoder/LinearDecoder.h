#pragma once
#include "FunctionBase.h"
#include <Eigen/Dense>
#include <ctime>
#include <fstream>

// sparse auto encoder class
class LinearDecoder:public FunctionBase
{
private:
	//weights
	MatrixXd theta1;
	MatrixXd theta2;
	//bias
	MatrixXd b1;
	MatrixXd b2;
	int inputSize;
	int hiddenSize;
public:
	LinearDecoder(int inputSize,int hiddenSize);
	void train(
	MatrixXd &trainData,double lambda,
	double alpha,double beta,double sp,int maxIter,
	int miniBatchSize);
	bool saveModel(char *szFileName);
	bool loadModel(char *szFileName);
	MatrixXd getTheta();
	MatrixXd getBias();
	
private:
	MatrixXd randomInitialize(int lIn,int lOut);
	void updateParameters(
		MatrixXd &theta1Grad1,MatrixXd &theta2Grad2,
		MatrixXd &b1Grad,MatrixXd &b2Grad,double alpha);
	void miniBatchSGD(MatrixXd &trainData,double lambda,
		double alpha,int maxIter,int batchSize,double beta,double sp);
	double computeCost(MatrixXd &data,double lambda,MatrixXd &theta1Grad,
		MatrixXd &theta2Grad,MatrixXd &b1Grad,MatrixXd &b2Grad,
		double beta,double sparsityParam);
};

//constructor
LinearDecoder::LinearDecoder(int inputSize,int hiddenSize)
{
	this->inputSize = inputSize;
	this->hiddenSize = hiddenSize;
	theta1 = randomInitialize(hiddenSize,inputSize);
	theta2 = randomInitialize(inputSize,hiddenSize);
	b1 = MatrixXd::Zero(hiddenSize,1);
	b2 = MatrixXd::Zero(inputSize,1);
}

MatrixXd LinearDecoder::getTheta()
{
	return theta1;
}

MatrixXd LinearDecoder::getBias()
{
	return b1;
}

//random initialize the weights
MatrixXd LinearDecoder::randomInitialize(int lIn,int lOut)
{
	//random initialize the weight
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


//gradient descent update rule
void LinearDecoder::updateParameters(
	MatrixXd &theta1Grad1,MatrixXd &theta2Grad2,
	MatrixXd &b1Grad,MatrixXd &b2Grad,double alpha)
{
	theta1 -= (theta1Grad1*alpha);
	theta2 -= (theta2Grad2*alpha);
	b1 -= (b1Grad * alpha);
	b2 -= (b2Grad * alpha);
}

//cost function
double LinearDecoder::computeCost(
	MatrixXd &data,double lambda,MatrixXd &theta1Grad,MatrixXd &theta2Grad,
	MatrixXd &b1Grad,MatrixXd &b2Grad,double beta,double sparsityParam)
{

	double cost = 0;

	int numOfExamples = data.cols();
	//cout << numOfExamples << endl;

	
	MatrixXd z2 = theta1 * data + b1.replicate(1,numOfExamples);
	MatrixXd a2 = sigmoid(z2);
	MatrixXd z3 = theta2 * a2 + b2.replicate(1,numOfExamples);

	MatrixXd rho = a2.rowwise().sum() * (1.0 / (double)numOfExamples);
	//cout << "rho:" << rho << endl;
	double sp = sparsityParam;
	MatrixXd term1 = MatrixXd::Ones(rho.rows(),rho.cols()) - rho;
	MatrixXd spDelta = reciprocalMat(term1) * (1.0 - sp)
		-  reciprocalMat(rho) * sp;
	//compute delta
	MatrixXd delta3 = z3 - data;

	MatrixXd delta2 = (theta2.transpose() * delta3 
		+ spDelta.replicate(1,numOfExamples) * beta).cwiseProduct(sigmoidGradient(z2));

	//compute gradients

	theta2Grad = delta3 * a2.transpose() * (1.0 / (double)numOfExamples)
		+ theta2 * lambda;
	b2Grad = delta3.rowwise().sum() * (1.0 / (double)numOfExamples);
	theta1Grad = delta2 * data.transpose() * ( 1.0 / (double)numOfExamples)
		+ theta1 * lambda;
	b1Grad = delta2.rowwise().sum() * (1.0  / (double)numOfExamples);

	MatrixXd term2 = reciprocalMat(rho) * sp;
	MatrixXd term3 = reciprocalMat(term1) * (1.0 - sp);
	//compute cost
	double spCost = (logMat(term2) * sp + logMat(term3) * (1.0 - sp)).array().sum() * beta;
	//cout << "spCost: " << spCost << endl;
	double regCost = (theta1.array().square().sum()
		+ theta2.array().square().sum()) * (lambda / 2.0);
	//cout << "regCost: " << regCost << endl;
	cost = (z3 - data).array().square().sum() * (1.0 / 2.0 / numOfExamples);
	//cout << "c: " << cost << endl; 
	cost = cost	+ regCost + spCost;
	//system("pause");
	return cost;
}


//mini batch stochastic gradient descent
void LinearDecoder::miniBatchSGD(
	MatrixXd &trainData,double lambda,double alpha,int maxIter,int batchSize,double beta,double sp)
{
	//get the binary code of labels
	MatrixXd theta1Grad(theta1.rows(),theta1.cols());
	MatrixXd theta2Grad(theta2.rows(),theta2.cols());
	MatrixXd b1Grad(b1.rows(),b1.cols());
	MatrixXd b2Grad(b2.rows(),b2.cols());
	MatrixXd miniTrainData(trainData.rows(),batchSize);
	int iter = 1;
	int numBatches = trainData.cols() / batchSize;
	
	//mini batch stochastic gradient decent
	for(int i = 0; i < maxIter;i++)
	{
		double J = 0;
		// compute the cost
		for(int j = 0;j < numBatches; j++)
		{
			miniTrainData = trainData.middleCols(j * batchSize,batchSize);
			J += computeCost(miniTrainData,lambda,theta1Grad,theta2Grad,
				b1Grad,b2Grad,beta,sp) * (1.0 / (double)numBatches);
#ifdef _IOSTREAM_
			if(miniTrainData.cols() < 1 || miniTrainData.rows() < 1)
			{
				cout << "Too few training examples!"  << endl; 
			}
#endif

			if(fabs(J) < 0.001)
			{
				break;
			}
			updateParameters(theta1Grad,theta2Grad,b1Grad,b2Grad,alpha);
		}
		
#ifdef _IOSTREAM_
		cout << "iter: " << iter++ << "  cost: " << J << endl;
#endif
	}
}

//train the model
void LinearDecoder::train(
	MatrixXd &trainData,double lambda,
	double alpha,double beta,double sp,int maxIter,
	int miniBatchSize)
{
	if(trainData.rows() != this->inputSize)
	{
#ifdef _IOSTREAM_
		cout << "TrainData rows:" << trainData.rows() << endl;
		cout << "dimension mismatch!" << endl;
#endif
		return;
	}
	miniBatchSGD(trainData,lambda,alpha,maxIter,miniBatchSize,beta,sp);
}


//save model to file
bool LinearDecoder::saveModel(char *szFileName)
{
	ofstream ofs(szFileName);
	if(!ofs)
	{
		return false;
	}
	int i,j;
	ofs << inputSize << " " << hiddenSize << endl;
	for(i = 0; i < theta1.rows(); i++)
	{
		for(j = 0;j < theta1.cols(); j++)
		{
			ofs << theta1(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < theta2.rows(); i++)
	{
		for(j = 0;j < theta2.cols(); j++)
		{
			ofs << theta2(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < b1.rows(); i++)
	{
		for(j = 0; j < b1.cols(); j++) 
		{
			ofs << b1(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < b2.rows(); i++)
	{
		for(j = 0; j < b2.cols(); j++) 
		{
			ofs << b2(i,j) << " ";
		}
	}
	ofs.close();
	return true;
}

//load model from file
bool LinearDecoder::loadModel(char *szFileName)
{
	ifstream ifs(szFileName);
	if(!ifs)
	{
		return false;
	}
	ifs >> this -> inputSize >> this -> hiddenSize;
	int i,j;
	theta1.resize(this->hiddenSize,this->inputSize);
	theta2.resize(this->inputSize,this->hiddenSize);
	b1.resize(1,hiddenSize);
	b2.resize(1,inputSize);

	for(i = 0; i < theta1.rows(); i++)
	{
		for(j = 0;j < theta1.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> theta1(i,j);
		}
	}
	for(i = 0; i < theta2.rows(); i++)
	{
		for(j = 0;j < theta2.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> theta2(i,j);
		}
	}
	for(i = 0; i < b1.rows(); i++)
	{
		for(j = 0; j < b1.cols(); j++) 
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> b1(i,j);
		}
	}
	for(i = 0; i < b2.rows(); i++)
	{
		for(j = 0; j < b2.cols(); j++) 
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> b2(i,j);
		}
	}
	ifs.close();
	return true;
}
#include <iostream>
#include <Eigen/Dense>
#include "FunctionBase.h"
#include <ctime>

using namespace std;
using namespace Eigen;

class RBM : public FunctionBase
{
private:
	//weights
	MatrixXd visHidWeight;
	//bias
	MatrixXd hidBias;
	MatrixXd visBias;
	//positive phase visible unit activation
	MatrixXd posVisAct;
	//negative phase visible unit activation
	MatrixXd negVisAct;
	//positive phase hidden unit activation
	MatrixXd posHidAct;
	//negative phase hidden unit activation
	MatrixXd negHidAct;
	MatrixXd posProds;
	MatrixXd negProds;
	//inc term
	MatrixXd visHidWeightInc;
	MatrixXd visBiasInc;
	MatrixXd hidBiasInc;
public:
	RBM(int numDims,int numHid);
	void train(MatrixXd &data,int batchSize,int maxIter,
		double epsilonW,double epsilonVb,double epsilonHb,
		double weightCost,double initialMomentum,double finalMomentum);
	MatrixXd getWeight();
	//bool saveModel(char *szFileName);
	//bool loadModel(char *szFileName);
private:
	MatrixXd randomInitialize(int lIn,int lOut);
	MatrixXd positivePhase(MatrixXd &data);
	MatrixXd negativePhase(MatrixXd &posHidStates);
	void update(double momentum,double epsilonW,double epsilonVb,
		double epsilonHb,double weightCost,int batchSize);
	MatrixXd getPosHidStates(MatrixXd &posHidProbs, MatrixXd &randomProbs);
};

RBM::RBM(int numDims,int numHid)
{
	//initialize parameters
	visHidWeight = randomInitialize(numDims,numHid);
	hidBias = MatrixXd::Zero(1,numHid);
	visBias = MatrixXd::Zero(1,numDims);

	visHidWeightInc = MatrixXd::Zero(numDims,numHid);
	hidBiasInc = MatrixXd::Zero(1,numHid);
	visBiasInc = MatrixXd::Zero(1,numDims);
}

MatrixXd RBM::getWeight()
{
	return visHidWeight;
}

//random initialize the weights
MatrixXd RBM::randomInitialize(int lIn,int lOut)
{
	//random initialize the weight
	int i,j;
	double epsilon = 4 * sqrt(6.0/(lIn + lOut + 1));
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

MatrixXd RBM::getPosHidStates(MatrixXd &posHidProbs, MatrixXd &randomProbs)
{
	int rows = posHidProbs.rows();
	int cols = posHidProbs.cols();
	MatrixXd result(rows,cols);
	for(int i = 0;i < rows; i++)
	{
		for(int j = 0;j < cols;j++)
		{
			result(i,j) = posHidProbs(i,j) > randomProbs(i,j);
		}
	}
	return result;
}

MatrixXd RBM::positivePhase(MatrixXd &data)
{
	//postive phase
	MatrixXd tmp = data*visHidWeight;
	MatrixXd posHidProbs = sigmoid(bsxfunPlus(tmp,hidBias));
	posProds = data.transpose() * posHidProbs;
	posHidAct = posHidProbs.colwise().sum();
	posVisAct = data.colwise().sum();
	MatrixXd posHidStates = getPosHidStates(posHidProbs,
		randMat(posHidProbs.rows(),posHidProbs.cols()));
	//(posHidProbs.array() > randMat(posHidProbs.rows(),posHidProbs.cols()).array());
	return posHidStates;
}

MatrixXd RBM::negativePhase(MatrixXd &posHidStates)
{
	//negative phase
	MatrixXd tmp = posHidStates*visHidWeight.transpose();
	MatrixXd negData = sigmoid(bsxfunPlus(tmp,visBias));
	tmp = negData * visHidWeight;
	MatrixXd negHidProbs = sigmoid(bsxfunPlus(tmp,hidBias));
	negProds = negData.transpose() * negHidProbs;
	negHidAct = negHidProbs.colwise().sum();
	negVisAct = negData.colwise().sum();
	return negData;
}

void RBM::update(double momentum,double epsilonW,double epsilonVb,
				 double epsilonHb,double weightCost,int batchSize)
{
	//calc inc
	visHidWeightInc = momentum * visHidWeightInc
		+ epsilonW * ((posProds - negProds) / (double)batchSize
		- weightCost * visHidWeight);
	visBiasInc = momentum * visBiasInc 
		+ (epsilonVb / (double)batchSize) * (posVisAct - negVisAct);
	hidBiasInc = momentum * hidBiasInc
		+ (epsilonHb / (double)batchSize) * (posHidAct - negHidAct);
	
	//update
	visHidWeight = visHidWeight + visHidWeightInc;
	visBias = visBias + visBiasInc;
	hidBias = hidBias + hidBiasInc;
}

void RBM::train(MatrixXd &data,int batchSize,int maxIter,
		double epsilonW,double epsilonVb,double epsilonHb,
		double weightCost,double initialMomentum,double finalMomentum)
{
	//train the RBM with mini batch stochastic gradient descent
	MatrixXd miniTrainData(batchSize,data.cols());
	int numBatches = data.rows() / batchSize;
	for(int i = 0;i < maxIter;i++)
	{
		double errSum = 0;
		for(int j = 0; j < numBatches;j++)
		{
			miniTrainData = data.middleRows(j * batchSize,batchSize);
			//perform CD-1
			MatrixXd posHidStates = positivePhase(miniTrainData);
			MatrixXd negData = negativePhase(posHidStates);
			// delta between data and model generated data
			MatrixXd delta = miniTrainData - negData;
			double err = delta.array().square().sum();
			errSum += err;
			double momentum = 0;
			if(i > 4)
			{
				momentum = finalMomentum;
			}
			else
			{
				momentum = initialMomentum;
			}
			//update parameters
			update(momentum,epsilonW,epsilonVb,epsilonHb,weightCost,batchSize);
		}
		cout << "Epoch: " << i << "    cost: " << errSum << endl;
	}
}
//training data:nDim * nExamples
#include "SAE.h"
#include "DAE.h"
#include "FunctionBase.h"
#include "SoftMax.h"
#include "getConfig.h"


typedef enum {
	SPARSE_AE,DENOISING_AE
}AE_TYPE;

class StackedAE : public FunctionBase
{
private:
	MatrixXd aeTheta1;
	MatrixXd aeTheta2;
	MatrixXd aeB1;
	MatrixXd aeB2;
	MatrixXd softMaxTheta;
	int numClasses;
	int ae1HiddenSize;
	int ae2HiddenSize;
	int inputSize;
public:
	StackedAE(int ae1HiddenSize,int ae2HiddenSize,int numClasses);
	MatrixXi predict(
		MatrixXd &data);
	void fineTune(MatrixXd &data,
		MatrixXi &labels,double lambda,
		double alpha,int maxIter,int batchSize);
	void preTrain(MatrixXd &data,MatrixXi &labels,
		double lambda[],double alpha[],int miniBatchSize[],
		int maxIter[],AE_TYPE aeType,double noiseRatio[] = NULL,
		double beta[] = NULL,double sparsityParam[] = NULL);
	bool saveModel(char *szFileName);
	bool loadModel(char *szFileName);
	MatrixXd getAe1Theta();
	MatrixXd getAe2Theta();
private:
	MatrixXd softmaxGradient(MatrixXd &x);
	MatrixXd feedForward(MatrixXd &theta,
		MatrixXd &b,MatrixXd data);
	void updateParameters(MatrixXd &theta1Grad,MatrixXd &theta2Grad,
						   MatrixXd &b1Grad,MatrixXd &b2Grad,
						   MatrixXd &softmaxTheta,double alpha);
	double computeCost(MatrixXd &theta1Grad,
		MatrixXd &b1Grad,MatrixXd &theta2Grad,MatrixXd &b2Grad,
		MatrixXd &softmaxThetaGrad,MatrixXd &data,
		MatrixXi &labels,double lambda);
};

//initialize the model
StackedAE::StackedAE(int ae1HiddenSize,int ae2HiddenSize,int numClasses)
{
	this->ae1HiddenSize = ae1HiddenSize;
	this->ae2HiddenSize = ae2HiddenSize;
	this->numClasses = numClasses;
}

MatrixXd StackedAE::getAe1Theta()
{
	return aeTheta1;
}

MatrixXd StackedAE::getAe2Theta()
{
	return aeTheta2;
}

//forward calculation
MatrixXd StackedAE::feedForward(MatrixXd &theta,MatrixXd &b,
									 MatrixXd data)
{
	int m = data.cols();
	MatrixXd z2 = theta * data + b.replicate(1,m);
	MatrixXd a2 = sigmoid(z2);
	return a2;
}

//predict
MatrixXi StackedAE::predict(
		MatrixXd &data)
{
	//forward calculation
	MatrixXd term1 = aeTheta1 * data;
	MatrixXd z2 = bsxfunPlus(term1,aeB1);
	MatrixXd a2 = sigmoid(z2);
	MatrixXd term2 = aeTheta2 * a2;
	MatrixXd z3 = bsxfunPlus(term2,aeB2);
	MatrixXd a3 = sigmoid(z3);
	MatrixXd z4 = softMaxTheta * a3;
	MatrixXd a4 = expMat(z4);
	MatrixXd a4ColSum = a4.colwise().sum();
	a4 = bsxfunRDivide(a4,a4ColSum);
	MatrixXi pred(1,a4.cols());
	for(int i = 0;i < a4.cols();i++)
	{
		double max = 0;
		int idx = 0;
		for(int j = 0;j < a4.rows();j++)
		{
			if(a4(j,i) > max)
			{
				idx = j;
				max = a4(j,i);
			}
		}
		pred(0,i) = idx;
	}
	return pred;
}

//component wise softmax gradient
MatrixXd StackedAE::softmaxGradient(MatrixXd &x)
{
	MatrixXd negX = x * (-1);
	MatrixXd expX = expMat(negX);
	MatrixXd term1 = (MatrixXd::Ones(expX.rows(),expX.cols())
		+ expX).array().square();
	MatrixXd grad = expX.cwiseQuotient(term1);
	return grad;
}

//update all parameters
void StackedAE::updateParameters(MatrixXd &theta1Grad,MatrixXd &theta2Grad,
						   MatrixXd &b1Grad,MatrixXd &b2Grad,
						   MatrixXd &softmaxThetaGrad,double alpha)
{
	aeTheta1 -= theta1Grad * alpha;
	aeTheta2 -= theta2Grad * alpha;
	aeB1 -= b1Grad * alpha;
	aeB2 -= b2Grad * alpha;
	softMaxTheta -= softmaxThetaGrad * alpha;
}

//cost function
double StackedAE::computeCost(MatrixXd &theta1Grad,
		MatrixXd &b1Grad,MatrixXd &theta2Grad,
		MatrixXd &b2Grad,MatrixXd &softmaxThetaGrad,
		MatrixXd &data,MatrixXi &labels,double lambda)
{
	MatrixXd groundTruth = binaryCols(labels,numClasses);
	int M = labels.rows();
	//forward calculate
	MatrixXd term1 = aeTheta1 * data;
	MatrixXd z2 = bsxfunPlus(term1,aeB1);
	MatrixXd a2 = sigmoid(z2);
	MatrixXd term2 = aeTheta2 * a2;
	MatrixXd z3 = bsxfunPlus(term2,aeB2);
	MatrixXd a3 = sigmoid(z3);
	MatrixXd z4 = softMaxTheta * a3;
	MatrixXd a4 = expMat(z4);
	MatrixXd a4ColSum = a4.colwise().sum();
	a4 = bsxfunRDivide(a4,a4ColSum);
	//calculate delta
	MatrixXd delta4 = a4 - groundTruth;
	MatrixXd delta3 = (softMaxTheta.transpose() * delta4).cwiseProduct(sigmoidGradient(z3));
	MatrixXd delta2 = (aeTheta2.transpose() * delta3).cwiseProduct(sigmoidGradient(z2));

	//calculate delta
	softmaxThetaGrad = (groundTruth - a4) * a3.transpose() * (-1.0 / M) + softMaxTheta * lambda;

	theta2Grad = delta3 * a2.transpose() * (1.0 / M) + aeTheta2 * lambda;
	b2Grad = delta3.rowwise().sum() * (1.0 / M);
	theta1Grad = delta2 * data.transpose() * (1.0 / M) + aeTheta1 * lambda;
	b1Grad = delta2.rowwise().sum() * (1.0 / M);

	//compute cost
	double cost = (-1.0 / M) * (groundTruth.cwiseProduct(logMat(a4))).array().sum()
		+ lambda / 2.0 * softMaxTheta.array().square().sum()
		+ lambda / 2.0 * aeTheta1.array().square().sum()
		+ lambda / 2.0 * aeTheta2.array().square().sum();

	return cost;
}

//fine tune the model
void StackedAE::fineTune(MatrixXd &data,MatrixXi &labels,
				   double lambda,double alpha,int maxIter,int batchSize)
{
	MatrixXd theta1Grad(aeTheta1.rows(),aeTheta1.cols());
	MatrixXd theta2Grad(aeTheta2.rows(),aeTheta2.cols());
	MatrixXd b1Grad(aeB1.rows(),aeB1.cols());
	MatrixXd b2Grad(aeB2.rows(),aeB2.cols());
	MatrixXd softmaxThetaGrad(softMaxTheta.rows(),softMaxTheta.cols());
	MatrixXd miniTrainData(data.rows(),batchSize);
	MatrixXi miniLabels(batchSize,1);
	int iter = 1;
	int numBatches = data.cols() / batchSize;
	
	//mini batch stochastic gradient decent
	for(int i = 0; i < maxIter;i++)
	{
		double J = 0;
		// compute the cost
		for(int j = 0;j < numBatches; j++)
		{
			miniTrainData = data.middleCols(j * batchSize,batchSize);
			miniLabels = labels.middleRows(j * batchSize,batchSize);
			J += computeCost(theta1Grad,b1Grad,theta2Grad,
				b2Grad,softmaxThetaGrad,miniTrainData,miniLabels,lambda);
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
			updateParameters(theta1Grad,theta2Grad,b1Grad,b2Grad,softmaxThetaGrad,alpha);
		}
		J = J / numBatches;
#ifdef _IOSTREAM_
		cout << "iter: " << iter++ << "  cost: " << J << endl;
#endif
	}
}

//pretrain the model
void StackedAE::preTrain(MatrixXd &data,MatrixXi &labels,
		double lambda[],double alpha[],int miniBatchSize[],
		int maxIter[],AE_TYPE aeType,double noiseRatio[],
		double beta[],double sparsityParam[])
{
	int numOfExamples = data.cols();
	int ndim = data.rows(); 
	inputSize = ndim;
	if(aeType == SPARSE_AE)
	{
		//stacked sparse autoencoders
		cout << "PreTraining with sparse autoencoder ..." << endl;
		SAE ae1(ndim,ae1HiddenSize);
		cout << "PreTraining ae1 ..." << endl;
		//train the first sae
		ae1.train(data,lambda[0],alpha[0],beta[0],
			sparsityParam[0],maxIter[0],miniBatchSize[0]);

		MatrixXd theta1 = ae1.getTheta();
		aeTheta1.resize(theta1.rows(),theta1.cols());
		aeTheta1 = theta1;
		MatrixXd b1 = ae1.getBias();
		aeB1.resize(b1.rows(),b1.cols());
		aeB1 = b1;
		//train the second sae
		MatrixXd ae1Features = feedForward(aeTheta1,aeB1,data);
		SAE ae2(ae1HiddenSize,ae2HiddenSize);
		cout << "PreTraining ae2 ..." << endl;
		ae2.train(ae1Features,lambda[1],alpha[1],beta[1],
			sparsityParam[1],maxIter[1],miniBatchSize[1]);

		MatrixXd theta2 = ae2.getTheta();
		aeTheta2.resize(theta2.rows(),theta2.cols());
		aeTheta2 = theta2;
		MatrixXd b2 = ae2.getBias();
		aeB2.resize(b2.rows(),b2.cols());
		aeB2 = b2;
		//train the softmax regression
		MatrixXd ae2Features = feedForward(aeTheta2,aeB2,ae1Features);
		cout << "PreTraining softmax ..." << endl;
		SoftMax softmax(ae2HiddenSize,numClasses);
		softmax.train(ae2Features,labels,lambda[2],alpha[2],maxIter[2],miniBatchSize[2]);
		MatrixXd smTheta = softmax.getTheta();
		softMaxTheta.resize(smTheta.rows(),smTheta.cols());
		softMaxTheta = smTheta;
	}
	else if(aeType == DENOISING_AE)
	{
		//stacked denoising autoencoders
		cout << "PreTraining with denoising autoencoder ..." << endl;
		//train the first denoising autoencoder
		DAE ae1(ndim,ae1HiddenSize);
		cout << "PreTraining ae1 ..." << endl;
		ae1.train(data,noiseRatio[0],alpha[0],maxIter[0],miniBatchSize[0]);

		MatrixXd theta1 = ae1.getTheta();
		aeTheta1.resize(theta1.rows(),theta1.cols());
		aeTheta1 = theta1;
		MatrixXd b1 = ae1.getBias();
		aeB1.resize(b1.rows(),b1.cols());
		aeB1 = b1;

		//train the second denoising autoencoder
		MatrixXd ae1Features = feedForward(aeTheta1,aeB1,data);
		DAE ae2(ae1HiddenSize,ae2HiddenSize);
		cout << "PreTraining ae2 ..." << endl;
		ae2.train(ae1Features,noiseRatio[1],alpha[1],maxIter[1],miniBatchSize[1]);

		MatrixXd theta2 = ae2.getTheta();
		aeTheta2.resize(theta2.rows(),theta2.cols());
		aeTheta2 = theta2;
		MatrixXd b2 = ae2.getBias();
		aeB2.resize(b2.rows(),b2.cols());
		aeB2 = b2;
		//train the softmax regression
		MatrixXd ae2Features = feedForward(aeTheta2,aeB2,ae1Features);
		cout << "PreTraining softmax ..." << endl;
		SoftMax softmax(ae2HiddenSize,numClasses);
		softmax.train(ae2Features,labels,lambda[2],alpha[2],maxIter[2],miniBatchSize[2]);
		MatrixXd smTheta = softmax.getTheta();
		softMaxTheta.resize(smTheta.rows(),smTheta.cols());
		softMaxTheta = smTheta;
	}
}

//save model to file
bool StackedAE::saveModel(char *szFileName)
{
	ofstream ofs(szFileName);
	if(!ofs)
	{
		return false;
	}
	int i,j;
	ofs << inputSize << " " << ae1HiddenSize << " "
		<< ae2HiddenSize << " " << numClasses << endl;
	for(i = 0; i < aeTheta1.rows(); i++)
	{
		for(j = 0;j < aeTheta1.cols(); j++)
		{
			ofs << aeTheta1(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < aeTheta2.rows(); i++)
	{
		for(j = 0;j < aeTheta2.cols(); j++)
		{
			ofs << aeTheta2(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < aeB1.rows(); i++)
	{
		for(j = 0; j < aeB1.cols(); j++) 
		{
			ofs << aeB1(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < aeB2.rows(); i++)
	{
		for(j = 0; j < aeB2.cols(); j++) 
		{
			ofs << aeB2(i,j) << " ";
		}
	}
	ofs << endl;
	for(i = 0; i < softMaxTheta.rows(); i++)
	{
		for(j = 0; j < softMaxTheta.cols(); j++)
		{
			ofs << softMaxTheta(i,j) << " ";
		}
	}
	ofs.close();
	return true;
}

//load model from file
bool StackedAE::loadModel(char *szFileName)
{
	ifstream ifs(szFileName);
	if(!ifs)
	{
		return false;
	}
	
	ifs >> inputSize >> ae1HiddenSize >> ae2HiddenSize >> numClasses;
	int i,j;
	aeTheta1.resize(ae1HiddenSize,inputSize);
	aeTheta2.resize(ae2HiddenSize,ae1HiddenSize);
	aeB1.resize(ae1HiddenSize,1);
	aeB2.resize(ae2HiddenSize,1);

	for(i = 0; i < aeTheta1.rows(); i++)
	{
		for(j = 0;j < aeTheta1.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> aeTheta1(i,j);
		}
	}
	for(i = 0; i < aeTheta2.rows(); i++)
	{
		for(j = 0;j < aeTheta2.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> aeTheta2(i,j);
		}
	}
	for(i = 0; i < aeB1.rows(); i++)
	{
		for(j = 0; j < aeB1.cols(); j++) 
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> aeB1(i,j);
		}
	}
	for(i = 0; i < aeB2.rows(); i++)
	{
		for(j = 0; j < aeB2.cols(); j++) 
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> aeB2(i,j);
		}
	}
	for(i = 0; i < softMaxTheta.rows(); i++)
	{
		for(j = 0; j < softMaxTheta.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> softMaxTheta(i,j);
		}
	}
	ifs.close();
	return true;
}
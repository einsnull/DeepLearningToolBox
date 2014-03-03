//training data:nExamples * ndim
#pragma once
#include <Eigen/Dense>
#include <ctime>
#include <fstream>

#define GD 1
#define MINI_BATCH_SGD 0

using namespace std;
using namespace Eigen;

class BPNN
{
private:
	MatrixXd theta1;
	MatrixXd theta2;
	int inputSize;
	int hiddenSize;
	int outputSize;
public:
	BPNN(int inputSize,int hiddenSize,int outputSize);
	MatrixXi predict(MatrixXd &x);
	double calcAccurancy(MatrixXi &pred,MatrixXi &labels);
	void train(
		   MatrixXd &trainData,MatrixXi &label,
		   double lambda,double alpha,int maxIter,int trainMode,
		   void * option,MatrixXd &testData,MatrixXi &testLabel);
	bool saveModel(char *szFileName);
	bool loadModel(char *szFileName);
private:
	MatrixXd binaryRows(MatrixXi &labels,int numOfClasses);
	MatrixXd randomInitialize(int lIn,int lOut);
	MatrixXd sigmoid(MatrixXd &z);
	MatrixXd sigmoidGradient(MatrixXd &z);
	MatrixXd logMat(MatrixXd &z);
	double computeCost(MatrixXd &x,MatrixXd &y,double lambda,
				   MatrixXd &grad1,MatrixXd &grad2);
	void updateParameters(
					 MatrixXd &grad1,MatrixXd &grad2,double alpha);
	void gradientDescent(MatrixXd &trainData,MatrixXi &label,
		   double lambda,double alpha,int maxIter,
		   MatrixXd &testData,MatrixXi &testLabel);
	void miniBatchSGD(MatrixXd &trainData,MatrixXi &label,
		   double lambda,double alpha,int maxIter,int batchSize,
		   MatrixXd &testData,MatrixXi &testLabel);
};

//constructor
BPNN::BPNN(int inputSize,int hiddenSize,int outputSize)
{
	this->inputSize = inputSize;
	this->hiddenSize = hiddenSize;
	this->outputSize = outputSize;
	theta1 = randomInitialize(inputSize,hiddenSize);
	theta2 = randomInitialize(hiddenSize,outputSize);
}


//random initialize the weights
MatrixXd BPNN::randomInitialize(int lIn,int lOut)
{
	//random initialize the weight
	int i,j;
	double epsilon = 4 * sqrt(6.0 / (this->inputSize + this->hiddenSize));
	MatrixXd result(lOut,lIn+1);
	srand((unsigned int)time(NULL));
	for(i = 0;i < lOut;i++)
	{
		for(j = 0;j < lIn + 1;j++)
		{
			result(i,j) = ((double)rand() / (double)RAND_MAX) * 2 * epsilon - epsilon;
		}
	}
	return result;
}

//sigmoid function
MatrixXd BPNN::sigmoid(MatrixXd &z)
{
	//return  1.0 ./ (1.0 + exp(-z));
	MatrixXd result(z.rows(),z.cols());
	for(int i = 0;i < z.rows();i++)
	{
		for(int j = 0;j < z.cols();j++)
		{
			result(i,j) = 1.0 / (1 + exp(-z(i,j)));
		}
	}
	return result;
}

//compute the gradient of sigmoid function
MatrixXd BPNN::sigmoidGradient(MatrixXd &z)
{
	//return sigmoid(z) .* (1 - sigmoid(z))
	MatrixXd result;
	MatrixXd sigm = sigmoid(z);
	MatrixXd item = MatrixXd::Ones(z.rows(),z.cols()) - sigm;
	result = sigm.cwiseProduct(item);
	return result;
}

//component wise log function
MatrixXd BPNN::logMat(MatrixXd &z)
{
	MatrixXd result(z.rows(),z.cols());
	for(int i = 0;i < z.rows();i++)
	{
		for(int j = 0;j < z.cols();j++)
		{
			result(i,j) = log(z(i,j));
		}
	}
	return result;
}

// simple gradient descent
void BPNN::updateParameters(
					 MatrixXd &grad1,MatrixXd &grad2,double alpha)
{
	theta1 -= grad1*alpha;
	theta2 -= grad2*alpha;
}

//predict the result
MatrixXi BPNN::predict(MatrixXd &x)
{
	MatrixXd p(x.rows(),1);
	int xRows = x.rows();
	int xCols = x.cols();
	MatrixXd a1(xRows,xCols+1);
	//a1 = [ones([size(x,1),1]) x]
	a1.leftCols(1) = MatrixXd::Ones(xRows,1);
	a1.rightCols(xCols) = x;
	//z2 = a1 * theta1'
	MatrixXd z2 = a1*theta1.transpose();
	MatrixXd h2 = sigmoid(z2);
	int h2Rows = h2.rows();
	int h2Cols = h2.cols();
	MatrixXd a2(h2Rows,h2Cols+1);
	//a2 = [ones([size(h2,1),1]) h2]
	a2.leftCols(1) = MatrixXd::Ones(h2Rows,1);
	a2.rightCols(h2Cols) = h2;
	//z3 = a2 * theta2'
	MatrixXd z3 = a2*theta2.transpose();
	//h3 = sigmoid(z3)
	MatrixXd h3 = sigmoid(z3);
	int cols = h3.cols();
	int rows = h3.rows();
	MatrixXi pred(rows,1);
	int i,j;
	//get the index of max item of each row
	for(j = 0; j < rows;j ++)
	{
		double max = 0;
		int idx = 0;
		for(i = 0; i < cols;i++)
		{
			if(h3(j,i) > max)
			{
				max = h3(j,i);
				idx = i;
			}
		}
		pred(j,0) = idx;
	}
	return pred;
}

//cost function
double BPNN::computeCost(MatrixXd &x,MatrixXd &y,double lambda,
				   MatrixXd &grad1,MatrixXd &grad2)
{
	double J = 0;
	//forward
	int xRows = x.rows();
	int xCols = x.cols();
	int numOfExamples = xRows;
	MatrixXd a1(xRows,xCols+1);
	//a1 = [ones([size(x,1),1]) x]
	a1.leftCols(1) = MatrixXd::Ones(xRows,1);
	a1.rightCols(xCols) = x;
	//z2 = a1 * theta1'
	MatrixXd z2 = a1*theta1.transpose();
	MatrixXd h2 = sigmoid(z2);
	int h2Rows = h2.rows();
	int h2Cols = h2.cols();
	MatrixXd a2(h2Rows,h2Cols+1);
	//a2 = [ones([size(h2,1),1]) h2]
	a2.leftCols(1) = MatrixXd::Ones(h2Rows,1);
	a2.rightCols(h2Cols) = h2;
	//z3 = a2 * theta2'
	MatrixXd z3 = a2*theta2.transpose();
	//h3 = sigmoid(z3)
	MatrixXd h3 = sigmoid(z3);
	
	MatrixXd newTheta1(theta1.rows(),theta1.cols() - 1);
	MatrixXd newTheta2(theta2.rows(),theta2.cols() - 1);
	newTheta1 = theta1.rightCols(newTheta1.cols());
	newTheta2 = theta2.rightCols(newTheta2.cols());
	//regularziation term of the cost function
	double JReg = lambda/(2*numOfExamples)*(newTheta1.array().square().sum()
		+ newTheta2.array().square().sum());
	
	MatrixXd item = MatrixXd::Ones(h3.rows(),h3.cols()) - h3;
	//cost function 
	//cout << y.rows() << " " << y.cols() << endl;
	//cout << item.rows() << " " << item.cols() << endl;
	//cout << d3.rows() << " " << d3.cols() << endl;
	//cout << h3.rows() << " " << h3.cols() << endl;

	J = (1.0/numOfExamples) * ((y - MatrixXd::Ones(y.rows(),y.cols())).cwiseProduct(logMat(item))
		- y.cwiseProduct(logMat(h3))).array().sum() + JReg; 
	
	//compute gradients
	MatrixXd d3 = h3 - y;
	
	MatrixXd d2 = (d3*newTheta2).cwiseProduct(sigmoidGradient(z2));
	
	grad1 = d2.transpose() * a1 * (1.0/numOfExamples);
	grad2 = d3.transpose() * a2 * (1.0/numOfExamples);
	
	//Add regularization
	grad1.rightCols(grad1.cols() - 1) += newTheta1 * (lambda/numOfExamples);
	grad2.rightCols(grad2.cols() - 1) += newTheta2 * (lambda/numOfExamples);
	return J;
}

//set targets to binary code
MatrixXd BPNN::binaryRows(MatrixXi &labels,int numOfClasses)
{
	// return binary code of labels
	//eye function
	MatrixXd e = MatrixXd::Identity(numOfClasses,numOfClasses);
	int rows = labels.rows();
	int cols = e.cols();
	MatrixXd result(rows,cols);
	for(int i = 0; i < rows; i++)
	{
		int idx = labels(i,0);
		result.row(i) = e.row(idx);
	}
	return result;
}

//gradient descent method
void BPNN::gradientDescent(MatrixXd &trainData,MatrixXi &label,
		   double lambda,double alpha,int maxIter,
		   MatrixXd &testData,MatrixXi &testLabel)
{
	double J = 0;
	//get the binary code of labels
	MatrixXd y = binaryRows(label,this -> outputSize);
	MatrixXd grad1(theta1.rows(),theta1.cols());
	MatrixXd grad2(theta2.rows(),theta2.cols());
	int iter = 1;
	//gradient decent
 	for(int i = 0; i < maxIter;i++)
	{
		// compute the cost
		J = computeCost(trainData,y,lambda,grad1,grad2);
#ifdef _IOSTREAM_
		MatrixXi pred = predict(testData);
		cout << "iter: " << iter++ << "  cost: " << J;
		cout << "   test accurancy: " << calcAccurancy(pred,testLabel) << endl;
#endif
		if(fabs(J) < 0.001)
		{
			break;
		}
		updateParameters(grad1,grad2,alpha);
		if(alpha > 0.2)
		{
			alpha -= 0.1;
		}
 	}
}

//mini batch stochastic gradient descent
void BPNN::miniBatchSGD(MatrixXd &trainData,MatrixXi &label,
		   double lambda,double alpha,int maxIter,int batchSize,
		   MatrixXd &testData,MatrixXi &testLabel)
{
	double J = 0;
	//get the binary code of labels
	MatrixXd y = binaryRows(label,this -> outputSize);
	MatrixXd grad1(theta1.rows(),theta1.cols());
	MatrixXd grad2(theta2.rows(),theta2.cols());
	MatrixXd miniTrainData(batchSize,trainData.cols());
	MatrixXd miniY(batchSize,label.cols());
	int iter = 1;
	//mini batch stochastic gradient decent
 	for(int i = 0; i < maxIter;i++)
	{
		// compute the cost
		for(int j = 0;j < trainData.rows() / batchSize; j++)
		{
			miniTrainData = trainData.middleRows(j * batchSize,batchSize);
			miniY = y.middleRows(j * batchSize,batchSize);
			J = computeCost(miniTrainData,miniY,lambda,grad1,grad2);
#ifdef _IOSTREAM_
			if(miniTrainData.rows() < 1)
			{
				cout << "Too few training examples!"  << endl; 
			}
			MatrixXi pred = predict(testData);
			cout << "iter: " << iter++ << "  cost: " << J;
			cout << "   test accurancy: " << calcAccurancy(pred,testLabel) << endl;
#endif
			if(fabs(J) < 0.001)
			{
				break;
			}
			updateParameters(grad1,grad2,alpha);
		}
 	}
}

//train the model
void BPNN::train(
		   MatrixXd &trainData,MatrixXi &label,
		   double lambda,double alpha,int maxIter,
		   int trainMode,void *option,
		   MatrixXd &testData,MatrixXi &testLabel)
{
	if(trainData.cols() != this->inputSize)
	{
#ifdef _IOSTREAM_
		cout << "dimension mismatch!" << endl;
#endif
		return;
	}
	if(trainMode == GD)
	{
		gradientDescent(trainData,label,lambda,alpha,maxIter,testData,testLabel);
	}
	else if(trainMode == MINI_BATCH_SGD)
	{
		miniBatchSGD(trainData,label,lambda,alpha,maxIter,*((int*)option),testData,testLabel);
	}
}

//calculate the accurancy of the given set 
double BPNN::calcAccurancy(MatrixXi &pred,MatrixXi &labels)
{
	int numOfExamples = pred.rows();
	double sum = 0;
	for(int i = 0; i < numOfExamples; i++)
	{
		if(pred(i,0) == labels(i,0))
		{
			sum += 1;
		}
	}
	return sum / numOfExamples;
}

//save model to file
bool BPNN::saveModel(char *szFileName)
{
	ofstream ofs(szFileName);
	if(!ofs)
	{
		return false;
	}
	int i,j;
	ofs << inputSize << " " << hiddenSize << " " << outputSize << endl;
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
	ofs.close();
	return true;
}

//load model from file
bool BPNN::loadModel(char *szFileName)
{
	ifstream ifs(szFileName);
	if(!ifs)
	{
		return false;
	}
	ifs >> this -> inputSize >> this -> hiddenSize >> this -> outputSize;
	int i,j;
	theta1.resize(this->hiddenSize,this->inputSize + 1);
	theta2.resize(this->outputSize,this->hiddenSize + 1);

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
	ifs.close();
	return true;
}
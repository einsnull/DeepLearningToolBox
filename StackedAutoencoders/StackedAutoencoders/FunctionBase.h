#pragma once
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

//scalar reciprocal
double reciprocalScalar(double x)
{
	return 1.0/x;
}

//scalar sigmoid function
double sigmoidScalar(double x)
{
	return 1.0 / (1 + exp(-x));
}

//scalar log function
double logScalar(double x)
{
	return log(x);
}

//scalar exp function
double expScalar(double x)
{
	return exp(x);
}

//scalar sqrt function
double sqrtScalar(double x)
{
	return sqrt(x);
}

class FunctionBase
{
public:
	MatrixXd bsxfunMinus(MatrixXd &m,MatrixXd &x);
	MatrixXd bsxfunRDivide(MatrixXd &m,MatrixXd &x);
	MatrixXd bsxfunPlus(MatrixXd &m,MatrixXd &x);
	MatrixXd sigmoid(MatrixXd &z);
	MatrixXd sigmoidGradient(MatrixXd &z);
	MatrixXd binaryCols(MatrixXi &labels,int numOfClasses);
	MatrixXd expMat(MatrixXd &z);
	MatrixXd logMat(MatrixXd &z);
	MatrixXd sqrtMat(MatrixXd &z);
	MatrixXd reciprocal(MatrixXd &z);
	double calcAccurancy(MatrixXi &pred,MatrixXi &labels);
	FunctionBase();
};

FunctionBase::FunctionBase()
{
#ifdef _WINDOWS_
	//set eigen threads
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	Eigen::setNbThreads(info.dwNumberOfProcessors);
#endif
}

MatrixXd FunctionBase::bsxfunMinus(MatrixXd &m,MatrixXd &x)
{
	MatrixXd r = m;
	if(x.rows() == 1)
	{
		r = x.replicate(m.rows(),1);
	}
	if(x.cols() == 1)
	{
		r = x.replicate(1,m.cols());
	}
	return m - r;
}

MatrixXd FunctionBase::bsxfunRDivide(MatrixXd &m,MatrixXd &x)
{
	MatrixXd r = m;
	if(x.rows() == 1)
	{
		r = x.replicate(m.rows(),1);
	}
	if(x.cols() == 1)
	{
		r = x.replicate(1,m.cols());
	}
	return m.cwiseQuotient(r);
}

MatrixXd FunctionBase::bsxfunPlus(MatrixXd &m,MatrixXd &x)
{
	MatrixXd r = m;
	if(x.rows() == 1)
	{
		r = x.replicate(m.rows(),1);
	}
	if(x.cols() == 1)
	{
		r = x.replicate(1,m.cols());
	}
	return m + r;
}

MatrixXd FunctionBase::sigmoidGradient(MatrixXd &z)
{
	//return sigmoid(z) .* (1 - sigmoid(z))
	MatrixXd result;
	MatrixXd sigm = sigmoid(z);
	MatrixXd item = MatrixXd::Ones(z.rows(),z.cols()) - sigm;
	result = sigm.cwiseProduct(item);
	return result;
}

//component wise sigmoid function
MatrixXd FunctionBase::sigmoid(MatrixXd &z)
{
	return z.unaryExpr(ptr_fun(sigmoidScalar));
}

MatrixXd FunctionBase::sqrtMat(MatrixXd &z)
{
	return z.unaryExpr(ptr_fun(sqrtScalar));
}

MatrixXd FunctionBase::binaryCols(MatrixXi &labels,int numOfClasses)
{
	// return binary code of labels
	//eye function
	MatrixXd e = MatrixXd::Identity(numOfClasses,numOfClasses);
	int numOfExamples = labels.rows();
	int inputSize = e.cols();
	MatrixXd result(inputSize,numOfExamples);
	for(int i = 0; i < numOfExamples; i++)
	{
		int idx = labels(i,0);
		result.col(i) = e.col(idx);
	}
	return result;
}

//component wise exp function
MatrixXd FunctionBase::expMat(MatrixXd &z)
{
	return z.unaryExpr(ptr_fun(expScalar));
}

//component wise log function
MatrixXd FunctionBase::logMat(MatrixXd &z)
{
	return z.unaryExpr(ptr_fun(logScalar));
}

double FunctionBase::calcAccurancy(MatrixXi &pred,MatrixXi &labels)
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

//return 1.0 ./ z
MatrixXd FunctionBase::reciprocal(MatrixXd &z)
{
	return z.unaryExpr(ptr_fun(reciprocalScalar));
}
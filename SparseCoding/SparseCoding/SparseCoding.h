#include "FunctionBase.h"
#include "dataAndImage.h"
#include <ctime>

class SparseCoding:public FunctionBase
{
private:
	MatrixXd featureMatrix;
	MatrixXd weightMatrix;
public:
	MatrixXd getWeight();
	MatrixXd getFeature();
	SparseCoding(int visibleSize,int numFeatures,int batchNumPatches);
	void train(MatrixXd &patches,int iter,
				int batchNumPatches,int numFeatures,int gdIter,
				double alpha,double lambda,double epsilon,double gamma,
				int imgWidth,int poolDim = 0,bool isTopo = false);
private:
	MatrixXd randomInitialize(int lIn,int lOut);
	void gradientDescent(MatrixXd &trainData,
		double gamma,double lambda,double epsilon,
		MatrixXd &groupMatrix,bool isTopo,
		double alpha,int maxIter);
	void updateParameters(MatrixXd &grad,double alpha);
	double sparseCodingFeatureCost(
		MatrixXd &patches,double gamma,double lambda,
		double epsilon,MatrixXd &groupMatrix,MatrixXd &grad,bool isTopo = false);
};

SparseCoding::SparseCoding(int visibleSize,int numFeatures,int batchNumPatches)
{
	weightMatrix = randomInitialize(visibleSize,numFeatures);
	featureMatrix = randomInitialize(numFeatures,batchNumPatches);
}

MatrixXd SparseCoding::getWeight()
{
	return weightMatrix;
}

MatrixXd SparseCoding::getFeature()
{
	return featureMatrix;
}

MatrixXd SparseCoding::randomInitialize(int lIn,int lOut)
{
	//random initialize the weight
	int i,j;
	double epsilon = 1;
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

double SparseCoding::sparseCodingFeatureCost(
		MatrixXd &patches,double gamma,double lambda,
		double epsilon,MatrixXd &groupMatrix,MatrixXd &grad,bool isTopo)
{
	int numOfExamples = patches.cols();
	//cout << "numOfExamples:" << numOfExamples << endl;
	MatrixXd delta = weightMatrix * featureMatrix - patches;
	// 重构误差
	double fResidue = delta.array().square().sum() / (double)numOfExamples;
	MatrixXd term1 = featureMatrix.array().square();
	MatrixXd term2 = groupMatrix * term1 
		+ MatrixXd::Ones(groupMatrix.rows(),term1.cols()) * epsilon;
	MatrixXd sparsityMatrix = sqrtMat(term2);
	//稀疏惩罚
	double fSparsity = lambda * sparsityMatrix.array().sum();
	double cost = fResidue + fSparsity;
	//目标函数的偏导数
	//MatrixXd g1 = weightMatrix.transpose() * patches * (-2.0);
	//MatrixXd g2 = weightMatrix.transpose() * weightMatrix * featureMatrix * 2.0;
	//cout << g1.rows() << " " << g2.rows() << endl;
	//cout << g1.cols() << " " << g2.cols() << endl;
	
	//dimension mismatch!!!!

	MatrixXd gradResidue = (weightMatrix.transpose() * patches * (-2.0)
		+ weightMatrix.transpose() * weightMatrix * featureMatrix * 2.0)
		* (1.0 / (double)numOfExamples);

	//sparsity gradient
	MatrixXd gradSparsity;
	if(!isTopo)
	{
		//non-topographic
		gradSparsity = featureMatrix.cwiseQuotient(sparsityMatrix) * lambda;
	}
	else
	{
		//topographic
		MatrixXd term3 = reciprocal(sqrtMat(term2));
		gradSparsity = (groupMatrix.transpose() 
			* term3).cwiseProduct(featureMatrix) * lambda;
	}
	grad = gradResidue + gradSparsity;

	return cost;
}

void SparseCoding::train(MatrixXd &patches,int iter,
				int batchNumPatches,int numFeatures,int gdIter,
				double alpha,double lambda,double epsilon,double gamma,
				int imgWidth,int poolDim,bool isTopo)
{
	MatrixXd groupMatrix;
	if(!isTopo)
	{
		groupMatrix = MatrixXd::Identity(numFeatures,numFeatures);
	}
	
	int numOfExamples = patches.cols();
	int batches = numOfExamples / batchNumPatches;
	//cout << "batches:" << batches << endl;
	//cvNamedWindow("weights",CV_WINDOW_AUTOSIZE);
	//int idx = 1;
	cout << "iter    fObj     fResidue    fSparsity    fWeight" << endl;
	for(int j = 0; j < iter;j++)
	{
		double tSprasity = 0;
		double tWeight = 0;
		double tResidue = 0;
		for(int i = 0;i < batches; i++)
		{
			MatrixXd batchPatches = patches.middleCols(i * batchNumPatches,batchNumPatches);
			//重构误差
			double error = (weightMatrix * featureMatrix - batchPatches).array().square().sum()
				/ (double)batchNumPatches;
			double fResidue = error;
			MatrixXd term1 = featureMatrix.array().square();
			MatrixXd R = groupMatrix * term1;
			R = R + MatrixXd::Ones(R.rows(),R.cols()) * epsilon;
			R = sqrtMat(R);
			//稀疏惩罚
			double fSparsity = R.array().sum() * lambda;
			//权重惩罚
			double fWeight = weightMatrix.array().square().sum() * gamma;

			//对feature Matrix重新初始化
			featureMatrix = weightMatrix.transpose() * batchPatches;
			MatrixXd normWM = ((weightMatrix.array().square())
				.colwise().sum()).transpose();
			featureMatrix = bsxfunRDivide(featureMatrix,normWM);
			
			gradientDescent(batchPatches,gamma,lambda,epsilon,
				groupMatrix,isTopo,alpha,gdIter);

			weightMatrix = batchPatches * featureMatrix.transpose()
				* ((MatrixXd::Identity(featureMatrix.rows(),featureMatrix.rows())
				* gamma * batchNumPatches + featureMatrix * featureMatrix.transpose()).inverse());
			tResidue += fResidue;
			tWeight += fWeight;
			tSprasity += fSparsity;
		}
		double avgResidue = tResidue / (double)batches;
		double avgSparsity = tSprasity / (double)batches;
		double avgWeight = tWeight / (double)batches;
		cout << j << "     " << avgResidue + avgSparsity + avgWeight << "     "
				<< avgResidue << "     " << avgSparsity << "     " << avgWeight << endl;
		//save to file
		MatrixXd wt = weightMatrix.transpose();
		buildImage(wt,imgWidth,"weights.jpg",false);
	}
	//cvDestroyWindow("weights");
}

void SparseCoding::updateParameters(MatrixXd &grad,double alpha)
{
	featureMatrix -= grad * alpha;
}

void SparseCoding::gradientDescent(
	MatrixXd &trainData,double gamma,double lambda,
		double epsilon,MatrixXd &groupMatrix,bool isTopo,
		double alpha,int maxIter)
{
	//get the binary code of labels

	MatrixXd grad(featureMatrix.rows(),featureMatrix.cols());
	int iter = 1;
	
	//mini batch stochastic gradient decent
	for(int i = 0; i < maxIter;i++)
	{

		// compute the cost
		double J = sparseCodingFeatureCost(
			trainData,gamma,lambda,epsilon,
			groupMatrix,grad,isTopo);
		//updateParameters(grad,alpha);
		featureMatrix -= grad * alpha;
#ifdef _IOSTREAM_
		//cout << "iter: " << iter++ << "  cost: " << J << endl;
#endif
	}
}

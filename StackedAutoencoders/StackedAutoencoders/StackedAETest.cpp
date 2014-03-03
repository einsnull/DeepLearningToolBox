//training data:nDim * nExamples
#include <windows.h>
#include <iostream>
#include "dataAndImage.h"
#include "StackedAE.h"
#include "ca.h"
using namespace std;

int main()
{
	int ae1HiddenSize = 100;
	int ae2HiddenSize = 100;
	int numClasses = 10;
	int imgWidth = 28;
	double noiseRatio[2] = {0.3,0.3};
	double lambda[4] = {3e-3,3e-3,3e-3,1e-4};
	double alpha[4] = {0.2,0.2,0.2,0.2};
	double beta[2] = {3,3};
	double sparsityParam[2] = {0.1,0.1};
	int maxIter[4] = {100,100,100,200};
	int miniBatchSize[4] = {1000,1000,1000,1000};

#ifdef _WINDOWS_
	//set eigen threads
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	Eigen::setNbThreads(info.dwNumberOfProcessors);
#endif

	MatrixXd trainingData(1,1);
	MatrixXi trainingLabels(1,1);
	MatrixXd testData(1,1);
	MatrixXi testLabels(1,1);
	char *fileBuf  = new char[4096];
	bool ret = loadFileToBuf("ParamConfig.ini",fileBuf,4096);
	if(ret)
	{
		getConfigDoubleValue(fileBuf,"noiseRatio0:",noiseRatio[0]);
		getConfigDoubleValue(fileBuf,"noiseRatio1:",noiseRatio[1]);
		getConfigDoubleValue(fileBuf,"lambda0:",lambda[0]);
		getConfigDoubleValue(fileBuf,"lambda1:",lambda[1]);
		getConfigDoubleValue(fileBuf,"lambda2:",lambda[2]);
		getConfigDoubleValue(fileBuf,"lambda3:",lambda[3]);
		getConfigDoubleValue(fileBuf,"alpha0:",alpha[0]);
		getConfigDoubleValue(fileBuf,"alpha1:",alpha[1]);
		getConfigDoubleValue(fileBuf,"alpha2:",alpha[2]);
		getConfigDoubleValue(fileBuf,"alpha3:",alpha[3]);
		getConfigDoubleValue(fileBuf,"beta0:",beta[0]);
		getConfigDoubleValue(fileBuf,"beta1:",beta[1]);
		getConfigDoubleValue(fileBuf,"sparseParam0:",sparsityParam[0]);
		getConfigDoubleValue(fileBuf,"sparseParam1:",sparsityParam[1]);
		getConfigIntValue(fileBuf,"maxIter0:",maxIter[0]);
		getConfigIntValue(fileBuf,"maxIter1:",maxIter[1]);
		getConfigIntValue(fileBuf,"maxIter2:",maxIter[2]);
		getConfigIntValue(fileBuf,"maxIter3:",maxIter[3]);
		getConfigIntValue(fileBuf,"miniBatchSize0:",miniBatchSize[0]);
		getConfigIntValue(fileBuf,"miniBatchSize1:",miniBatchSize[1]);
		getConfigIntValue(fileBuf,"miniBatchSize2:",miniBatchSize[2]);
		getConfigIntValue(fileBuf,"miniBatchSize3:",miniBatchSize[3]);
		getConfigIntValue(fileBuf,"ae1HiddenSize:",ae1HiddenSize);
		getConfigIntValue(fileBuf,"ae2HiddenSize:",ae2HiddenSize);
		getConfigIntValue(fileBuf,"imgWidth:",imgWidth);
		cout << "lambda0: " << lambda[0] << endl;
		cout << "lambda1: " << lambda[1] << endl;
		cout << "lambda2: " << lambda[2] << endl;
		cout << "lambda3: " << lambda[3] << endl;
		cout << "alpha0: " << alpha[0] << endl;
		cout << "alpha1: " << alpha[1] << endl;
		cout << "alpha2: " << alpha[2] << endl;
		cout << "alpha3: " << alpha[3] << endl;
		/*cout << "beta0: " << beta[0] << endl;
		cout << "beta1: " << beta[1] << endl;
		cout << "sparseParam0: " << sparsityParam[0] << endl;
		cout << "sparseParam1: " << sparsityParam[1] << endl;*/
		cout << "maxIter0: " << maxIter[0] << endl;
		cout << "maxIter1: " << maxIter[1] << endl;
		cout << "maxIter2: " << maxIter[2] << endl;
		cout << "maxIter3: " << maxIter[3] << endl;
		cout << "miniBatchSize0: " << miniBatchSize[0] << endl;
		cout << "miniBatchSize1: " << miniBatchSize[1] << endl;
		cout << "miniBatchSize2: " << miniBatchSize[2] << endl;
		cout << "miniBatchSize3: " << miniBatchSize[3] << endl;
		cout << "ae1HiddenSize: " << ae1HiddenSize << endl;
		cout << "ae2HiddenSize: " << ae2HiddenSize << endl;
		cout << "imgWidth: " << imgWidth << endl;
	}
	delete []fileBuf;
	//timer
	clock_t start = clock();
	ret = loadMnistData(trainingData,"mnist\\train-images-idx3-ubyte");
	cout << "Loading training data..." << endl;
	if(ret == false)
	{
		return -1;
	}
	ret = loadMnistLabels(trainingLabels,"mnist\\train-labels-idx1-ubyte");
	if(ret == false)
	{
		return -1;
	}
	MatrixXd showData = trainingData.leftCols(100).transpose();
	buildImage(showData,imgWidth,"data.jpg",false);

	StackedAE stackedAE(ae1HiddenSize,ae2HiddenSize,numClasses);
	stackedAE.preTrain(trainingData,trainingLabels,lambda,alpha,miniBatchSize,
		maxIter,DENOISING_AE,noiseRatio,beta,sparsityParam);
	cout << "Loading test data..." << endl;
	ret = loadMnistData(testData,"mnist\\t10k-images-idx3-ubyte");
	if(ret == false)
	{
		return -1;
	}
	ret = loadMnistLabels(testLabels,"mnist\\t10k-labels-idx1-ubyte");
	if(ret == false)
	{
		return -1;
	}
	MatrixXi pred1 = stackedAE.predict(testData);
	double acc1 = stackedAE.calcAccurancy(testLabels,pred1);
	cout << "Accurancy before fine tuning: " << acc1 * 100 << "%" << endl;
	MatrixXd aeTheta1 = stackedAE.getAe1Theta();
	MatrixXd aeTheta2 = stackedAE.getAe2Theta();
	MatrixXd filter = aeTheta2 * aeTheta1;
	buildImage(aeTheta1,imgWidth,"ae1Before.jpg",false);
	buildImage(filter,imgWidth,"ae2Before.jpg",false);
	cout << "Fine Tuning..." << endl;
	stackedAE.fineTune(trainingData,trainingLabels,lambda[3],
		alpha[3],maxIter[3],miniBatchSize[3]);
	MatrixXi pred2 = stackedAE.predict(testData);
	double acc2 = stackedAE.calcAccurancy(testLabels,pred2);
	cout << "Accurancy: " << acc2 * 100 << "%" << endl;
	stackedAE.saveModel("StackedAE_Model.txt");
	aeTheta1 = stackedAE.getAe1Theta();
	aeTheta2 = stackedAE.getAe2Theta();
	filter = aeTheta2 * aeTheta1;
	buildImage(aeTheta1,imgWidth,"ae1After.jpg",false);
	buildImage(filter,imgWidth,"ae2After.jpg",false);
	clock_t end = clock();
	cout << "The code ran for " << 
		(end - start)/(double)(CLOCKS_PER_SEC*60) <<
		" minutes on " << Eigen::nbThreads() << " thread(s)." << endl;
	system("pause");
	return 0;
}


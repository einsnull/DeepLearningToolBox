//training data:nDim * nExamples
#include "dataAndImage.h"
#include "SAE.h"
#include "cmath"
#include "getConfig.h"

int main()
{
	//regularization coefficient
	double lambda = 0.0001;
	//learning rate
	double alpha = 0.03;
	double beta = 3;
	double sp = 0.01;
	int maxIter = 15000;
	int miniBatchSize = 1000;
	int n = 2;
	int inputSize = 28*28;
	int hiddenSize = 28;
	int imgWidth = 8; 
	char *fileBuf  = new char[4096];
	bool ret = loadFileToBuf("ParamConfig.ini",fileBuf,4096);
	if(ret)
	{
		getConfigDoubleValue(fileBuf,"lambda:",lambda);
		getConfigDoubleValue(fileBuf,"alpha:",alpha);
		getConfigDoubleValue(fileBuf,"beta:",beta);
		getConfigDoubleValue(fileBuf,"sparseParam:",sp);
		getConfigIntValue(fileBuf,"maxIter:",maxIter);
		getConfigIntValue(fileBuf,"miniBatchSize:",miniBatchSize);
		getConfigIntValue(fileBuf,"hiddenSize:",hiddenSize);
		getConfigIntValue(fileBuf,"inputSize:",inputSize);
		getConfigIntValue(fileBuf,"imgWidth:",imgWidth);
		cout << "lambda:" << lambda << endl;
		cout << "alpha:" << alpha << endl;
		cout << "beta:" << beta << endl;
		cout << "sparseParam:" << sp << endl;
		cout << "maxIter:" << maxIter << endl;
		cout << "miniBatchSize:" << miniBatchSize << endl;
		cout << "hiddenSize:" << hiddenSize << endl;
		cout << "inputSize:" << inputSize << endl;
		cout << "imgWidth:" << imgWidth << endl;
	}
	delete []fileBuf;
	MatrixXd trainData(n,inputSize);
	SAE sae(inputSize,hiddenSize);
	ret = loadMnistData(trainData,"mnist\\train-images-idx3-ubyte");
	cout << "Loading training data..." << endl;
	if(ret == false)
	{
		return -1;
	}
	//buildImage(trainData.topRows(100),8,"data.bmp");
	//sae.loadModel("SAE_Model.txt");
	clock_t start = clock();
	//cout << trainData.rows() << " " << trainData.cols() << endl; 
	MatrixXd showImage = trainData.leftCols(100).transpose();
	buildImage(showImage,imgWidth,"data.jpg");
	sae.train(trainData,lambda,alpha,beta,sp,maxIter,MINI_BATCH_SGD,&miniBatchSize);
	cout << "End Train" << endl;
	MatrixXd hiddenTheta = sae.getTheta();
	buildImage(hiddenTheta,imgWidth,"weights.jpg",true);
	cout << "Saving hidden neurons" << endl;
	sae.saveModel("SAE_Model.txt");
	clock_t end = clock();
	cout << "The code ran for " << (end - start)/(double)(CLOCKS_PER_SEC*60) << " minutes." << endl;
	/*sae.loadModel("SAE_Model.txt");
	buildImage(sae.theta1,imgWidth,"weights.bmp",true);*/
	cout << "lambda:" << lambda << endl;
	cout << "alpha:" << alpha << endl;
	cout << "beta:" << beta << endl;
	cout << "sparseParam:" << sp << endl;
	cout << "miniBatchSize:" << miniBatchSize << endl;
	system("pause");
	return 0;
}



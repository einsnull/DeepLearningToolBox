//training data:nDim * nExamples
#include <windows.h>
#include "dataAndImage.h"
#include "DAE.h"
#include "cmath"
#include "getConfig.h"

int main()
{
	//learning rate
	double alpha = 0.03;
	int maxIter = 15000;
	int miniBatchSize = 1000;
	double noiseRatio = 0.3;
	int inputSize = 28*28;
	int hiddenSize = 28;
	int imgWidth = 8; 
	char *fileBuf  = new char[4096];
	bool ret = loadFileToBuf("ParamConfig.ini",fileBuf,4096);
	if(ret)
	{
		getConfigDoubleValue(fileBuf,"noiseRatio:",noiseRatio);
		getConfigDoubleValue(fileBuf,"alpha:",alpha);
		getConfigIntValue(fileBuf,"maxIter:",maxIter);
		getConfigIntValue(fileBuf,"miniBatchSize:",miniBatchSize);
		getConfigIntValue(fileBuf,"hiddenSize:",hiddenSize);
		getConfigIntValue(fileBuf,"inputSize:",inputSize);
		getConfigIntValue(fileBuf,"imgWidth:",imgWidth);
		cout << "noiseRatio:" << noiseRatio << endl;
		cout << "alpha:" << alpha << endl;
		cout << "maxIter:" << maxIter << endl;
		cout << "miniBatchSize:" << miniBatchSize << endl;
		cout << "hiddenSize:" << hiddenSize << endl;
		cout << "inputSize:" << inputSize << endl;
		cout << "imgWidth:" << imgWidth << endl;
	}
	delete []fileBuf;

	//set eigen threads
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	Eigen::setNbThreads(info.dwNumberOfProcessors);

	MatrixXd trainData(1,inputSize);
	DAE dae(inputSize,hiddenSize);
	ret = loadMnistData(trainData,"mnist\\train-images-idx3-ubyte");
	cout << "Loading training data..." << endl;
	if(ret == false)
	{
		return -1;
	}
	clock_t start = clock();
	//cout << trainData.rows() << " " << trainData.cols() << endl; 
	MatrixXd showImage = trainData.leftCols(100).transpose();
	buildImage(showImage,imgWidth,"data.jpg");
	dae.train(trainData,noiseRatio,alpha,maxIter,miniBatchSize);
	cout << "End Train" << endl;
	MatrixXd hiddenTheta = dae.getTheta();
	buildImage(hiddenTheta,imgWidth,"weights.jpg",true);
	cout << "Saving hidden neurons" << endl;
	dae.saveModel("DAE_Model.txt");
	clock_t end = clock();
	cout << "The code ran for " << (end - start)/(double)(CLOCKS_PER_SEC*60)
		<< " minutes on " << Eigen::nbThreads() << " thread(s)." << endl;
	cout << "noiseRatio: " << noiseRatio << endl;
	cout << "alpha: " << alpha << endl;
	cout << "miniBatchSize: " << miniBatchSize << endl;
	system("pause");
	return 0;
}



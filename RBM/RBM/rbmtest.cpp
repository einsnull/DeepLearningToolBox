//training data:nExamples * ndim
#include <iostream>
#include "RBM.h"
#include "dataAndImage.h"
#include <windows.h>
using namespace std;

int main()
{
	int visbleSize = 28;
	int numHidden = 100;
	double epsilonW = 0.1;
	double epsilonVb = 0.1;
	double epsilonHb = 0.1;
	double weightCost = 2e-4;
	double initialMomentum = 0.5;
	double finalMomentum = 0.9;
	int imgWidth = visbleSize;
	int miniBatchSize = 100;
	int maxIter = 10;

#ifdef _WINDOWS_
	//set eigen threads
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	Eigen::setNbThreads(info.dwNumberOfProcessors);
#endif

	RBM rbm(visbleSize*visbleSize,numHidden);
	MatrixXd data(1,1);
	clock_t start = clock();
	cout << "Loading training data..." << endl;
		
	if(!loadMnistData(data,"mnist\\train-images-idx3-ubyte"))
	{
		return -1;
	}
	MatrixXd showData = data.leftCols(100).transpose();
	buildImage(showData,imgWidth,"data.jpg",false);
	MatrixXd trainData = data.transpose();
	rbm.train(trainData,miniBatchSize,maxIter,epsilonW,
		epsilonVb,epsilonHb,weightCost,initialMomentum,finalMomentum);
	MatrixXd weight = rbm.getWeight().transpose();
	buildImage(weight,imgWidth,"weights.jpg",false);

	clock_t end = clock();
	cout << "The code ran for " << 
		(end - start)/(double)(CLOCKS_PER_SEC*60) <<
		" minutes on " << Eigen::nbThreads() << " thread(s)." << endl;

	system("pause");
	return 0;
}
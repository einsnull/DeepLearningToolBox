#include <iostream>
#include "bpnn.h"
#include "getConfig.h"

bool loadDataSet(MatrixXd &x,MatrixXi &y,MatrixXd &xt,MatrixXi &yt,double per);

int main()
{
	/*You should take care of these paramters,
	it's a trick to make your model well when
	you don't have very big data set*/

	//regularization coefficient
	double lambda = 0;
	//learning rate
	double alpha = 0.1;
	int maxIter = 4000;
	//mini batch size (optional)
	int batchSize = 300;
	double trainSetPercent = 0.8;
	char *fileBuf = new char[4096];
	bool ret = loadFileToBuf("ParamConfig.ini",fileBuf,4096);
	if(ret)
	{
		getConfigDoubleValue(fileBuf,"lambda:",lambda);
		getConfigDoubleValue(fileBuf,"alpha:",alpha);
		getConfigIntValue(fileBuf,"maxIter:",maxIter);
		getConfigDoubleValue(fileBuf,"trainSetPercent:",trainSetPercent);
		cout << "lambda:" << lambda << endl;
		cout << "alpha:" << alpha << endl;
		cout << "maxIter:" << maxIter << endl;
		cout << "trainSetPercent:" << trainSetPercent << endl;
	}
	int n = 2;
	int input = 1260;
	int hidden = 10;
	int output = 14;
	
	MatrixXd trainData(n,input);
	MatrixXi label(n,1);
	MatrixXd testData(n,input);
	MatrixXi testLabel(n,1);
	BPNN bp(input,hidden,output);
	
	loadDataSet(trainData,label,testData,testLabel,trainSetPercent);
	
	clock_t start = clock();

	//train 
	bp.train(trainData,label,lambda,alpha,maxIter,GD,NULL,testData,testLabel);
	//predict
	MatrixXi pred = bp.predict(trainData);
	cout << "trainset accurancy:" << bp.calcAccurancy(pred,label) << endl;
	clock_t end = clock();
	cout << "The code ran for " << (end - start)/(double)(CLOCKS_PER_SEC*60) << "minutes" << endl;
	bp.saveModel("Model.txt");

	system("pause");
	return 0;
}

bool loadDataSet(MatrixXd &x,MatrixXi &y,MatrixXd &xt,MatrixXi &yt,double per)
{
	ifstream ifs("data.txt");
	if(!ifs)
	{
		return false;
	}
	cout << "Loading data..." << endl;
	int inputLayerSize;
	ifs >> inputLayerSize;
	int dataSetSize;
	ifs >> dataSetSize;
	MatrixXd data(dataSetSize,inputLayerSize);
	MatrixXi label(dataSetSize,1);
	for(int i = 0; i < dataSetSize; i++)
	{
		for(int j = 0; j < inputLayerSize; j++)
		{
			ifs >> data(i,j);
		}
	}
	for(int i = 0; i < dataSetSize; i++)
	{
		ifs >> label(i,0);
	}
	ifs.close();
	data = data - MatrixXd::Ones(data.rows(),data.cols())*0.5;

	int numTrainSet = (int)(per * data.rows());
	x.resize(numTrainSet,inputLayerSize);
	y.resize(numTrainSet,1);
	x = data.topRows(numTrainSet);
	y = label.topRows(numTrainSet);
	xt.resize(dataSetSize - numTrainSet,inputLayerSize);
	yt.resize(dataSetSize - numTrainSet,1);
	xt = data.bottomRows(dataSetSize - numTrainSet);
	yt = label.bottomRows(dataSetSize-numTrainSet);

	return true;
}

/*
1 0 0 0
0 1 0 1
*/


/*MatrixXd data = trainData.topRows(100);
	MatrixXi dl = label.topRows(100);
	bp.train(data,dl,output,0.000001,0.1,3000);
	MatrixXi pred = bp.predict(data);
	cout << "accurancy:" << bp.calcAccurancy(pred,dl) << endl;
	bp.saveModel("Model.txt");*/
	/*MatrixXd data = trainData.topRows(100);
	MatrixXi dl = label.topRows(100);
	MatrixXi pred = bp.predict(data);
	cout << "accurancy:" << bp.calcAccurancy(pred,dl) << endl;
	bp.loadModel("Model.txt");
	pred = bp.predict(data);
	cout << "accurancy:" << bp.calcAccurancy(pred,dl) << endl;*/




int min(MatrixXi &labels)
{
	int m = INT_MAX;
	int items = labels.rows();
	for(int i = 0; i < items; i++)
	{
		if(labels(i,0) < m)
		{
			m = labels(i,0);
		}
	}
	return m;
}

int max(MatrixXi &labels)
{
	int m = -1;
	int items = labels.rows();
	for(int i = 0; i < items; i++)
	{
		if(labels(i,0) > m)
		{
			m = labels(i,0);
		}
	}
	return m;
}
//training data:nDim * nExamples
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "LinearDecoder.h"
#include "ca.h"
using namespace std;
using namespace Eigen;

#pragma comment(lib,"opencv_core243.lib")
#pragma comment(lib,"opencv_highgui243.lib")
#pragma comment(lib,"opencv_imgproc243.lib")

bool loadDataSet(MatrixXd &data);
void buildColorImage(MatrixXd &theta,int imgWidth,char* szFileName,bool showFlag = false,double ratio = 1);

int main()
{
	//regularization coefficient
	double lambda = 0.003;
	//learning rate
	double alpha = 1;
	double beta = 5;
	double sp = 0.035;
	int maxIter = 1000;
	int miniBatchSize = 10000;//mini batch too small will cost Nan
	int imgWidth = 8;
	int inputSize = imgWidth * imgWidth * 3;
	int hiddenSize = 400;
	 
	MatrixXd data(1,1);
	loadDataSet(data);
	MatrixXd showData = data.topRows(100);
	//cout << showData.rows() << " " << showData.cols() << endl;
	buildColorImage(showData,imgWidth,"colordata.jpg");
	LinearDecoder ld(inputSize,hiddenSize);
	
	CA ca;
	MatrixXd x = data.transpose();
	//cout << tmpData.rows() << " " << tmpData.cols() << endl;
	cout << "Whitening..." << endl;
	//Apply ZCA Whitening
	FunctionBase fb;
	MatrixXd avg = x.rowwise().mean();
	x = fb.bsxfunMinus(x,avg);
	MatrixXd sigma = x * x.transpose() * (1.0 / x.cols());
	JacobiSVD<MatrixXd> svd(sigma, ComputeFullU | ComputeFullV);
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();
	MatrixXd S = svd.singularValues();
	double epsilon = 0.1;
	MatrixXd term1 = S + MatrixXd::Ones(S.rows(),S.cols()) * epsilon;
	MatrixXd term2 = fb.reciprocalMat(fb.sqrtMat(term1));
	MatrixXd xZCAWhiteMat = U * term2.asDiagonal() * U.transpose();
	MatrixXd trainData = xZCAWhiteMat * x;
	cout << "Training..." << endl;
	ld.train(trainData,lambda,alpha,beta,sp,maxIter,miniBatchSize);
	buildColorImage(ld.getTheta(),imgWidth,"weights.jpg");
	//should reimplenment zca
	MatrixXd feature = ld.getTheta() * xZCAWhiteMat;
	buildColorImage(feature,imgWidth,"feature.jpg");
	ld.saveModel("LinearDecoder_Model.txt");
	system("pause");
	return 0;
}

//data dim * numOfExamples
bool loadDataSet(MatrixXd &data)
{
	ifstream ifs("colorData.txt");
	if(!ifs)
	{
		return false;
	}
	cout << "Loading data..." << endl;
	int inputLayerSize;
	ifs >> inputLayerSize;
	int dataSetSize;
	ifs >> dataSetSize;
	data.resize(dataSetSize,inputLayerSize);
	for(int i = 0; i < dataSetSize; i++)
	{
		for(int j = 0; j < inputLayerSize; j++)
		{
			ifs >> data(i,j);
		}
	}
	ifs.close();
	return true;
}

void buildColorImage(MatrixXd &theta,int imgWidth,char* szFileName,bool showFlag,double ratio)
{
	int margin = 1;
	int channels = 3;
	double maxGrayVal = 255.0;
	int rows = theta.rows();
	int cols = theta.cols();
	if(rows <= 0 || cols <= 0)
	{
		return ;
	}
	/*cout << rows << endl;
	cout << cols << endl;*/
	double pr = sqrt((double)rows);
	int perRow = (int)pr + (pr - (int)pr > 0);
	double tc = (double)rows / (double)perRow;
	int tCols = (int)tc + (tc - (int)tc > 0);
	/*cout << "perRow: " << perRow << endl;
	cout << "tCols: " << tCols << endl;*/
	int ndim = cols/channels;
	//cout << "ndim " << ndim << endl;
	//theta = theta - theta.rowwise().mean().replicate(1,theta.cols());
	MatrixXd r = theta.leftCols(ndim);
	MatrixXd g = theta.middleCols(ndim,ndim);
	MatrixXd b = theta.rightCols(ndim);
	MatrixXd minR = r.rowwise().minCoeff();
	MatrixXd minG = g.rowwise().minCoeff();
	MatrixXd minB = b.rowwise().minCoeff();
	MatrixXd maxR = r.rowwise().maxCoeff();
	MatrixXd maxG = g.rowwise().maxCoeff();
	MatrixXd maxB = b.rowwise().maxCoeff();

	int imgHeight = ndim/imgWidth;
	IplImage* iplImage = cvCreateImage(
		cvSize(imgWidth * perRow + margin * (perRow+1),imgHeight * tCols + margin * (tCols + 1)),
		IPL_DEPTH_8U,channels);
	
	int step = iplImage->widthStep;
	uchar *data = (uchar *)iplImage->imageData;
	int h = iplImage->height;
	int w = iplImage->width;
	for(int x = 0; x < w; x++)
	{
		for(int y = 0; y < h; y++)
		{
			data[y * step + x * channels] = 0;
			data[y * step + x * channels + 1] = 0;
			data[y * step + x * channels + 2] = 0;
		}
	}

	for(int i = 0;i < rows;i++)
	{
		
		int n = 0;
		int hIdx = i / perRow;
		int wIdx = i % perRow;
		for(int j = 0;j < imgWidth;j++)
		{
			for(int k = 0;k < imgHeight; k++)
			{
		
				int val = (hIdx * imgHeight + k + margin * (hIdx+1)) * step
					+ (wIdx * imgWidth + j + margin * (wIdx+1)) * channels;
				if(val > step * (imgHeight * tCols + margin * (tCols + 1)))
				{
					cout << "error" << endl;
				}

				double per = (r(i,n) - minR(i,0)) / (double)(maxR(i,0) - minR(i,0));
				data[val + 2] = (uchar)(int)(per * maxGrayVal);
				per = (g(i,n) - minG(i,0)) / (double)(maxG(i,0) - minG(i,0));
				data[val + 1] = (uchar)(int)(per * maxGrayVal);
				per = (b(i,n) - minB(i,0)) / (double)(maxB(i,0) - minB(i,0));
				data[val] = (uchar)(int)(per * maxGrayVal);
				n ++;
			}
		}
		
	}
	
	cvSaveImage(szFileName,iplImage);

	if(showFlag)
	{
		cvNamedWindow(szFileName,CV_WINDOW_AUTOSIZE);
		IplImage* iplImageShow = cvCreateImage(
			cvSize((int)(iplImage->width * ratio),(int)(iplImage->height * ratio)),IPL_DEPTH_8U,channels);
		cvResize(iplImage,iplImageShow,CV_INTER_CUBIC);
		cvShowImage(szFileName,iplImageShow);
		cvWaitKey(100000);
		cvDestroyWindow(szFileName);
		cvReleaseImage(&iplImageShow);
	}
	cvReleaseImage(&iplImage);
}
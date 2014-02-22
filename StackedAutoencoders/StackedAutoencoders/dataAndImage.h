#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
using namespace std;
using namespace Eigen;

#pragma comment(lib,"opencv_core243.lib")
#pragma comment(lib,"opencv_highgui243.lib")
#pragma comment(lib,"opencv_imgproc243.lib")

bool loadMnistData(MatrixXd &data,char *szFileName)
{
	FILE *fp;
	fopen_s(&fp,szFileName,"rb");
	if(!fp)
	{
		cout << "Could not Open " << szFileName << endl;
		return false;
	}
	unsigned int magic = 0;
	unsigned char temp;
	unsigned int numOfImages = 0;
	unsigned int rows;
	unsigned int cols;
	for(int i = 0;i < 4;i++)
	{
		if(feof(fp))
		{
			fclose(fp);
			return false;
		}
		fread(&temp,sizeof(char),1,fp);
		magic = magic << 8 | temp;
	}
	//printf("magic: %d\n",magic);
	for(int i = 0;i < 4;i++)
	{
		if(feof(fp))
		{
			fclose(fp);
			return false;
		}
		fread(&temp,sizeof(char),1,fp);
		numOfImages = numOfImages << 8 | temp;
	}
	//printf("numOfImages: %d\n",numOfImages);
	for(int i = 0;i < 4;i++)
	{
		if(feof(fp))
		{
			fclose(fp);
			return false;
		}
		fread(&temp,sizeof(char),1,fp);
		rows = rows << 8 | temp;
	}
	//printf("rows: %d\n",rows);
	for(int i = 0;i < 4;i++)
	{
		if(feof(fp))
		{
			fclose(fp);
			return false;
		}
		fread(&temp,sizeof(char),1,fp);
		cols = cols << 8 | temp;
	}
	//printf("cols: %d\n",cols);
	data.resize(rows*cols,numOfImages);
	for(int i = 0;i < (int)numOfImages;i++)
	{
		for(int j = 0; j < (int)(rows * cols); j++)
		{
			if(feof(fp))
			{
				cout << "Error reading file" << endl;
				fclose(fp);
				return false;
			}
			fread(&temp,sizeof(char),1,fp);
			data(j,i) = (temp / 255.0);
		}
	}
	fclose(fp);
	return true;
}


bool loadMnistLabels(MatrixXi &labels,char *szFileName)
{
	FILE *fp;
	fopen_s(&fp,szFileName,"rb");
	if(!fp)
	{
		cout << "Could not Open " << szFileName << endl;
		return false;
	}
	unsigned int magic = 0;
	unsigned char temp;
	unsigned int numOfLabels = 0;
	for(int i = 0;i < 4;i++)
	{
		if(feof(fp))
		{
			fclose(fp);
			return false;
		}
		fread(&temp,sizeof(char),1,fp);
		magic = magic << 8 | temp;
	}
	for(int i = 0;i < 4;i++)
	{
		if(feof(fp))
		{
			fclose(fp);
			return false;
		}
		fread(&temp,sizeof(char),1,fp);
		numOfLabels = numOfLabels << 8 | temp;
	}
	//printf("numOfLabels: %d\n",numOfLabels);
	labels.resize(numOfLabels,1);
	for(int i = 0;i < (int)numOfLabels;i++)
	{
		if(feof(fp))
		{
			fclose(fp);
			return false;
		}
		fread(&temp,sizeof(char),1,fp);
		labels(i,0) = temp;
	}
	fclose(fp);
	return true;
}

void buildImage(MatrixXd &theta,int imgWidth,char* szFileName,bool showFlag = false,double ratio = 1)
{
	int margin = 1;
	int rows = theta.rows();
	int cols = theta.cols();
	if(rows <= 0 || cols <= 0)
	{
		return ;
	}

	double pr = sqrt((double)rows);
	int perRow = (int)pr + (pr - (int)pr > 0);
	double tc = (double)rows / (double)perRow;
	int tCols = (int)tc + (tc - (int)tc > 0);
	/*cout << "perRow: " << perRow << endl;
	cout << "tCols: " << tCols << endl;*/
	MatrixXd max = theta.rowwise().maxCoeff();
	MatrixXd min = theta.rowwise().minCoeff();

	int imgHeight = cols/imgWidth;
	IplImage* iplImage = cvCreateImage(
		cvSize(imgWidth * perRow + margin * (perRow+1),imgHeight * tCols + margin * (tCols + 1)),
		IPL_DEPTH_8U,1);
	
	int step = iplImage->widthStep;
	uchar *data = (uchar *)iplImage->imageData;
	int h = iplImage->height;
	int w = iplImage->width;
	for(int x = 0; x < w; x++)
	{
		for(int y = 0; y < h; y++)
		{
			data[y * step + x] = 0;
		}
	}

	for(int i = 0;i < rows;i++)
	{
		
		int n = 0;
		int hIdx = i / perRow;
		int wIdx = i % perRow;

		for(int j = 0;j < imgHeight;j++)
		{
			for(int k = 0;k < imgWidth; k++)
			{
				double per = (theta(i,n) - min(i,0) ) / (max(i,0) - min(i,0));
				//data[j * step + k] = 255;
				int val = (hIdx * imgHeight+j + margin * (hIdx+1)) * step
					+ (wIdx * imgWidth + k + margin * (wIdx+1));
				if(val > step * (imgHeight * tCols + margin * (tCols + 1)))
				{
					cout << "error" << endl;
				}
				data[val] = (uchar)(int)(per * 230.0);
				n ++;
			}
		}
		
	}
	
	cvSaveImage(szFileName,iplImage);

	if(showFlag)
	{
		cvNamedWindow(szFileName,CV_WINDOW_AUTOSIZE);
		IplImage* iplImageShow = cvCreateImage(
			cvSize((int)(iplImage->width * ratio),(int)(iplImage->height * ratio)),IPL_DEPTH_8U,1);
		cvResize(iplImage,iplImageShow,CV_INTER_CUBIC);
		cvShowImage(szFileName,iplImageShow);
		cvWaitKey(100000);
		cvDestroyWindow(szFileName);
		cvReleaseImage(&iplImageShow);
	}
	cvReleaseImage(&iplImage);
}

bool saveMatrix(MatrixXd &m,char *szFileName)
{
	ofstream ofs(szFileName);
	if(!ofs)
	{
		return false;
	}
	ofs << m.rows() << " " << m.cols() << endl;
	for(int i = 0; i < m.rows(); i++)
	{
		for(int j = 0; j < m.cols(); j++)
		{
			ofs << m(i,j) << " ";
		}
	}
	ofs.close();
	return true;
}

bool LoadMatrix(MatrixXd &m,char *szFileName)
{
	ifstream ifs(szFileName);
	if(!ifs)
	{
		return false;
	}
	int rows;
	int cols;
	ifs >> rows >> cols;
	m.resize(rows,cols);
	for(int i = 0; i < m.rows(); i++)
	{
		for(int j = 0; j < m.cols(); j++)
		{
			if(ifs.eof())
			{
				return false;
			}
			ifs >> m(i,j);
		}
	}
	ifs.close();
	return true;
}
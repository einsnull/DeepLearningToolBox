#include "getConfig.h"

bool loadFileToBuf(char *fileName,char *buf,int bufSize)
{
	FILE *fp;
	int size;
	fopen_s(&fp,fileName,"rb");
	if(fp)
	{
		fseek(fp,0,SEEK_END);
		size = ftell(fp);
		fseek(fp,0,SEEK_SET);
		if(size > bufSize-1)
		{
			fclose(fp);
			return false;
		}
		fread(buf,1,size,fp);
		fclose(fp);
		buf[size] = NULL;
		return true;
	}
	return false;
}

bool getConfigIntValue(char * buf,char *key,int &val)
{
	int bufLen = (int)strlen(buf);
	int keyLen = (int)strlen(key);
	int start = findStr(buf,key,bufLen,keyLen);
	int valueStart = start+keyLen;
	int end = findStr(buf,"\r",bufLen,1,valueStart);
	if(start == -1 || end == -1)
	{
		return false;
	}
	char str[500] = {0};
	int i,j;
	if(end - valueStart >= 500)
	{
		return false;
	}
	for(i = valueStart,j = 0; i < end; i++,j++)
	{
		str[j] = buf[i];
	}
	str[end] = 0;
	val = atoi(str);
	return true;
}

bool getConfigStrValue(char * buf,char *key,char *dstStr,int dstLen)
{
	int bufLen = (int)strlen(buf);
	int keyLen = (int)strlen(key);
	int start = findStr(buf,key,bufLen,keyLen);
	int valueStart = start+keyLen;
	int end = findStr(buf,"\r",bufLen,1,valueStart);
	if(start == -1 || end == -1)
	{
		return false;
	}
	if(end - valueStart >= dstLen)
	{
		return false;
	}
	int i,j;
	for(i = valueStart,j = 0; i < end; i++,j++)
	{
		dstStr[j] = buf[i];
	}
	dstStr[end] = 0;
	return true;
}

bool getConfigDoubleValue(char * buf,char *key,double &val)
{
	int bufLen = (int)strlen(buf);
	int keyLen = (int)strlen(key);
	int start = findStr(buf,key,bufLen,keyLen);
	int valueStart = start+keyLen;
	int end = findStr(buf,"\r",bufLen,1,valueStart);
	if(start == -1 || end == -1)
	{
		return false;
	}
	char str[500] = {0};
	int i,j;
	if(end - valueStart >= 500)
	{
		return false;
	}
	for(i = valueStart,j = 0; i < end; i++,j++)
	{
		str[j] = buf[i];
	}
	str[end] = 0;
	val = atof(str);
	return true;
}

bool getConfigMapTable(char * buf,char *key,int *dstTable,int dstLen)
{
	int bufLen = (int)strlen(buf);
	int keyLen = (int)strlen(key);
	int start = findStr(buf,key,bufLen,keyLen);
	int valueStart = start+keyLen;
	int end = findStr(buf,"\r",bufLen,1,valueStart);
	if(start == -1 || end == -1)
	{
		return false;
	}
	char str[500];
	if(end - valueStart >= 500)
	{
		return false;
	}
	int i,j;
	int idx = 0;
	int x = 0;
	for(i = valueStart,j = 0; i <= end; i++,j++)
	{
		if(buf[i] == ',' || buf[i] == '\r')
		{
			str[x] = 0;
			x = -1;
			if(idx >= dstLen)
			{
				return false;
			}
			dstTable[idx++] = atoi(str);
		}
		else
		{
			str[x] = buf[i];
		}
		x++;
	}
	return true;
}
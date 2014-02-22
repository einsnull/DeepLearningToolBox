#pragma once
#include <stdlib.h>

//查找串
int findStr(char *src,char *dst,int srcLen,int dstLen,int startPos = 0);

//获取串中的double数据
double getDouble(char *str,char *startStr,char *endStr,int strLen,int startStrLen,int endStrLen,int startPos = 0);

//获取串中的int数据
int getInt(char *str,char *startStr,char *endStr,int strLen,int startStrLen,int endStrLen,int startPos = 0);

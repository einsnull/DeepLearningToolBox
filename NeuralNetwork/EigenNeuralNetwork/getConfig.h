#include "myString.h"
#include <string.h>
#include <stdio.h>

bool loadFileToBuf(char *fileName,char *buf,int bufSize);

bool getConfigIntValue(char * buf,char *key,int &val);

bool getConfigStrValue(char * buf,char *key,char *dstStr,int dstLen);

bool getConfigDoubleValue(char * buf,char *key,double &val);

bool getConfigMapTable(char * buf,char *key,int *dstTable,int dstLen);

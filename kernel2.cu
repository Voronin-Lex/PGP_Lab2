#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X,Y)  ((X) > (Y)   ? (X) : (Y))


__host__ double* GetOneDimKernel(int Radius)
{
	int KerSize=2*Radius+1;
	double* GausKernel=(double*)malloc(KerSize*sizeof(double));

	const double PI = acos(-1.0);

	double denominator = Radius*sqrt(2.0*PI);
	double TwoPowRad = 2.0*Radius*Radius;

	double SumKernel=0;

	for(int i=-Radius; i<Radius+1; i++)
	{
		GausKernel[i+Radius] = exp(-i*i/TwoPowRad)/denominator;
		SumKernel+=GausKernel[i+Radius];
	}

	for(int i=-Radius; i<Radius+1; i++)
	{
		GausKernel[i+Radius]/=SumKernel;
	}

	double SumNormKernel=0;

	for(int i=-Radius; i<Radius+1; i++)
		SumNormKernel+=GausKernel[i+Radius];

	//printf("\n %lf ", SumNormKernel);

	return GausKernel;


}

__global__ void FirstPass(unsigned char* DevSourceImg, double* DevTmpImg, double* Kernel, int W, int H, int Radius)
{
	int CountElementsInStr=W*4;
	int i,j;

	   
	for(int offset = 4*(blockIdx.x*blockDim.x+threadIdx.x); offset<H*CountElementsInStr; offset+=4*(gridDim.x*blockDim.x))
		{
			i=offset/CountElementsInStr;
			j=offset%CountElementsInStr;

		  double RedSum=0;
		  double GreenSum=0;
		  double BlueSum=0;

		  for(int k=-Radius; k<Radius+1; k++)
		  {

			  int StrInImage = MAX(MIN(i+k, H-1), 0);


			  RedSum+=Kernel[Radius+k]*DevSourceImg[StrInImage*CountElementsInStr+j];
			  GreenSum+=Kernel[Radius+k]*DevSourceImg[StrInImage*CountElementsInStr+j+1];
			  BlueSum+=Kernel[Radius+k]*DevSourceImg[StrInImage*CountElementsInStr+j+2];
		  
		  }
		  DevTmpImg[i*CountElementsInStr+j] = RedSum;
		  DevTmpImg[i*CountElementsInStr+j+1] = GreenSum;
		  DevTmpImg[i*CountElementsInStr+j+2] = BlueSum;
		}
}

__global__ void SecondPass(double* DevTmpImg, unsigned char* DevTargetImg, double* Kernel, int W, int H, int Radius)
{
         int CountElementsInStr=W*4; 
		 int i,j;

		 for(int offset=4*(blockIdx.x*blockDim.x+threadIdx.x); offset<H*CountElementsInStr; offset+=4*(gridDim.x*blockDim.x))
         {
				i=offset/CountElementsInStr;
			    j=offset%CountElementsInStr;


			   double RedSum=0;
			   double GreenSum=0;
		       double BlueSum=0;
			
			   for(int k=-Radius; k<Radius+1; k++)
			   {
				   int ColumnIndex = MAX(MIN(j+k*4, CountElementsInStr-4), 0);

				   RedSum+= Kernel[Radius+k]*DevTmpImg[i*CountElementsInStr+ColumnIndex];  
				   GreenSum+=Kernel[Radius+k]*DevTmpImg[i*CountElementsInStr+ColumnIndex+1];  
				   BlueSum+= Kernel[Radius+k]*DevTmpImg[i*CountElementsInStr+ColumnIndex+2];  
			   }
			
			   DevTargetImg[i*CountElementsInStr+j] = RedSum;
			   DevTargetImg[i*CountElementsInStr+j+1] = GreenSum;
			   DevTargetImg[i*CountElementsInStr+j+2] = BlueSum;
			   DevTargetImg[i*CountElementsInStr+j+3] = 0;
			}

}

*/
int main()
{
    char path1[256];
	char path2[256];

	int Radius;
	int width;
	int heigh;

	scanf("%s", path1); 
	FILE* InFile = fopen(path1, "rb");

	if (InFile == NULL)
	{
		fprintf(stderr, "Cannot open InFile");
		exit(0);
	}

	scanf("%s", path2);
	FILE* OutFile = fopen(path2, "wb");

	if (OutFile == NULL)
	{
		fprintf(stderr, "Cannot open OutFile");
		exit(0);
	}

	scanf("%d", &Radius);
   
	fread(&width, sizeof(int), 1, InFile);
	fread(&heigh, sizeof(int), 1, InFile);
	

	unsigned char *Image = (unsigned char*)malloc(4*width*heigh*sizeof(unsigned char)); //сюда загружаем картинку
	fread(Image, sizeof(unsigned char), 4*width*heigh, InFile);


	if(Radius==0)
	{

		fwrite(&width, sizeof(int), 1, OutFile);
		fwrite(&heigh, sizeof(int), 1, OutFile);
		fwrite(Image, 4*width*heigh*sizeof(unsigned char), 1, OutFile);	
		return 0;
	}

	unsigned char* DevSourceImg;     
	double* DevTmpImg;                
	unsigned char* DevTargetImg;    
	double* DevBlurKernel;          


	unsigned char* HostTargetImg;    



	HostTargetImg = (unsigned char*)malloc(4*width*heigh*sizeof(unsigned char));

	cudaMalloc((void**)&DevSourceImg, 4*width*heigh*sizeof(unsigned char));
	cudaMalloc((void**)&DevTmpImg, 4*width*heigh*sizeof(double));
	cudaMalloc((void**)&DevTargetImg, 4*width*heigh*sizeof(unsigned char));
	cudaMalloc((void**)&DevBlurKernel, (2*Radius+1)*sizeof(double));

	cudaMemcpy(DevSourceImg, Image, 4*width*heigh*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
		
	double* GausKernel = GetOneDimKernel(Radius);
	cudaMemcpy(DevBlurKernel, GausKernel, (2*Radius+1)*sizeof(double), cudaMemcpyHostToDevice);
	
	FirstPass<<<128,256>>>(DevSourceImg, DevTmpImg, DevBlurKernel, width, heigh, Radius); 
	SecondPass<<<128,256>>>(DevTmpImg, DevTargetImg, DevBlurKernel, width, heigh, Radius);  

	cudaMemcpy(HostTargetImg, DevTargetImg, 4*width*heigh*sizeof(unsigned char), cudaMemcpyDeviceToHost);


	fwrite(&width, sizeof(int), 1, OutFile); 
	fwrite(&heigh, sizeof(int), 1, OutFile);
	fwrite(HostTargetImg, 4*width*heigh*sizeof(unsigned char), 1, OutFile);

    return 0;
}


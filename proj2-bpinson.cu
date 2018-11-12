/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the rc machines
   ==================================================================
*/

/*
	Brian Pinson
	U91813366
	Project 2
	06/28/2018

*/




#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */



/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
bucket * histogram_CUDA;	/* list of all buckets in the histogram caluclated by CUDA	*/
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */




__constant__ long long PDH_acnt_CUDA;
__constant__ double PDH_res_CUDA;
extern __shared__ atom sharedMemory[];





int BLOCK_SIZE;

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}








/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}




/*
        CUDA SDH solution
*/

__global__ void CUDA_PDH_baseline(bucket *histogram_cuda, atom *atom_list_CUDA, int cuda_block_size, int cuda_block_number, int bucket_num)
{


        int b = blockIdx.x; // block id
	int B = blockDim.x;
	int t = threadIdx.x; // thread id

	if(b*B + t < PDH_acnt_CUDA)
	{


		atom *L = (atom *)sharedMemory;
		atom *R = (atom *)&L[cuda_block_size];
		int *SHMOut = (int *)&R[cuda_block_size];

		int z = 0;
		for(;z < bucket_num;z++) SHMOut[z] = 0;

		L[t] = atom_list_CUDA[b*B+t];
		register atom reg = L[t];


		int h_pos = 0;
		double dist = 0;

		int i = b+1;
		for(; i < cuda_block_number; i++)
		{
	
			R[t] = atom_list_CUDA[i*B+t];

			__syncthreads();

			int j = 0;
			for(; j < cuda_block_size && i*B+j < PDH_acnt_CUDA; j++)
			{
        			dist = sqrt((reg.x_pos-R[j].x_pos)*(reg.x_pos-R[j].x_pos)+(reg.y_pos-R[j].y_pos)*(reg.y_pos-R[j].y_pos)+
					(reg.z_pos-R[j].z_pos)*(reg.z_pos-R[j].z_pos));
				h_pos = (int) (dist / PDH_res_CUDA);
				atomicAdd(&SHMOut[h_pos], 1);
			}
			__syncthreads();
		}

	
		for(i = t + 1; i < cuda_block_size && b*B+i<PDH_acnt_CUDA; i++)
		{
			
        		dist = sqrt((reg.x_pos-L[i].x_pos)*(reg.x_pos-L[i].x_pos)+(reg.y_pos-L[i].y_pos)*(reg.y_pos-L[i].y_pos)+
				(reg.z_pos-L[i].z_pos)*(reg.z_pos-L[i].z_pos));

			h_pos = (int) (dist / PDH_res_CUDA);
			atomicAdd(&SHMOut[h_pos], 1);
		}

		__syncthreads();

		if(threadIdx.x == 0)
		{
			for( i = 0; i < bucket_num; i++)
			{
				atomicAdd((unsigned long long int*) &histogram_cuda[i].d_cnt, (unsigned long long int) SHMOut[i]);
			}
		}
	}
}






/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
        printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

double report_running_time_GPU() {
        long sec_diff, usec_diff;
        gettimeofday(&endTime, &Idunno);
        sec_diff = endTime.tv_sec - startTime.tv_sec;
        usec_diff= endTime.tv_usec-startTime.tv_usec;
        if(usec_diff < 0) {
                sec_diff --;
                usec_diff += 1000000;
        }
        printf("Running time for GPU version: %ld.%06ld\n", sec_diff, usec_diff);
        return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/*
        print the counts in all buckets of the CUDA histogram
*/
void output_histogram(bucket *input) /* CUDA GPU histogram printer*/
{
        int i;
        long long total_cnt = 0;
        for(i=0; i< num_buckets; i++) {
                if(i%5 == 0) /* we print 5 buckets in a row */
                        printf("\n%02d: ", i);
                printf("%15lld ", input[i].d_cnt);
                total_cnt += input[i].d_cnt;
                /* we also want to make sure the total distance count is correct */
                if(i == num_buckets - 1)
                        printf("\n T:%lld \n", total_cnt);
                else printf("| ");
        }
}



__host__ void  CUDA_Histogram_Calculator()
{

	cudaMemcpyToSymbol (PDH_acnt_CUDA, &PDH_acnt, sizeof(signed long long));

	cudaMemcpyToSymbol (PDH_res_CUDA, &PDH_res, sizeof(double));


	bucket *cuda_histogram = NULL; /* Mallocs histogram in GPU */
	cudaMalloc((void **) &cuda_histogram, num_buckets * sizeof(bucket));
	cudaMemcpy(cuda_histogram, histogram_CUDA, num_buckets * sizeof(bucket), cudaMemcpyHostToDevice);


	atom *cuda_atom_list = NULL; /* Mallocs atom list in GPU */
	cudaMalloc((void **) &cuda_atom_list, PDH_acnt * sizeof(atom));
	cudaMemcpy(cuda_atom_list, atom_list, PDH_acnt * sizeof(atom), cudaMemcpyHostToDevice);


	if(BLOCK_SIZE > 1023)
	{
		printf("ERROR maximum size of each block is 1024\n");
		BLOCK_SIZE = 1023;
	}
	if(BLOCK_SIZE < 1)
	{
		printf("ERROR block size is less than 1\n");
		BLOCK_SIZE = 1;
	}




	int cuda_block_size = BLOCK_SIZE;
	int cuda_block_number = ceil(PDH_acnt/BLOCK_SIZE) + 1;
	int cuda_shared_memory = (cuda_block_size) * sizeof(atom)*2 + num_buckets * sizeof(int);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	CUDA_PDH_baseline <<<cuda_block_number, BLOCK_SIZE, cuda_shared_memory>>> (cuda_histogram, cuda_atom_list, cuda_block_size, cuda_block_number, num_buckets);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %0.5f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	cudaMemcpy(histogram_CUDA, cuda_histogram, num_buckets * sizeof(bucket), cudaMemcpyDeviceToHost);
	
	cudaFree(cuda_histogram);
	cudaFree(cuda_atom_list);

}


/* Prints difference between buckets by using an altered histogram printing function */
void histogram_comparison(bucket *input1, bucket *input2)
{
        int i;
        long long total_cnt = 0;
        for(i=0; i< num_buckets; i++) {
                if(i%5 == 0) /* we print 5 buckets in a row */
                        printf("\n%02d: ", i);
                printf("%15lld ", input2[i].d_cnt - input1[i].d_cnt);
                total_cnt += input1[i].d_cnt;
                /* we also want to make sure the total distance count is correct */
                if(i == num_buckets - 1)
                        printf("\n T:%lld \n", total_cnt);
                else printf("| ");
        }
}




int main(int argc, char **argv)
{
	int i;
	printf("\nCPU Results:\n");

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	BLOCK_SIZE = atoi(argv[3]);
	//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

	histogram_CUDA = (bucket *)malloc(sizeof(bucket)*num_buckets);

	int z = 0; /* initalizes both histograms to zero */
	for(z = 0; z < num_buckets; z++)
	{
		histogram[z].d_cnt = 0;
		histogram_CUDA[z].d_cnt = 0;
	}

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram(histogram);



	/* GPU Histogram Calculator */
	printf("\n\nGPU Kernel results:\n");
	CUDA_Histogram_Calculator();
	output_histogram(histogram_CUDA);
	printf("\n");


	/* Comapring Histograms */
	printf("Difference between histogram calculated using CUDA and CPU\n");
	histogram_comparison(histogram, histogram_CUDA);

	free(histogram);
	free(histogram_CUDA);
	free(atom_list);

	return 0;
}

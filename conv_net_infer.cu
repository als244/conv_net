#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

// #pragma unroll

#define CHUNK_SIZE 64

#define SM_COUNT 82
#define WARP_PER_SM 4
#define THREAD_PER_WARP 32
#define MAX_THREAD_PER_BLOCK 1024



// ASSUME BLOCK + THREAD ARE BOTH 1-D
// very un-optimized, but good for testing...
// zero out matrix before matrix multiply
// blockDim.x * blockIdx.x + threadIdx.x represents index in 1-D array of AB
__global__  void matMulSimp(int M, int K, int N, float *A, float *B, float *out){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / N;
  int colInd = i % N;
  if (i < M * N){
    out[rowInd * N + colInd] = 0;
    for (int i = 0; i < K; i++){
      out[rowInd * N + colInd] += A[rowInd * K + i] * B[i * N + colInd];
    }
  }
}


// ASSUME BLOCK + THREAD ARE 1-D
// can work with vectors as well
// size is total number of elements (nRows * nCols), width = nCols
__global__ void transposeSimp(int size, int width, float * M, float *out){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / width;
  int colInd = i % width;
  int height = size / width;
  if (i < size){
    out[height * colInd + rowInd] = M[width * rowInd + colInd];
  }
}

// size is the number of samples = number of columns (minibatch size)
// rowInd is the row value in each sample to set to null
__global__  void setRowVal(int width, int rowInd, float *A, float val){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width){
      A[width * rowInd + i] = val;  
  }
}

// ASSUME BLOCK + THREAD ARE 1-D
// size is batch size
// one thread per sample in batch
__global__ void softMax(int size, int output_len, float*X){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
    float sum = 0;
    for (int j = 0; j < output_len; j++){
      sum += __expf(X[i + size * j]);
    }
    for (int j = 0; j < output_len; j++){
      X[i + size * j] = __expf(X[i + size * j]) / sum;
    }
  }
}


// ASSUME BLOCK + THREAD ARE 1-D
// can work with vectors as well
// size is total number of elements (nRows * nCols), width = nCols
__global__  void addBias(int size, int width, float *X, float *B){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / width;
  int colInd = i % width;
  if (i < size){
    X[width * rowInd + colInd] = X[width * rowInd + colInd] + B[rowInd];
  }
}

// ASSUME BLOCK + THREAD ARE 1-D
// can work with vectors as well
// size is total number of nodes in outputted layer, width = nCols
__global__  void addBiasAndActivate(int size, int width, float *X, float *B){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / width;
  int colInd = i % width;
  float z;
  if (i < size){
    X[width * rowInd + colInd] += B[rowInd];
    z = X[width * rowInd + colInd];
    X[width * rowInd + colInd] = (__expf(z) - __expf(-z)) / (__expf(z) + __expf(-z));
  }
}



// size is the batch size (number of values in loss array)
// output_len is the number of nodes per sample (10 in MNIST Case)
// X is the value of output nodes (output_len X batch size), Y is the correct labels (output_len X batch size)
__global__  void computeLoss(int size, int output_len, float *X, float *Y, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
    float sum, diff;
    sum = 0;
    for (int j = 0; j < output_len; j++){
      diff = (Y[i + j * size] - X[i + j * size]);
      sum += (diff * diff);
    }
    out[i] = sum;
  }
}



// ASSUME BLOCK + THREAD ARE 1-D
// size is the batch size (number of values in loss array)
// output_len is the number of nodes per sample (10 in MNIST Case)
// X is the value of output nodes (one prediction per sample [total length of "batch size"])
// taking index of max value in column to make prediction...
__global__  void makePredict(int size, int output_len, float *X, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float pred_val = -1;
  float pred_ind = -1;
  float val;
  if (i < size){
  	for (int j = 0; j < output_len; j++){
  		val = X[i + j * size];
  		if (val > pred_val) {
  			pred_val = val;
  			pred_ind = j;
  		}
  	}
  	out[i] = pred_ind;
  }
}




int main(void)
{

	// INITIALIZE MODEL...

	// DEFINE ARCHITECURAL PARAMTERS FOR NEURAL NET 

  	// mini batch size
  	int batch_size = 10;

  	int input_len = 257;
  	int output_len = 10;

  	int h1_size = 769;
  	int h2_size = 192;
  	int h3_size = 30;

  	/*
  	int input_null_ind = 256;
  	int h1_null_ind = 768;
  	int h1_map_dim = 8;
  	int h2_map_dim = 4;
  	*/

  	int h1_maps = 12;
  	int h1_kernel_dim = 5;
  	
  	int h2_maps = 12;
  	int h2_kernel_dim = 5;
  	
  	int W_h1_size = h1_maps * (h1_kernel_dim * h1_kernel_dim);
  	int W_h1_full_size = input_len * h1_size;
  	int W_h2_size = h2_maps * (h2_kernel_dim * h2_kernel_dim);
  	int W_h2_full_size = h1_size * h2_size;
  	int W_h3_size = h2_size * h3_size;
  	int W_out_size = h3_size * output_len;

  	// input and labels
  	float *X_in_host, *X_out_host, *Y_out_host;
  	X_in_host = (float*)malloc(input_len * batch_size *sizeof(float));
  	X_out_host = (float*)malloc(output_len * batch_size * sizeof(float));
  	Y_out_host = (float*)malloc(output_len * batch_size * sizeof(float));

  	// predicted values
  	float *predicted_host;
  	predicted_host = (float*)malloc(batch_size *sizeof(float));

  	// weights
  	float *W_h1_host, *W_h2_host, *W_h3_host, *W_out_host;
  	W_h1_host = (float *)malloc(W_h1_size * sizeof(float));
  	W_h2_host = (float *)malloc(W_h2_size * sizeof(float));
  	W_h3_host = (float *)malloc(W_h3_size * sizeof(float));
  	W_out_host = (float *)malloc(W_out_size * sizeof(float));

  	float *W_h1_full_host, *W_h2_full_host;
  	W_h1_full_host = (float *)malloc(W_h1_full_size * sizeof(float));
  	W_h2_full_host = (float *)malloc(W_h2_full_size * sizeof(float));


  	// biases
  	float *B_h1_host, *B_h2_host, *B_h3_host, *B_out_host;
  	B_h1_host = (float *)malloc(h1_size * sizeof(float));
  	B_h2_host = (float *)malloc(h2_size * sizeof(float));
  	B_h3_host = (float *)malloc(h3_size * sizeof(float));
  	B_out_host = (float *)malloc(output_len * sizeof(float));

  	// loss
  	float *loss_host;
  	loss_host = (float *)malloc(batch_size * sizeof(float));



  	// GPU MEMORY...

  	// nodes
  	float *X_in, *X_h1, *X_h2, *X_h3, *X_out, *Y_out;

  	// weights
  	float *W_h1_full, *W_h2_full, *W_h3, *W_out;

  	// biases
  	float *B_h1, *B_h2, *B_h3, *B_out;

  	// allocate space for input/output from data
  	cudaMalloc(&X_in, input_len * batch_size * sizeof(float));
  	cudaMalloc(&Y_out, output_len * batch_size* sizeof(float));

  	// allocate space for hidden nodes
  	cudaMalloc(&X_h1, h1_size * batch_size*sizeof(float));
  	cudaMalloc(&X_h2, h2_size * batch_size*sizeof(float));
  	cudaMalloc(&X_h3, h3_size * batch_size*sizeof(float));
  	cudaMalloc(&X_out, output_len * batch_size*sizeof(float));
  	
  	// allocate space for weights
  	cudaMalloc(&W_h1_full, W_h1_full_size * sizeof(float));
  	cudaMalloc(&W_h2_full, W_h2_full_size * sizeof(float));
  	cudaMalloc(&W_h3, W_h3_size * sizeof(float));
  	cudaMalloc(&W_out, W_out_size * sizeof(float));

  	// allocate space for biases
  	cudaMalloc(&B_h1, h1_size * sizeof(float));
  	cudaMalloc(&B_h2, h2_size * sizeof(float));
  	cudaMalloc(&B_h3, h3_size * sizeof(float));
  	cudaMalloc(&B_out, output_len * sizeof(float));


  	// for reading raw data in...
  	float *X_in_T, *Y_out_T;

  	cudaMalloc(&X_in_T, input_len * batch_size * sizeof(float));
  	cudaMalloc(&Y_out_T, output_len * batch_size* sizeof(float));

  	// loss data
  	float *loss;
  	// allocate space to store values for loss function per sample
  	cudaMalloc(&loss, batch_size * sizeof(float));

  	// predicted values
  	float *predicted;
  	cudaMalloc(&predicted, batch_size * sizeof(float));



	// NEED TO LOAD WEIGHTS + BIASES FROM SAVED MODEL
	const char * model_path = "/mnt/storage/data/image_text/mnist/trained_model";

  	FILE * model_file = fopen(model_path, "rb+");

  	// read in weights
  	fread(W_h1_host, sizeof(float), W_h1_size, model_file);
  	fread(W_h1_full_host, sizeof(float), W_h1_full_size, model_file);  
  	fread(W_h2_host, sizeof(float), W_h2_size, model_file);
  	fread(W_h2_full_host, sizeof(float), W_h2_full_size, model_file);
  	fread(W_h3_host, sizeof(float), W_h3_size, model_file);
  	fread(W_out_host, sizeof(float), W_out_size, model_file);

  	// read in biases
  	fread(B_h1_host, sizeof(float), h1_size, model_file);
  	fread(B_h2_host, sizeof(float), h2_size, model_file);
  	fread(B_h3_host, sizeof(float), h3_size, model_file);
  	fread(B_out_host, sizeof(float), output_len, model_file);

  	fclose(model_file);


  	// copy weights + biases to GPU...
  	cudaMemcpy(W_h1_full, W_h1_full_host, W_h1_full_size * sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(W_h2_full, W_h2_full_host, W_h2_full_size * sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(W_h3, W_h3_host, W_h3_size * sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(W_out, W_out_host, W_out_size * sizeof(float), cudaMemcpyHostToDevice);

  
  	cudaMemcpy(B_h1, B_h1_host, h1_size * sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(B_h2, B_h2_host, h2_size * sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(B_h3, B_h3_host, h3_size * sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(B_out, B_out_host, output_len * sizeof(float), cudaMemcpyHostToDevice);


	// TEST DATA
	const char * test_images_path = "/mnt/storage/data/image_text/mnist/t10k-images.idx3-ubyte";
  	const char * test_labels_path = "/mnt/storage/data/image_text/mnist/t10k-labels.idx1-ubyte";

  	// LOAD IN TEST IMAGES + LABELS, PREDICT, CALCULATE LOSS...

  	FILE * test_images_file, *test_labels_file;
  	unsigned char * test_images_raw, *test_labels_raw;
  	float *test_images, *test_labels;

  	test_images_file = fopen(test_images_path, "r");
  	test_labels_file = fopen(test_labels_path, "r");

  	// from "http://yann.lecun.com/exdb/mnist/"
  	off_t test_images_offset = 16;
  	off_t test_labels_offset = 8;

  	// skipping offset bytes in beginning then measuring til end = skipping offset bytes in end and measuring from start
  	fseek(test_images_file, test_images_offset, SEEK_END);
  	long test_images_nbytes = ftell(test_images_file);
  	fseek(test_labels_file, test_labels_offset, SEEK_END);
  	long test_labels_nbytes = ftell(test_labels_file);

 	 // raw because going to downsample..
  	test_images_raw = (unsigned char *) calloc(test_images_nbytes, sizeof(unsigned char));
  	test_labels_raw = (unsigned char *) calloc(test_labels_nbytes, sizeof(unsigned char));

  	// set to beginning...
  	fseek(test_images_file, test_images_offset, SEEK_SET);
  	fseek(test_labels_file, test_labels_offset, SEEK_SET);

  	fread(test_images_raw, sizeof(unsigned char), test_images_nbytes, test_images_file);
  	fclose(test_images_file);

  	fread(test_labels_raw, sizeof(unsigned char), test_labels_nbytes, test_labels_file);
  	fclose(test_labels_file);


  	int test_n = 10000;
  	

  	int image_raw_dim = 28;
  	int image_dim = 16;
  	float ratio = float(image_raw_dim) / float(image_dim);

  	int raw_floor_row, raw_floor_col, top_left, pixel_ind;
  	float ave_raw_pixel, pixel_val;

  	// store images as array of 16*16 images, but then additional -1 input
  	test_images = (float *) calloc(test_n * input_len, sizeof(float));


  	for (int img_num = 0; img_num < test_n; img_num++){
    	//printf("Img Num: %i\n\n", img_num);
    	for (int i = 0; i < image_dim; i++){
      		for (int j = 0; j < image_dim; j++){
        		// averaging 4 closest pixels in original image
        		raw_floor_row = floor(i * ratio);
        		raw_floor_col = floor(j * ratio);
        		top_left = img_num * (image_raw_dim * image_raw_dim) + image_raw_dim * raw_floor_row + raw_floor_col;
        		ave_raw_pixel = (float)(((float)test_images_raw[top_left] + (float)test_images_raw[top_left + 1] + (float)test_images_raw[top_left + image_raw_dim] + (float)test_images_raw[top_left + image_raw_dim + 1]) / float(4));
        		// scale to be between -1 and 1
        		pixel_val = ave_raw_pixel * (2.0 / 255.0) - 1;
        		// storing average pixel value into downsampled array
        		pixel_ind = img_num * input_len + image_dim * i + j;
        		test_images[pixel_ind] = pixel_val;
      		}
    	}
    	// have last value in input image be a -1
    	test_images[img_num * input_len + (image_dim * image_dim)] = -1;
  	}
  	// stored in downsampled test_images so can free now
  	free(test_images_raw);

  	// store image labels as series of 10 floats
  	test_labels = (float *) calloc(test_n * output_len, sizeof(float));

  	int label;
  	for (int img_num = 0; img_num < test_n; img_num++){
    	label = test_labels_raw[img_num];
    	for (int dig = 0; dig < output_len; dig++){
      		if (label == dig){
        		test_labels[img_num * output_len + dig] = 1.0;
      		}
      		else { 
        		test_labels[img_num * output_len + dig] = 0.0;
      		}
    	}
  	}
  	




  	// DO INFERENCE...


  	int n_batches = test_n / batch_size;
  	float n_wrong = 0;
  	bool *is_wrong_host = (bool *) calloc(test_n, sizeof(bool));
  	float totalLoss = 0;
  	for (int batch_i = 0; batch_i < n_batches; batch_i++) {

  		printf("\n\nBatch: %d\n\n", batch_i);

  		// ENSURE GPU MEM (X_in, Y_out) contain each sample as columns 
  		// (X_in has minibatch column with input_len rows, Y_out has minibatch columns with output_len rows)
  		memcpy(X_in_host, test_images + batch_i * batch_size * input_len, batch_size * input_len * sizeof(float));
        memcpy(Y_out_host, test_labels + batch_i * batch_size * output_len, batch_size * output_len * sizeof(float));
        // read in as consective images (so pixels are rows). want to transpose, then send back to host (for debug purposes...)
        cudaMemcpy(X_in_T, X_in_host, input_len *batch_size* sizeof(float), cudaMemcpyHostToDevice);
        transposeSimp<<< SM_COUNT, ceil((float)input_len * batch_size / SM_COUNT)>>> (input_len * batch_size, input_len, X_in_T, X_in);
        cudaMemcpy(X_in_host, X_in, input_len *batch_size* sizeof(float), cudaMemcpyDeviceToHost);
        // read in as consective sequences of output lables, want to transpose
        cudaMemcpy(Y_out_T, Y_out_host, output_len*batch_size*sizeof(float), cudaMemcpyHostToDevice);
        transposeSimp<<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>> (output_len * batch_size, output_len, Y_out_T, Y_out);
        cudaMemcpy(Y_out_host, Y_out, output_len *batch_size* sizeof(float), cudaMemcpyDeviceToHost);


        // PERFORM SERIES OF MAT_MULS with ACTIVATION FUNCTION TO PREDICT...


        // X_in to X_h1
        matMulSimp<<< SM_COUNT, ceil((float)h1_size * batch_size / SM_COUNT) >>>(h1_size, input_len, batch_size, W_h1_full, X_in, X_h1);
        addBiasAndActivate <<< SM_COUNT, ceil((float)h1_size * batch_size / SM_COUNT) >>>(h1_size * batch_size, batch_size, X_h1, B_h1);
        // set constant of -1 to last val in h1 for all samples in batch
        setRowVal<<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, h1_size - 1, X_h1, -1.0);

        // X_h1 to X_h2
        matMulSimp<<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT) >>>(h2_size, h1_size, batch_size, W_h2_full, X_h1, X_h2);
        addBiasAndActivate <<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT) >>>(h2_size * batch_size, batch_size, X_h2, B_h2);

        // X_h2 to X_h3
        matMulSimp<<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT) >>>(h3_size, h2_size, batch_size, W_h3, X_h2, X_h3);
        addBiasAndActivate <<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT) >>>(h3_size * batch_size, batch_size, X_h3, B_h3);

        // X_h3 to X_out
        matMulSimp<<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len, h3_size, batch_size, W_out, X_h3, X_out);
        addBias <<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>> (output_len * batch_size, batch_size, X_out, B_out);

        softMax <<< SM_COUNT, ceil((float)batch_size / SM_COUNT) >>> (batch_size, output_len, X_out);
        // get network outputs back on host...
        cudaMemcpy(X_out_host, X_out, output_len * batch_size * sizeof(float), cudaMemcpyDeviceToHost);


        // TO compare to TRAINING, use the same loss function...
        computeLoss <<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, output_len, X_out, Y_out, loss);
        cudaMemcpy(loss_host, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < batch_size; i++){
          totalLoss += loss_host[i];
        }

        // CAN DETERMINE HOW MANY LABELS WERE CORRECT...
        makePredict <<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>>(batch_size, output_len, X_out, predicted);
        // get network predictions back on host...
        cudaMemcpy(predicted_host, predicted, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

        int correct_label;
        int pred_label;
        
        for (int i = 0; i < batch_size; i++){
        	correct_label = test_labels_raw[batch_i * batch_size + i];
        	pred_label = predicted_host[i];
        	if (correct_label != pred_label){
        		n_wrong++;
        		is_wrong_host[batch_i * batch_size + i] = true;
        	}
        }


        printf("\n\nX OUT MATRIX:\n\n");
        for (int i = 0; i < output_len * batch_size; i++){
        	if ((i % batch_size) == 0) {
            	printf("\n");
            }
            printf("%f ", X_out_host[i]);
        }


        printf("\n\nY OUT MATRIX:\n\n");
        for (int i = 0; i < output_len * batch_size; i++){
        	if ((i % batch_size) == 0) {
        		printf("\n");
            }
            printf("%f ", Y_out_host[i]);
        }
  	}
  	float error_rate = (float) n_wrong / (float) test_n;
  	
  	printf("Test MSE Avg: %d\n", totalLoss / (float) test_n);
  	printf("Accuracy: %f\n", 1 - error_rate);



  	// FREE MEMORY...

  	// ON GPU

  	cudaFree(X_in);
  	cudaFree(Y_out);

  	cudaFree(X_h1);
  	cudaFree(X_h2);
  	cudaFree(X_h3);
  	cudaFree(X_out);

  	cudaFree(W_h1_full);
  	cudaFree(W_h2_full);
  	cudaFree(W_h3);
  	cudaFree(W_out);

  	cudaFree(B_h1);
  	cudaFree(B_h2);
  	cudaFree(B_h3);
  	cudaFree(B_out);

  	cudaFree(X_in_T);
  	cudaFree(Y_out_T);

  	cudaFree(loss);
  	cudaFree(predicted);


  	// ON CPU

  	free(test_labels_raw);
  	free(test_images);
  	free(test_labels);


  	free(X_in_host);
  	free(X_out_host);
  	free(Y_out_host);

  	free(predicted_host);

  	free(W_h1_host);
  	free(W_h1_full_host);
  	free(W_h2_host);
  	free(W_h2_full_host);
  	free(W_h3_host);
  	free(W_out_host);

  	free(B_h1_host);
  	free(B_h2_host);
  	free(B_h3_host);
  	free(B_out_host);

  	free(loss_host);
  	free(is_wrong_host);



}
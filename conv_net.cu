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



// matrix set zero is same as vector zero, (flatten to 1D)
__global__  void setZero(int size, float *A){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      A[i] = 0;
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

// matrix add is same as vector add, flatten to 1D
__global__  void matAdd(int size, float *A, float *B, float *out){

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size){
      out[i] = A[i] + B[i];
  }
}


__global__  void matSub(int size, float *A, float *B, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      out[i] = A[i] - B[i];
  }
}


__global__  void matScale(int size, float *A, float factor){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      A[i] = A[i] * factor;
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


// for sigmoid: hadamard(gradient, (hadamard(nodes, 1 - nodes)))
// for tanh: 1 - f(x) ^ 2
__global__  void activationDeriv(int size, float *gradient, float *nodes, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      float TANH_FREQ = 2.0 / 3.0;
      float TANH_AMP = 1.7159;
      out[i] = gradient[i] * (1 - (nodes[i] * nodes[i])) * TANH_FREQ * TANH_AMP;
  }
}



// size is size of output (# of bias derivs); width is width of input matrix (batch size)
__global__  void biasDerivs(int size, int width, float *node_derivs, float *out){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      float sum = 0;
      for (int j = 0; j < width; j++){
        sum += node_derivs[width * i + j];
      }
      out[i] = sum / width;
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
__global__ void transposeSimp(int size, int width, float * M, float *out){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / width;
  int colInd = i % width;
  int height = size / width;
  if (i < size){
    out[height * colInd + rowInd] = M[width * rowInd + colInd];
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
// size is total number of elements (nRows * nCols), width = nCols
__global__  void addBiasAndActivate(int size, int width, float *X, float *B){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / width;
  int colInd = i % width;
  float z;
  if (i < size){
    X[width * rowInd + colInd] += B[rowInd];
    float TANH_FREQ = 2.0 / 3.0;
    float TANH_AMP = 1.7159;
    z = TANH_FREQ * X[width * rowInd + colInd];
    X[width * rowInd + colInd] = TANH_AMP * (__expf(z) - __expf(-z)) / (__expf(z) + __expf(-z));
  }
}


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


__global__ void fullWeightDerivToUnique(int size, int width, int *map_to_unique_ind, float *M_weight,  float *out){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / width;
  int colInd = i % width;
  if (i < size){
    int unique_ind = map_to_unique_ind[rowInd * width + colInd];
    atomicAdd(&out[unique_ind], M_weight[rowInd * width + colInd]);
  }
}

// size is size of unique weights, unique_width is width of unique (# of elements with shared weight), full_width is width of full matrix
__global__ void uniqueWeightToFull(int size, int unique_width, int full_width, int *map_from_unique, float *unique_weights,  float *out){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size){
    float unique_weight_val = unique_weights[i];
    for (int j = 0; j < unique_width; j++){
      int full_ind = map_from_unique[unique_width * i + j];
      out[full_ind] = unique_weight_val;
    }
  }
}





// // one thread per (row, col) in output matrix
// // tile size is (blockDim.x, blockDim.y) where blockDim.x / blockDim.y = M / N
// // load by tile to utilize shared memory 

// // matrix b is transposed
// __global__ void matrixMatrixMul(int K, float *A, float *B, float *out){
  
//   __shared__ float A_tile[blockDim.x][blockDim.y];
//   __shared__ float B_tile[blockDim.y][blockDim.x];


//   // (rowInd, colInd) is the value in output matrix we are working on in this thread
//   int rowInd = blockDim.x * blockIdx.x + threadIdx.x;
//   int colInd = blockDim.y * blockIdx.y + threadIdx.y;

//   // value of (row, col) in output matrix
//   float p = 0;
//   int n_tiles = ceil(K / float(blockDim.x)); 

//   // loop over each tile (all the phases of progression of block)
//   for (int i = 0; i < n_tiles; i++){
//     A_tile[threadIdx.x][threadIdx.y] = A[K * rowInd + blockDim.y * i + y];
//     B_tile[threadIdx.y][threadIdx.x] = B[K * colInd + blockDim.y * i + x];
//   }
// }


  // going to compute partial dot products by doing dot product of vector chunk and corresponding columns of matrix
  // each block with process multiplications for all rows and N / nVecChunk columns
__global__ 
void matrixVectorMult(int N, float *mat, float *vec, float *out){
  
  // preload chucks of vector into shared memory because these are used repeatedly for multiple rows
  __shared__ float partVect[CHUNK_SIZE];
  int colInd = blockIdx.y * blockDim.y + threadIdx.y;
  // only need to access global memory for one row of block
  if (threadIdx.x == 0){
    partVect[threadIdx.y] = vec[colInd];
  }
  // ensure whole chunk is loaded so other rows in block can use shared memory to access chunk
  __syncthreads();
  
  int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
  if (rowInd < N && colInd < N){
    atomicAdd(&out[rowInd], partVect[threadIdx.y] * mat[rowInd * N + colInd]);
  }
}


void add_conv_mappings(int k, int map_i, int map_j, int inp_N, int conv_N, int map_N, int null_ind, int * map_to_unique, int map_to_unique_width, int * map_from_unique, int map_from_unique_width){
  int input_focus_row = 2 * map_i;
  int input_focus_col = 2 * map_j;
  int mid = conv_N / 2;
  int input_connection_ind, output_connection_ind, weight_ind;
  for (int kern_i = -mid; kern_i <= mid; kern_i++){
    for (int kern_j = -mid; kern_j <= mid; kern_j++){
      if ((input_focus_row + kern_i < 0) || (input_focus_row + kern_i >= inp_N) || (input_focus_col + kern_j < 0) || (input_focus_col + kern_i >= inp_N)){
        input_connection_ind = null_ind;
      }
      else{
        input_connection_ind = k * (inp_N * inp_N) + inp_N * (input_focus_row + kern_i) + (input_focus_col + kern_j);
      }
      output_connection_ind = k * (map_N * map_N) + map_N * map_i + map_j;
      weight_ind = k * (conv_N * conv_N) + conv_N * (kern_i + mid) + (kern_j + mid);
      map_to_unique[map_to_unique_width * output_connection_ind + input_connection_ind] = weight_ind;
      for (int insert_ind = 0; insert_ind < map_from_unique_width; insert_ind++){
        if (map_from_unique[map_from_unique_width * weight_ind + insert_ind] == -1){
          map_from_unique[map_from_unique_width * weight_ind + insert_ind] = map_to_unique_width * output_connection_ind + input_connection_ind;
          break;
        }
      }
    }
  }
}


void repopulate_with_unique(float * full_W, float *unique_W, int unique_W_len, int * map_from_unique, int map_from_unique_width){
  int full_w_ind;
  for (int weight_ind = 0; weight_ind < unique_W_len; weight_ind++){
    for (int j = 0; j < map_from_unique_width; j++){
        full_w_ind = map_from_unique[map_from_unique_width * weight_ind + j];
        full_W[full_w_ind] = unique_W[weight_ind];
    }
  }
}


int main(void)
{


  // DEFINE ARCHITECURAL PARAMTERS FOR NEURAL NET 

  // mini batch size
  int batch_size = 1;
  // how many times to repeat dataset
  int repeat_n = 23;
  float learning_rate_sched[23];
  for (int i = 0; i < 23; i++){
    if (i < 2){
      learning_rate_sched[i] = .0005;
    }
    else if (i < 4){
      learning_rate_sched[i] = .001;
    }
    else if (i < 8){
      learning_rate_sched[i] = .0005;
    }
    else if (i < 16){
      learning_rate_sched[i] = .0001;
    }
    else{
      learning_rate_sched[i] = .00005;
    }
  }

  int input_len = 257;
  int output_len = 10;

  int h1_size = 769;
  int h2_size = 192;
  int h3_size = 30;

  int input_dim = 16;
  int input_null_ind = 256;

  int h1_maps = 12;
  int h1_kernel_dim = 5;
  int h1_map_dim = 8;
  int h1_null_ind = 768;

  int h2_maps = 12;
  int h2_kernel_dim = 5;
  int h2_map_dim = 4;

  int W_h1_size = h1_maps * (h1_kernel_dim * h1_kernel_dim);
  int W_h1_full_size = input_len * h1_size;
  int W_h2_size = h2_maps * (h2_kernel_dim * h2_kernel_dim);
  int W_h2_full_size = h1_size * h2_size;
  int W_h3_size = h2_size * h3_size;
  int W_out_size = h3_size * output_len;


  // input and labels
  float *X_in_host, *Y_out_host;
  X_in_host = (float*)malloc(input_len * batch_size *sizeof(float));
  Y_out_host = (float*)malloc(output_len * batch_size * sizeof(float));

  // for checking values...
  float *X_h1_host, *X_h2_host, *X_h3_host, *X_out_host;
  X_h1_host = (float*)malloc(h1_size * batch_size *sizeof(float));
  X_h2_host = (float*)malloc(h2_size * batch_size *sizeof(float));
  X_h3_host = (float*)malloc(h3_size * batch_size *sizeof(float));
  X_out_host = (float*)malloc(output_len * batch_size *sizeof(float));

  float *X_h3_T_host;
  X_h3_T_host = (float *)malloc(h3_size * batch_size * sizeof(float));

  float *dX_h1_host, *dX_h2_host, *dX_h3_host, *dX_out_host;
  dX_h1_host = (float*)malloc(h1_size * batch_size *sizeof(float));
  dX_h2_host = (float*)malloc(h2_size * batch_size *sizeof(float));
  dX_h3_host = (float*)malloc(h3_size * batch_size *sizeof(float));
  dX_out_host = (float*)malloc(output_len * batch_size *sizeof(float));

  float *dX_h1_activation_host, *dX_h2_activation_host, *dX_h3_activation_host, *dX_out_activation_host;
  dX_h1_activation_host = (float*)malloc(h1_size * batch_size *sizeof(float));
  dX_h2_activation_host = (float*)malloc(h2_size * batch_size *sizeof(float));
  dX_h3_activation_host = (float*)malloc(h3_size * batch_size *sizeof(float));
  dX_out_activation_host = (float*)malloc(output_len * batch_size *sizeof(float));





  // weights
  float *W_h1_host, *W_h2_host, *W_h3_host, *W_out_host;
  W_h1_host = (float *)malloc(W_h1_size * sizeof(float));
  W_h2_host = (float *)malloc(W_h2_size * sizeof(float));
  W_h3_host = (float *)malloc(W_h3_size * sizeof(float));
  W_out_host = (float *)malloc(W_out_size * sizeof(float));


  // for checking derivs...
  float *dW_h1_host, *dW_h2_host, *dW_h3_host, *dW_out_host;
  dW_h1_host = (float *)malloc(W_h1_size * sizeof(float));
  dW_h2_host = (float *)malloc(W_h2_size * sizeof(float));
  dW_h3_host = (float *)malloc(W_h3_size * sizeof(float));
  dW_out_host = (float *)malloc(W_out_size * sizeof(float));

  float *W_h1_full_host, *W_h2_full_host;
  W_h1_full_host = (float *)malloc(W_h1_full_size * sizeof(float));
  W_h2_full_host = (float *)malloc(W_h2_full_size * sizeof(float));

  // biases
  float *B_h1_host, *B_h2_host, *B_h3_host, *B_out_host;
  B_h1_host = (float *)malloc(h1_size * sizeof(float));
  B_h2_host = (float *)malloc(h2_size * sizeof(float));
  B_h3_host = (float *)malloc(h3_size * sizeof(float));
  B_out_host = (float *)malloc(output_len * sizeof(float));

  // loss (storing loss values per batch)
  float *loss_host;
  loss_host = (float *)malloc(batch_size * sizeof(float));

  // predicted values
  float *predicted_host;
  predicted_host = (float*)malloc(batch_size *sizeof(float));


  // create mappings between indicies of duplicated weight matrix (full) and indicies unique weight vector 

  // array of h1_size rows and input_len columns
  int * h1_full_to_unique_host = (int *)malloc(input_len * h1_size * sizeof(int));
  // array of kernel_maps * (kern_dim ** 2) rows with map_dim ** 2 columns 
  int * h1_unique_to_full_host = (int *)malloc((h1_maps * (h1_kernel_dim * h1_kernel_dim)) * (h1_map_dim * h1_map_dim) * sizeof(int));
  // initalize unique -> full with -1 to know insertion location for reverse mapping...
  for (int i = 0; i < h1_maps * (h1_kernel_dim * h1_kernel_dim) * (h1_map_dim * h1_map_dim); i++){
    h1_unique_to_full_host[i] = -1;
  }

  // array of h2_size rows and h1_size columns
  int * h2_full_to_unique_host = (int *)malloc(h1_size * h2_size * sizeof(int));
  int * h2_unique_to_full_host = (int *)malloc((h2_maps * (h2_kernel_dim * h2_kernel_dim)) * (h2_map_dim * h2_map_dim) * sizeof(int));
  for (int i = 0; i < h2_maps * (h2_kernel_dim * h2_kernel_dim) * (h2_map_dim * h2_map_dim); i++){
    h2_unique_to_full_host[i] = -1;
  }

 
  // input to h1 mappings
  for (int k = 0; k < h1_maps; k++){
    for (int map_i = 0; map_i < h1_kernel_dim; map_i++){
      for (int map_j = 0; map_j < h1_kernel_dim; map_j++){
        add_conv_mappings(k, map_i, map_j, input_dim, h1_kernel_dim, h1_map_dim, input_null_ind, h1_full_to_unique_host, input_len, h1_unique_to_full_host, h1_map_dim * h1_map_dim);
      }
    }
  }

  // h1 to h2 mappings
  for (int k = 0; k < h2_maps; k++){
    for (int map_i = 0; map_i < h2_kernel_dim; map_i++){
      for (int map_j = 0; map_j < h2_kernel_dim; map_j++){
        add_conv_mappings(k, map_i, map_j, h1_map_dim, h2_kernel_dim, h2_map_dim, h1_null_ind, h2_full_to_unique_host, h1_size, h2_unique_to_full_host, h2_map_dim * h2_map_dim);
      }
    }
  }



  // initalize weights and biases

  // uniform random weights between +/ (24 / (# inputs to unit which connection belongs))
  float conv_init_bound = 2.4 / 25;
  float conv2_init_bound = 2.4 / 25;
  float h3_init_bound = 2.4 / 192;
  float out_init_bound = 2.4 / 30;
  for (int i = 0; i < W_h1_size; i++){
    // 50/50 for sign, then [0, 24/25]
    W_h1_host[i] = (2 * (rand() % 2) - 1) * (float)rand()/(float)(RAND_MAX/conv_init_bound);
  }
  for (int i = 0; i < W_h2_size; i++){
    W_h2_host[i] = (2 * (rand() % 2) - 1) * (float)rand()/(float)(RAND_MAX/conv2_init_bound);
  }
  for (int i = 0; i < W_h3_size; i++){
    W_h3_host[i] = (2 * (rand() % 2) - 1) * (float)rand()/(float)(RAND_MAX/h3_init_bound);
  }
  for (int i = 0; i < W_out_size; i++){
    W_out_host[i] = (2 * (rand() % 2) - 1) * (float)rand()/(float)(RAND_MAX/out_init_bound);
  }

  for (int i = 0; i < W_h1_full_size; i++){
    W_h1_full_host[i] = 0;
  }
  for (int i = 0; i < W_h2_full_size; i++){
    W_h2_full_host[i] = 0;
  }

  // fill up the Weight matrix from unique weights
  repopulate_with_unique(W_h1_full_host, W_h1_host, W_h1_size, h1_unique_to_full_host, h1_map_dim * h1_map_dim);
  repopulate_with_unique(W_h2_full_host, W_h2_host, W_h2_size, h2_unique_to_full_host, h2_map_dim * h2_map_dim);


  // init biases to be 0
  for (int i = 0; i < h1_size; i++){
    B_h1_host[i] = 0;
  }
  for (int i = 0; i < h2_size; i++){
    B_h2_host[i] = 0;
  }
  for (int i= 0; i < h3_size; i++){
    B_h3_host[i] = 0;
  }
  for (int i=0; i < output_len; i++){
    B_out_host[i] = 0;
  }

  // READ FROM DATASET!!!!
  FILE * training_images_file, *training_labels_file;

  unsigned char * training_images_raw, *training_labels_raw;

  float *training_images, *training_labels;

  const char * training_images_path = "/mnt/storage/data/image_text/mnist/train-images.idx3-ubyte";
  const char * training_labels_path = "/mnt/storage/data/image_text/mnist/train-labels.idx1-ubyte";

  training_images_file = fopen(training_images_path, "r");
  training_labels_file = fopen(training_labels_path, "r");


  // from "http://yann.lecun.com/exdb/mnist/"
  off_t training_images_offset = 16;
  off_t training_labels_offset = 8;

  // skipping offset bytes in beginning then measuring til end = skipping offset bytes in end and measuring from start
  fseek(training_images_file, training_images_offset, SEEK_END);
  long training_images_nbytes = ftell(training_images_file);
  fseek(training_labels_file, training_labels_offset, SEEK_END);
  long training_labels_nbytes = ftell(training_labels_file);

  // raw because going to downsample..
  training_images_raw = (unsigned char *) calloc(training_images_nbytes, sizeof(unsigned char));
  training_labels_raw = (unsigned char *) calloc(training_labels_nbytes, sizeof(unsigned char));

  // set to beginning...
  fseek(training_images_file, training_images_offset, SEEK_SET);
  fseek(training_labels_file, training_labels_offset, SEEK_SET);

  fread(training_images_raw, sizeof(unsigned char), training_images_nbytes, training_images_file);
  fclose(training_images_file);

  fread(training_labels_raw, sizeof(unsigned char), training_labels_nbytes, training_labels_file);
  fclose(training_labels_file);


  int training_n = 60000;
  int image_raw_dim = 28;
  int image_dim = 16;
  float ratio = float(image_raw_dim) / float(image_dim);

  int raw_floor_row, raw_floor_col, top_left, pixel_ind;
  float ave_raw_pixel, pixel_val;

  // store images as array of 16*16 images, but then additional -1 input
  training_images = (float *) calloc(training_n * input_len, sizeof(float));


  for (int img_num = 0; img_num < training_n; img_num++){
    //printf("Img Num: %i\n\n", img_num);
    for (int i = 0; i < image_dim; i++){
      for (int j = 0; j < image_dim; j++){
        // averaging 4 closest pixels in original image
        raw_floor_row = floor(i * ratio);
        raw_floor_col = floor(j * ratio);
        top_left = img_num * (image_raw_dim * image_raw_dim) + image_raw_dim * raw_floor_row + raw_floor_col;
        ave_raw_pixel = (float)(((float)training_images_raw[top_left] + (float)training_images_raw[top_left + 1] + (float)training_images_raw[top_left + image_raw_dim] + (float)training_images_raw[top_left + image_raw_dim + 1]) / float(4));
        // scale to be between -1 and 1
        pixel_val = ave_raw_pixel * (2.0 / 255.0) - 1;
        // storing average pixel value into downsampled array
        pixel_ind = img_num * input_len + image_dim * i + j;
        training_images[pixel_ind] = pixel_val;
      }
    }
    // have last value in input image be a -1
    training_images[img_num * input_len + (image_dim * image_dim)] = -1;
  }
  // stored in downsampled training_images so can free now
  free(training_images_raw);

  // store image labels as series of 10 floats
  training_labels = (float *) calloc(training_n * output_len, sizeof(float));

  int label;
  for (int img_num = 0; img_num < training_n; img_num++){
    label = training_labels_raw[img_num];
    for (int dig = 0; dig < output_len; dig++){
      if (label == dig){
        training_labels[img_num * output_len + dig] = 1.0;
      }
      else { 
        training_labels[img_num * output_len + dig] = 0.0;
      }
    }
  }

  


  // GPU variables

  // initalize a timer if want to use
  cudaEvent_t gpu_start, gpu_stop;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);


  // hidden nodes
  float *X_in, *X_h1, *X_h2, *X_h3, *X_out, *Y_out;
  float *X_in_T, *X_h1_T, *X_h2_T, *X_h3_T, *Y_out_T;
  float *dX_h1, *dX_h2, *dX_h3, *dX_out;
  float *dX_h1_activation, *dX_h2_activation, *dX_h3_activation, *dX_out_activation;

  // weights
  float *W_h1, *W_h1_full, *W_h2, *W_h2_full, *W_h3, *W_out;
  float *W_h1_full_T, *W_h2_full_T, *W_h3_T, *W_out_T;
  float *dW_h1, *dW_h1_full, *dW_h2, *dW_h2_full, *dW_h3, *dW_out;

  // biases
  float *B_h1, *B_h2, *B_h3, *B_out;
  float *dB_h1, *dB_h2, *dB_h3, *dB_out;


  // weight mappings
  int *h1_full_to_unique, *h1_unique_to_full; 
  int *h2_full_to_unique, *h2_unique_to_full;

  // loss per sample
  float *loss;

  // predicted values
  float *predicted;

  // ALLOCATE GPU MEMORY

  // allocate space for hidden nodes on gpu
  cudaMalloc(&X_in, input_len * batch_size * sizeof(float)); 
  cudaMalloc(&X_h1, h1_size * batch_size*sizeof(float));
  cudaMalloc(&X_h2, h2_size * batch_size*sizeof(float));
  cudaMalloc(&X_h3, h3_size * batch_size*sizeof(float));
  cudaMalloc(&X_out, output_len * batch_size*sizeof(float));
  cudaMalloc(&Y_out, output_len * batch_size* sizeof(float));


  // allocate space for hidden node transposed (used in intermediate computations)
  cudaMalloc(&X_in_T, input_len * batch_size * sizeof(float)); 
  cudaMalloc(&X_h1_T, h1_size * batch_size*sizeof(float));
  cudaMalloc(&X_h2_T, h2_size * batch_size*sizeof(float));
  cudaMalloc(&X_h3_T, h3_size * batch_size*sizeof(float));
  cudaMalloc(&Y_out_T, output_len * batch_size* sizeof(float));

  // allocate space for node gradients
  cudaMalloc(&dX_h1, h1_size * batch_size*sizeof(float));
  cudaMalloc(&dX_h2, h2_size * batch_size*sizeof(float));
  cudaMalloc(&dX_h3, h3_size * batch_size*sizeof(float));
  cudaMalloc(&dX_out, output_len * batch_size*sizeof(float));


  // allocate space for activation gradients (used in intermediate computations)
  cudaMalloc(&dX_h1_activation, h1_size * batch_size*sizeof(float));
  cudaMalloc(&dX_h2_activation, h2_size * batch_size*sizeof(float));
  cudaMalloc(&dX_h3_activation, h3_size * batch_size*sizeof(float));
  cudaMalloc(&dX_out_activation, output_len * batch_size*sizeof(float));


  // alllocate space for weights
  cudaMalloc(&W_h1, W_h1_size * sizeof(float));
  cudaMalloc(&W_h1_full, W_h1_full_size * sizeof(float));
  cudaMalloc(&W_h2, W_h2_size * sizeof(float));
  cudaMalloc(&W_h2_full, W_h2_full_size * sizeof(float));
  cudaMalloc(&W_h3, W_h3_size * sizeof(float));
  cudaMalloc(&W_out, W_out_size * sizeof(float));

  // alllocate space for weight_transpose (used in intermediate computations)
  cudaMalloc(&W_h1_full_T, W_h1_full_size * sizeof(float));
  cudaMalloc(&W_h2_full_T, W_h2_full_size * sizeof(float));
  cudaMalloc(&W_h3_T, W_h3_size * sizeof(float));
  cudaMalloc(&W_out_T, W_out_size * sizeof(float));

  // allocate space for weight mappings...

  cudaMalloc(&h1_full_to_unique, input_len * h1_size * sizeof(int));
  cudaMalloc(&h1_unique_to_full, (h1_maps * (h1_kernel_dim * h1_kernel_dim)) * (h1_map_dim * h1_map_dim) * sizeof(int));

  cudaMalloc(&h2_full_to_unique, h1_size * h2_size * sizeof(int));
  cudaMalloc(&h2_unique_to_full, (h2_maps * (h2_kernel_dim * h2_kernel_dim)) * (h2_map_dim * h2_map_dim) * sizeof(int));


  // allocate space for weight gradients
  cudaMalloc(&dW_h1, W_h1_size * sizeof(float));
  cudaMalloc(&dW_h1_full, W_h1_full_size * sizeof(float));
  cudaMalloc(&dW_h2, W_h2_size * sizeof(float));
  cudaMalloc(&dW_h2_full, W_h2_full_size * sizeof(float));
  cudaMalloc(&dW_h3, W_h3_size * sizeof(float));
  cudaMalloc(&dW_out, W_out_size * sizeof(float));

  // allocate space for biases
  cudaMalloc(&B_h1, h1_size * sizeof(float));
  cudaMalloc(&B_h2, h2_size * sizeof(float));
  cudaMalloc(&B_h3, h3_size * sizeof(float));
  cudaMalloc(&B_out, output_len * sizeof(float));

  // allocate space for bias gradients
  cudaMalloc(&dB_h1, h1_size * sizeof(float));
  cudaMalloc(&dB_h2, h2_size * sizeof(float));
  cudaMalloc(&dB_h3, h3_size * sizeof(float));
  cudaMalloc(&dB_out, output_len * sizeof(float));

  // allocate space to store values for loss function per sample
  cudaMalloc(&loss, batch_size * sizeof(float));

  cudaMalloc(&predicted, batch_size * sizeof(float));


  // COPY VALUES FROM CPU

  // initalized weights and biases
  cudaMemcpy(W_h1, W_h1_host, W_h1_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(W_h2, W_h2_host, W_h2_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(W_h3, W_h3_host, W_h3_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(W_out, W_out_host, W_out_size * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(W_h1_full, W_h1_full_host, W_h1_full_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(W_h2_full, W_h2_full_host, W_h2_full_size * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(B_h1, B_h1_host, h1_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_h2, B_h2_host, h2_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_h3, B_h3_host, h3_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_out, B_out_host, output_len * sizeof(float), cudaMemcpyHostToDevice);

  // conv weight mappings
  cudaMemcpy(h1_full_to_unique, h1_full_to_unique_host, input_len * h1_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(h1_unique_to_full, h1_unique_to_full_host, (h1_maps * (h1_kernel_dim * h1_kernel_dim)) * (h1_map_dim * h1_map_dim) * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(h2_full_to_unique, h2_full_to_unique_host, h1_size * h2_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(h2_unique_to_full, h2_unique_to_full_host, (h2_maps * (h2_kernel_dim * h2_kernel_dim)) * (h2_map_dim * h2_map_dim) * sizeof(int), cudaMemcpyHostToDevice);


  /* copied from original simple example....

  // nVectChunkSize vs. nColsPerBlock is tradeoff between minimize access to global memory. Larger chunks => blocks share more memory, but smaller chunks => larger colsPerBlock => taking advantage of shared memory
  // keep nVectChunkSize * nColsPerBlock = 1024 to maximize threads per blocks (capped at 1024)
  // chunk size is defined in CHUNK_SIZE
  int nVectChunkSize = 64;
  // 1024 is max number of threads per block
  int nColsPerBlock = 1024 / nVectChunkSize;

  // each block contains (32 x 32) threads
  // there are (32 x 32) blocks
  
  dim3 threadsPerBlock (nColsPerBlock, nVectChunkSize);
  dim3 numBlocks (ceil(N / threadsPerBlock.x), ceil(N / threadsPerBlock.y));

  */


  cudaEventRecord(gpu_start);

  // TRAINNNNNN

  int n_batches = training_n / batch_size;
  for (int cnt = 0; cnt < repeat_n; cnt++){
    float totalLoss = 0;
    float n_wrong = 0;

    printf("\nDataset Iteration: %i\n\n", cnt);
    for (int batch_i = 0; batch_i < n_batches; batch_i++){

        if (batch_i % 1000 == 0){
          printf("Batch #: %d\n", batch_i);
        }
        // get new batch
        memcpy(X_in_host, training_images + batch_i * batch_size * input_len, batch_size * input_len * sizeof(float));
        memcpy(Y_out_host, training_labels + batch_i * batch_size * output_len, batch_size * output_len * sizeof(float));
        // read in as consective images (so pixels are rows). want to transpose, then send back to host
        cudaMemcpy(X_in_T, X_in_host, input_len *batch_size* sizeof(float), cudaMemcpyHostToDevice);
        transposeSimp<<< SM_COUNT, ceil((float)input_len * batch_size / SM_COUNT)>>> (input_len * batch_size, input_len, X_in_T, X_in);
        cudaMemcpy(X_in_host, X_in, input_len *batch_size* sizeof(float), cudaMemcpyDeviceToHost);
        // read in as consective sequences of output lables, want to transpose
        cudaMemcpy(Y_out_T, Y_out_host, output_len*batch_size*sizeof(float), cudaMemcpyHostToDevice);
        transposeSimp<<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>> (output_len * batch_size, output_len, Y_out_T, Y_out);
        cudaMemcpy(Y_out_host, Y_out, output_len *batch_size* sizeof(float), cudaMemcpyDeviceToHost);

        
        // FORWARD PASS

        /// (769, 257) x (257, 2000)
        cudaMemcpy(W_h1_host, W_h1, W_h1_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(W_h1_full_host, W_h1_full, W_h1_full_size * sizeof(float), cudaMemcpyDeviceToHost);

        matMulSimp<<< SM_COUNT, ceil((float)h1_size * batch_size / SM_COUNT) >>>(h1_size, input_len, batch_size, W_h1_full, X_in, X_h1);
        addBiasAndActivate <<< SM_COUNT, ceil((float)h1_size * batch_size / SM_COUNT) >>>(h1_size * batch_size, batch_size, X_h1, B_h1);
        // set constant of -1 to last val in h1 for all samples in batch
        setRowVal<<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, h1_size - 1, X_h1, -1.0);

        cudaMemcpy(X_h1_host, X_h1, h1_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

          // h1 to h2

        cudaMemcpy(W_h2_host, W_h2, W_h2_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(W_h2_full_host, W_h2_full, W_h2_full_size * sizeof(float), cudaMemcpyDeviceToHost);

        matMulSimp<<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT) >>>(h2_size, h1_size, batch_size, W_h2_full, X_h1, X_h2);
        addBiasAndActivate <<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT) >>>(h2_size * batch_size, batch_size, X_h2, B_h2);

        cudaMemcpy(X_h2_host, X_h2, h2_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

          // h2 to h3

        cudaMemcpy(W_h3_host, W_h3, W_h3_size * sizeof(float), cudaMemcpyDeviceToHost);

        matMulSimp<<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT) >>>(h3_size, h2_size, batch_size, W_h3, X_h2, X_h3);
        addBiasAndActivate <<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT) >>>(h3_size * batch_size, batch_size, X_h3, B_h3);

        cudaMemcpy(X_h3_host, X_h3, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

          // h3 to output

        cudaMemcpy(W_out_host, W_out, W_out_size * sizeof(float), cudaMemcpyDeviceToHost);


        matMulSimp<<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len, h3_size, batch_size, W_out, X_h3, X_out);
        
        addBias <<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>> (output_len * batch_size, batch_size, X_out, B_out);
        softMax <<< SM_COUNT, ceil((float)batch_size / SM_COUNT) >>> (batch_size, output_len, X_out);

        cudaMemcpy(X_out_host, X_out, output_len * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

        // COMPUTE LOSS 
        // average loss per sample in batch...
        // not optimized...
        computeLoss <<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, output_len, X_out, Y_out, loss);
        cudaMemcpy(loss_host, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < batch_size; i++){
          totalLoss += loss_host[i];
        }

        makePredict <<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, output_len, X_out, predicted);

        cudaMemcpy(predicted_host, predicted, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < batch_size; i++){
          if (predicted_host[i] != training_labels_raw[batch_i * batch_size + i]){
            n_wrong++;
          }
        }
        
        
        // BACK PROP => want dW_out, dW_h3, dW_h2, dW_h1 + dB_out, dB_h3, dB_h2, dB_h1

          // compute dX_out
        // already accounting for flipped sign (X_out - Y_out => want to add gradients because loss deriv multiplied by -1)
        matSub<<<SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len*batch_size, X_out, Y_out, dX_out);
        matScale<<<SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len*batch_size, dX_out, 2);
        cudaMemcpy(dX_out_host, dX_out, output_len * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

          // average rows of dX_out to get dB_out (equal to matMul of dX_out by (batch size X 1) matrix of 1's and scale by (1 / batch size))
          // bad grid dimensions, can fix later...
          // biases dB_out
        biasDerivs <<< SM_COUNT, ceil((float)output_len / SM_COUNT)>>> (output_len, batch_size, dX_out, dB_out);

          // prep for next layer back ...
        activationDeriv<<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len * batch_size, dX_out, X_out, dX_out_activation);
        cudaMemcpy(dX_out_activation_host, dX_out_activation, output_len * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

          // compute dW_out
        transposeSimp<<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT)>>>(h3_size * batch_size, batch_size, X_h3, X_h3_T);
        cudaMemcpy(X_h3_T_host, X_h3_T, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
          // VERY STUPID GRID/BLOCK DIMENSIONS (should fix when the matrix multiply kernel switches...)
        matMulSimp<<< SM_COUNT, ceil((float)output_len * h3_size / SM_COUNT) >>>(output_len, batch_size, h3_size, dX_out_activation, X_h3_T, dW_out);

        cudaMemcpy(dW_out_host, dW_out, W_out_size * sizeof(float), cudaMemcpyDeviceToHost);

          // compute dX_h3
        transposeSimp<<< SM_COUNT, ceil((float)W_out_size / SM_COUNT)>>>(W_out_size, h3_size, W_out, W_out_T);
        matMulSimp<<<SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT)>>>(h3_size, output_len, batch_size, W_out_T, dX_out_activation, dX_h3);
        cudaMemcpy(dX_h3_host, dX_h3, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);



          // biases dB_h3
        biasDerivs <<< SM_COUNT, ceil((float)h3_size / SM_COUNT)>>> (h3_size, batch_size, dX_h3, dB_h3);

          // prep for next layer back...
        activationDeriv<<< SM_COUNT, ceil((float)h3_size * batch_size/ SM_COUNT)>>>(h3_size * batch_size, dX_h3, X_h3, dX_h3_activation);
        cudaMemcpy(dX_h3_activation_host, dX_h3_activation, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

          // compute dW_h3
        transposeSimp<<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT)>>>(h2_size * batch_size, batch_size, X_h2, X_h2_T);
          // VERY STUPID GRID/BLOCK DIMENSIONS (should fix when the matrix multiply kernel switches...)
        matMulSimp<<< SM_COUNT, ceil((float)h3_size * h2_size / SM_COUNT) >>>(h3_size, batch_size, h2_size, dX_h3_activation, X_h2_T, dW_h3);

        cudaMemcpy(dW_h3_host, dW_h3, W_h3_size * sizeof(float), cudaMemcpyDeviceToHost);

          // compute dX_h2
        transposeSimp<<< SM_COUNT, ceil((float)W_h3_size / SM_COUNT)>>>(W_h3_size, h2_size, W_h3, W_h3_T);
        matMulSimp<<<SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT)>>>(h2_size, h3_size, batch_size, W_h3_T, dX_h3_activation, dX_h2);
        cudaMemcpy(dX_h2_host, dX_h2, h2_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

          // biases dB_h2
        biasDerivs <<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>> (h2_size, batch_size, dX_h2, dB_h2);

          // prep for next layer back...
        activationDeriv<<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT)>>>(h2_size *batch_size, dX_h2, X_h2, dX_h2_activation);
        cudaMemcpy(dX_h2_activation_host, dX_h2_activation, h2_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);


          // NOW AT CONVOLUTION LAYER, so need to aggregate shared weights...

          // first compute derivs of full matrix, then condense to how unique vectors change. then add back to full matrix

          // compute W_h2
        transposeSimp<<< SM_COUNT, ceil((float)h1_size * batch_size / SM_COUNT)>>>(h1_size * batch_size, batch_size, X_h1, X_h1_T);
          // VERY STUPID GRID/BLOCK DIMENSIONS (should fix when the matrix multiply kernel switches...)
        matMulSimp<<< SM_COUNT, ceil((float)h2_size * h1_size / SM_COUNT) >>>(h2_size, batch_size, h1_size, dX_h2_activation, X_h1_T, dW_h2_full);

          // go from full matrix derivs to unique weight derivs by summing over group of shared of weights corresponding to unique weight
        fullWeightDerivToUnique<<< SM_COUNT, ceil((float)h2_size * h1_size / SM_COUNT)>>>(W_h2_full_size, h1_size, h2_full_to_unique, dW_h2_full,  dW_h2);
          // go from unique back to full 
        uniqueWeightToFull<<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT)>>>(W_h2_size, h2_map_dim * h2_map_dim, h1_size, h2_unique_to_full, dW_h2, dW_h2_full);

        cudaMemcpy(dW_h2_host, dW_h2, W_h2_size * sizeof(float), cudaMemcpyDeviceToHost);


          // compute dX_h1
        transposeSimp<<< SM_COUNT, ceil((float)W_h2_full_size / SM_COUNT)>>>(W_h2_full_size, h1_size, W_h2_full, W_h2_full_T);
        matMulSimp<<<SM_COUNT, ceil((float)h1_size * batch_size / SM_COUNT)>>>(h1_size, h2_size, batch_size, W_h2_full_T, dX_h2_activation, dX_h1);
        cudaMemcpy(dX_h1_host, dX_h1, h1_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

          // bises dB_h1
        biasDerivs <<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>> (h1_size, batch_size, dX_h1, dB_h1);

          // prep for next layer back...
        activationDeriv<<< SM_COUNT, ceil((float)h1_size * batch_size / SM_COUNT)>>>(h1_size *batch_size, dX_h1, X_h1, dX_h1_activation);
        cudaMemcpy(dX_h1_activation_host, dX_h1_activation, h1_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);


          // ANOTHER CONVOLUATION LAYER BETWEEN H1 and H2

          // compute W_h1
        transposeSimp<<< SM_COUNT, ceil((float)input_len * batch_size / SM_COUNT)>>>(input_len * batch_size, batch_size, X_in, X_in_T);
          // VERY STUPID GRID/BLOCK DIMENSIONS (should fix when the matrix multiply kernel switches...)
        matMulSimp<<< SM_COUNT, ceil((float)h1_size * input_len / SM_COUNT) >>>(h1_size, batch_size, input_len, dX_h1_activation, X_in_T, dW_h1_full);

          // go from full matrix derivs to unique weight derivs by summing over group of shared of weights corresponding to unique weight
        fullWeightDerivToUnique<<< SM_COUNT, ceil((float)W_h1_full_size / SM_COUNT)>>>(W_h1_full_size, input_len, h1_full_to_unique, dW_h1_full, dW_h1);
          // go from unique back to full 
        uniqueWeightToFull<<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT)>>>(W_h1_size, h1_map_dim * h1_map_dim, input_len, h1_unique_to_full, dW_h1, dW_h1_full);

        cudaMemcpy(dW_h1_host, dW_h1, W_h1_size * sizeof(float), cudaMemcpyDeviceToHost);



          /// DEBUG STUFF....

        if (batch_i == n_batches - 1){
          printf("\nBatch #: %d\n", batch_i);

          printf("\n\nX_in MATRIX:\n\n");
          for (int i = 0; i < input_len * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", X_in_host[i]);
          }

          printf("\n\nW_h1 values:\n\n");
          for (int i = 0; i < W_h1_size; i++){
            printf("%f\n", W_h1_host[i]);
          }

          // printf("\n\nW_h1_full values:\n\n");
          // for (int i = 0; i < W_h1_full_size; i++){
          //   if ((i % input_len == 0)){
          //     printf("\n");
          //   }
          //   printf("%f ", W_h1_full_host[i]);
          // }


          printf("\n\nX_h1 MATRIX:\n\n");
          for (int i = 0; i < h1_size * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", X_h1_host[i]);
          }

          printf("\n\nW_h2 values:\n\n");
          for (int i = 0; i < W_h2_size; i++){
            printf("%f\n", W_h2_host[i]);
          }

          printf("\n\nX_h2 MATRIX:\n\n");
          for (int i = 0; i < h2_size * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", X_h2_host[i]);
          }

          printf("\n\nW_h3 MATRIX:\n\n");
          for (int i = 0; i < W_h3_size; i++){
            if ((i % h2_size) == 0) {
              printf("\n");
            }
            printf("%f ", W_h3_host[i]);
          }
          
          printf("\n\nX_h3 MATRIX:\n\n");
          for (int i = 0; i < h3_size * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", X_h3_host[i]);
          }

          printf("\n\nW_out MATRIX:\n\n");
          for (int i = 0; i < W_out_size; i++){
            if ((i % h3_size) == 0) {
              printf("\n");
            }
            printf("%f ", W_out_host[i]);
          }

          printf("\n\n Iteration #: %d\n\n", cnt);
          printf("\n\nX OUT MATRIX:\n\n");
          for (int i = 0; i < output_len * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n\n");
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

          printf("\n\ndX OUT MATRIX:\n\n");
          for (int i = 0; i < output_len * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", dX_out_host[i]);
          }

          printf("\n\ndX OUT ACTIVATION MATRIX:\n\n");
          for (int i = 0; i < output_len * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", dX_out_activation_host[i]);
          }


          printf("\n\ndX_H3 MATRIX:\n\n");
          for (int i = 0; i < h3_size * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", dX_h3_host[i]);
          }

          printf("\n\ndX_H3 ACTIVATION MATRIX:\n\n");
          for (int i = 0; i < h3_size * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", dX_h3_activation_host[i]);
          }

          printf("\n\nGRADIENT LAST WEIGHT MATRIX:\n\n");
          for (int i = 0; i < W_out_size; i++){
            if ((i % h3_size) == 0) {
              printf("\n");
            }
            printf("%f ", dW_out_host[i]);
          }


          printf("\n\n\n");

        }







          // UPDATE WEIGHTS + BIASES (apply learning rate and add gradients to existing params)
        float learning_rate = learning_rate_sched[cnt];
          // apply learning rate to gradients, and reverse direction
        matScale <<< SM_COUNT, ceil((float)W_out_size / SM_COUNT)>>>(W_out_size, dW_out, -learning_rate);
        matScale <<< SM_COUNT, ceil((float)W_h3_size / SM_COUNT)>>>(W_h3_size, dW_h3, -learning_rate);
        matScale <<< SM_COUNT, ceil((float)W_h2_full_size / SM_COUNT)>>>(W_h2_full_size, dW_h2_full, -learning_rate);
        matScale <<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT)>>>(W_h2_size, dW_h2, -learning_rate);
        matScale <<< SM_COUNT, ceil((float)W_h1_full_size / SM_COUNT)>>>(W_h1_full_size, dW_h1_full, -learning_rate);
        matScale <<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT)>>>(W_h1_size, dW_h1, -learning_rate);

        matScale <<< SM_COUNT, ceil((float)output_len / SM_COUNT)>>>(output_len, dB_out, -learning_rate);
        matScale <<< SM_COUNT, ceil((float)h3_size / SM_COUNT)>>>(h3_size, dB_h3, -learning_rate);
        matScale <<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>>(h2_size, dB_h2, -learning_rate);
        matScale <<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>>(h1_size, dB_h1, -learning_rate);


          // add to previous parameters
        matAdd <<< SM_COUNT, ceil((float)W_out_size / SM_COUNT) >>>(W_out_size, W_out, dW_out, W_out);
        matAdd <<< SM_COUNT, ceil((float)W_h3_size / SM_COUNT) >>>(W_h3_size, W_h3, dW_h3, W_h3);
        matAdd <<< SM_COUNT, ceil((float)W_h2_full_size / SM_COUNT) >>>(W_h2_full_size, W_h2_full, dW_h2_full, W_h2_full);
        matAdd <<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT) >>>(W_h2_size, W_h2, dW_h2, W_h2);
        matAdd <<< SM_COUNT, ceil((float)W_h1_full_size / SM_COUNT) >>>(W_h1_full_size, W_h1_full, dW_h1_full, W_h1_full);
        matAdd <<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT) >>>(W_h1_size, W_h1, dW_h1, W_h1);

        matAdd <<< SM_COUNT, ceil((float)output_len / SM_COUNT)>>>(output_len, B_out, dB_out, B_out);
        matAdd <<< SM_COUNT, ceil((float)h3_size / SM_COUNT)>>>(h3_size, B_h3, dB_h3, B_h3);
        matAdd <<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>>(h2_size, B_h2, dB_h2, B_h2);
        matAdd <<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>>(h1_size, B_h1, dB_h1, B_h1);


        // RESET INTERMEDIATE MEMORY TO ZEROs (already do this within matMul, but need to reset unique Weight derivs for conv layers)
        setZero <<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT) >>> (W_h2_size, dW_h2);
        setZero <<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT) >>> (W_h1_size, dW_h1);

    }
    printf("Avg Loss: %f\n", (float) totalLoss / training_n);
    printf("Accuracy: %f\n", (float) 1 - (n_wrong / training_n));
  }

  cudaEventRecord(gpu_stop);

  // output weights and biases to cpu


  // weights

  cudaMemcpy(W_h1_host, W_h1, W_h1_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(W_h1_full_host, W_h1_full, W_h1_full_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(W_h2_host, W_h2, W_h2_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(W_h2_full_host, W_h2_full, W_h2_full_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(W_h3_host, W_h3, W_h3_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(W_out_host, W_out, W_out_size * sizeof(float), cudaMemcpyDeviceToHost);

  // biases

  cudaMemcpy(B_h1_host, B_h1, h1_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(B_h2_host, B_h2, h2_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(B_h3_host, B_h3, h3_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(B_out_host, B_out, output_len * sizeof(float), cudaMemcpyDeviceToHost);



  cudaEventSynchronize(gpu_stop);
  float gpu_millis = 0;
  cudaEventElapsedTime(&gpu_millis, gpu_start, gpu_stop);
  printf("GPU Elapsed Millis: %f\n", gpu_millis);



  // SAVE MODEL!


  // write output to files here

  const char * model_path = "/mnt/storage/data/image_text/mnist/trained_model";

  FILE * model_file = fopen(model_path, "wb+");

  // write out weights
  fwrite(W_h1_host, sizeof(float), (size_t) W_h1_size, model_file);
  fwrite(W_h1_full_host, sizeof(float), (size_t) W_h1_full_size, model_file);  
  fwrite(W_h2_host, sizeof(float), (size_t) W_h2_size, model_file);
  fwrite(W_h2_full_host, sizeof(float), (size_t) W_h2_full_size, model_file);
  fwrite(W_h3_host, sizeof(float), (size_t) W_h3_size, model_file);
  fwrite(W_out_host, sizeof(float), (size_t) W_out_size, model_file);

  // write out biases
  fwrite(B_h1_host, sizeof(float), (size_t) h1_size, model_file);
  fwrite(B_h2_host, sizeof(float), (size_t) h2_size, model_file);
  fwrite(B_h3_host, sizeof(float), (size_t) h3_size, model_file);
  fwrite(B_out_host, sizeof(float), (size_t) output_len, model_file);

  fclose(model_file);





  // CLEANUP MEMORY...

  // FREE GPU Memory

  // free space for nodes
  cudaFree(X_in);
  cudaFree(X_h1);
  cudaFree(X_h2);
  cudaFree(X_h3);
  cudaFree(X_out);
  cudaFree(Y_out);

  // free space for transposes
  cudaFree(X_in_T);
  cudaFree(X_h1_T);
  cudaFree(X_h2_T);
  cudaFree(X_h3_T);

  // free space for node gradients
  cudaFree(dX_h1);
  cudaFree(dX_h2);
  cudaFree(dX_h3);
  cudaFree(dX_out);

  // free space for node gradient_temporary
  cudaFree(dX_h1_activation);
  cudaFree(dX_h2_activation);
  cudaFree(dX_h3_activation);
  cudaFree(dX_out_activation);

  // free space for weights
  cudaFree(W_h1);
  cudaFree(W_h1_full);
  cudaFree(W_h2);
  cudaFree(W_h2_full);
  cudaFree(W_h3);
  cudaFree(W_out);

  // free space for weight transpose
  cudaFree(W_h1_full_T);
  cudaFree(W_h2_full_T);
  cudaFree(W_h3_T);
  cudaFree(W_out_T);


  // free space for weight mappings...

  cudaFree(h1_full_to_unique);
  cudaFree(h1_unique_to_full);

  cudaFree(h2_full_to_unique);
  cudaFree(h2_unique_to_full);

  // free space for weight gradients
  cudaFree(dW_h1);
  cudaFree(dW_h2);
  cudaFree(dW_h3);
  cudaFree(dW_out);

  // free space for biases
  cudaFree(B_h1);
  cudaFree(B_h2);
  cudaFree(B_h3);
  cudaFree(B_out);

  // free space for bias gradients
  cudaFree(dB_h1);
  cudaFree(dB_h2);
  cudaFree(dB_h3);
  cudaFree(dB_out);


  // free space for storing loss values per batch
  cudaFree(loss);
  cudaFree(predicted);
  
  /*
  
  // Test GPU GEMM Kernel speed vs cpu...

  // PREFORM MATRIX-MATRIX on CPU
  clock_t cpu_start, cpu_end;
  float cpu_millis;
  cpu_start = clock();
  cpuMatrixMatrixMult(M, K, N, mat_left, mat_right, out);
  cpu_end = clock();
  cpu_millis = (((float) (cpu_end - cpu_start)) / CLOCKS_PER_SEC) * 1000;
  printf("CPU Elapsed Millis: %f\n", cpu_millis);
  
  */


  free(training_labels_raw);

  free(training_images);
  free(training_labels);

  free(X_in_host);
  free(Y_out_host);

  // intermediate checking values
  free(X_h1_host);
  free(X_h2_host);
  free(X_h3_host);
  free(X_out_host);
  free(X_h3_T_host);
  free(dX_h1_host);
  free(dX_h2_host);
  free(dX_h3_host);
  free(dX_out_host);
  free(dX_h1_activation_host);
  free(dX_h2_activation_host);
  free(dX_h3_activation_host);
  free(dX_out_activation_host);

  free(W_h1_host);
  free(W_h1_full_host);
  free(W_h2_host);
  free(W_h2_full_host);
  free(W_h3_host);
  free(W_out_host);

  free(h1_full_to_unique_host);
  free(h1_unique_to_full_host);
  free(h2_full_to_unique_host);
  free(h2_unique_to_full_host);

  free(B_h1_host);
  free(B_h2_host);
  free(B_h3_host);
  free(B_out_host);

  free(loss_host);
  free(predicted_host);
}

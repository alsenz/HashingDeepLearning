#define ADAM 1
#define BETA1 0.9
#define BETA2 0.999
#define EPS 0.00000001

// 1: wta; 2: Densified wta; 3: topk minhash; 4: simhash
#define HASH_FUNCTION_WTA 1
#define HASH_FUNCTION_DWTA 2
#define HASH_FUNCTION_TOPK_MIN 3
#define HASH_FUNCTION_SIMHASH 4

#define HashFunction 3
#define BUCKETSIZE 128
// for minhash
#define TOPK 30
// for simhash
#define Ratio 3
// for wta/dwta
#define binsizeConst 8

// Mode 1: Topk thresholding Mode 4: Sampling
#define MODE_TOPK_THRESHOLD 1
#define MODE_SAMPLING 4

#define Mode 4

#define THRESH 2

#define FIFO 1

#define LOADWEIGHT 1

#define MAPLEN 325056

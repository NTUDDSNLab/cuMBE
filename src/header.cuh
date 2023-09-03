#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include <sys/ioctl.h>
using namespace std;
using namespace std::chrono;
using namespace cooperative_groups;
namespace cg = cooperative_groups;

#define INF 1073741824
#define ONE          1
#define ZERO         0

#define ALGORITHM (argc > 2 ? argv[2] : "cuMBE")
// kernel params
#define LOG_WARP_SIZE   5
#define WARP_SIZE      32
#define NUM_THDS      512
#define NUM_BLKS      999

// for block debugging
#define WORDS_1ROW 16
#define WORD_WIDTH  8
#define LOG_BLK_ID  1

// for clk analyzing
#define NUM_CLK 11
#ifdef DESECTION
#define CLK(IDX) if (!threadIdx.x) { clk[IDX] += clock() - clk_; clk_ = clock(); }
#define CLK_CPU(IDX) clk[IDX] += clock() - clk_; clk_ = clock();
#else  /* DESECTION */
#define CLK(IDX) ;
#define CLK_CPU(IDX) ;
#endif /* DESECTION */

// for independent subtree fetching in MBE kernels
__device__ int P_ptr;
// for while loop breaking in BUBBLE_SORT kernel
__device__ bool done;

typedef struct {
	int start;     // Index of first adjacent node in Ea
	int length;    // Number of adjacent nodes 
} Node;

// typedef struct {
// 	unordered_set<int> L;
// 	unordered_set<int> R;
// } Biclique;

template <class T>
void my_memset(T *SA, T val, int len) {
    for (int i = 0; i < len; i++)
        SA[i] = val;
}

void my_memset_order(int *SA, int val_start, int val_end) {
    for (int i = val_start; i < val_end; i++)
        SA[i - val_start] = i;
}

bool cmp(pair<int, int>& a, pair<int, int>& b) { return a.second > b.second; }
void my_memset_sort(int *SA, int val_start, int val_end, Node *node) {
    vector<pair<int, int>> res;
    for (int i = val_start; i < val_end; i++)
        res.push_back({i, node[i - val_start].length});
    sort(res.begin(), res.end(), cmp);
    for (int i = 0; i < res.size(); i++)
        SA[i] = res[i].first;
}

inline __device__ void PARALLEL_BUBBLE_SORT(int *a, int *n, int *deg) {
    if (*n == 0) return;

    grid_group grid = this_grid();
    int id = threadIdx.x + blockIdx.x * blockDim.x, temp;

    done = false;

    grid.sync();

    while (!done) {

        grid.sync();

        done = true;

        grid.sync();

        for (int i = id; ((i + 1) << 1) < *n; i += blockDim.x * gridDim.x)
            if (deg[a[(i << 1) + 1]] < deg[a[(i << 1) + 2]]) {
                temp = a[(i << 1) + 1];
                a[(i << 1) + 1] = a[(i << 1) + 2];
                a[(i << 1) + 2] = temp;
                done = false;
            }

        grid.sync();

        for (int i = id; ((i << 1) + 1) < *n; i += blockDim.x * gridDim.x)
            if (deg[a[(i << 1)]] < deg[a[(i << 1) + 1]]) {
                temp = a[(i << 1)];
                a[(i << 1)] = a[(i << 1) + 1];
                a[(i << 1) + 1] = temp;
                done = false;
            }

        grid.sync();
        
    }
    
}
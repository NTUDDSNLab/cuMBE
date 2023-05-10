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

#define INF  1073741824
#define ONE  1
#define ZERO 0

#define MIN_SH 1
#define WARP_SIZE 32
#define NUM_THDS 256
#define NUM_BLKS atoi(argv[2])

#define LOG_BLK_ID 1
#define NUM_CLK 10
#define CLK(IDX) if (!threadIdx.x) { clk[IDX] += clock() - clk_; clk_ = clock(); }
#define CLK_CPU(IDX) clk[IDX] += clock() - clk_; clk_ = clock();
// #define CLK(IDX) ;
// #define CLK_CPU(IDX) ;

typedef struct {
	int start;     // Index of first adjacent node in Ea
	int length;    // Number of adjacent nodes 
} Node;

typedef struct {
	unordered_set<int> L;
	unordered_set<int> R;
} Biclique;

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
        res.push_back({i, node[i].length});
    sort(res.begin(), res.end(), cmp);
    for (int i = 0; i < res.size(); i++)
        SA[i] = res[i].first;
}
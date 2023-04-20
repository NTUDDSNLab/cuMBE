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

#define NUM_THDS 512 // argv[2]
#define FEAT_DIM 128 // argv[3]
#define FEAT_AVG   0
#define FEAT_STD  10
#define DEF_STEP  16 // argv[4]

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

struct EdgePair
{
	int row;
	int col;
    EdgePair(int row_, int col_): row(row_), col(col_) {}
};

typedef struct
{
    int num_rows;
    int num_cols;
    unordered_set<int> row;
    unordered_set<int> col;
} Biclique;

typedef struct
{
    int row_end, r;
    int col_end, c;
    int *row;
    int *col;
} Candidate;

random_device rd;
mt19937 en(rd());
uniform_int_distribution<unsigned long long> rand_64;

char t_ms_idx = 0; long long t_ms_start = 0, t_ms_end = 0; vector<long long> t_ms(12, 0);
void mark_time_ms(short new_init_idx) {
    // t_ms_end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    t_ms_end = clock();
    if (new_init_idx >= 0) t_ms_idx = new_init_idx;
    else t_ms[t_ms_idx++] += t_ms_end - t_ms_start;
    t_ms_start = t_ms_end;
    return;
}

char choose_op(int num_rows_cand, int num_cols_cand, int num_rows_bic, int num_cols_bic) {
    // cout << num_rows_cand << ' ' << num_cols_cand << ' ' << num_rows_bic << ' ' << num_cols_bic << ' ';
    if ((num_rows_cand == 0 && num_rows_bic == 1) || (num_cols_cand == 0 && num_cols_bic == 1) || (num_rows_cand | num_cols_cand) == 0) return 'x';
    long long prob_rows = 1, prob_cols = 1;
    if (num_rows_bic != 1 || num_cols_bic != 1) {
        prob_rows = (num_rows_cand != 0) * (num_cols_bic - 1);
        prob_cols = (num_cols_cand != 0) * (num_rows_bic - 1) * 2;
    }
    uniform_int_distribution<> rand_cand(0, prob_rows + prob_cols - 1);
    return (rand_cand(en) < prob_rows ? 'r' : 'c');
}

__global__ void CUDA_GNN_GEN_FEATS(double *feat_i, long long *NUM_NODES, long long *NUM_FEATS)
{
    for(long long i = blockIdx.x; i < *NUM_NODES; i += gridDim.x)
        for(long long j = threadIdx.x; j < *NUM_FEATS; j += blockDim.x)
            feat_i[(*NUM_FEATS) * i + j] = threadIdx.x + blockIdx.x;
}

__global__ void CUDA_CMNBRGNN_KERNEL(Node *node_bic, int *edge_bic, double *feat_bic, Node *node, int *edge, double *feat_i, double *feat_o, int *dest, int *BIC_STEP, long long *NUM_BICS, int *NODE_STEP, long long *NUM_NODES, long long *NUM_FEATS_i, long long *NUM_FEATS_o)
{
    grid_group grid = this_grid();
    int lid = threadIdx.x & 0x1f;
    long long did, sid;
    double result_tmp;

    *dest = 0;
    grid.sync();
    
    if (lid == 0) did = atomicAdd(dest, *BIC_STEP);
    did = __shfl_sync(0xffffffff, did, 0, 32);
    for (; did < *NUM_BICS; ){
        // if (lid == 0) printf("%d\n", did);
        for (int did_end = min(did + *BIC_STEP, *NUM_BICS); did < did_end; did++) {
            for (long long fid = lid; fid < *NUM_FEATS_i; fid += 32) {
                result_tmp = 0;
                for (int eid = node_bic[did].start, eid_end = eid + node_bic[did].length; eid < eid_end; eid++) {
                    sid = edge_bic[eid];
                    result_tmp += feat_i[sid * (*NUM_FEATS_i) + fid];
                }
                feat_bic[did * (*NUM_FEATS_i) + fid] = result_tmp;
        }   }
        if (lid == 0) did = atomicAdd(dest, *BIC_STEP);
        did = __shfl_sync(0xffffffff, did, 0, 32);
    }
    grid.sync();

    *dest = 0;
    grid.sync();
    
    if (lid == 0) did = atomicAdd(dest, *NODE_STEP);
    did = __shfl_sync(0xffffffff, did, 0, 32);
    for (; did < *NUM_NODES; ){
        for (int did_end = min(did + *NODE_STEP, *NUM_NODES); did < did_end; did++) {
            for (long long fid = lid; fid < *NUM_FEATS_i; fid += 32) {
                result_tmp = 0;
                for (int eid = node[did].start, eid_end = eid + node[did].length; eid < eid_end; eid++) {
                    sid = edge[eid];
                    // if (did * (*NUM_FEATS_i) + fid == 118479) printf("%d %d %d\n", eid, fid, feat_i[sid * (*NUM_FEATS_i) + fid]);
                    result_tmp += sid >= 0 ? feat_i[sid * (*NUM_FEATS_i) + fid] : feat_bic[~sid * (*NUM_FEATS_i) + fid];
                }
                feat_o[did * (*NUM_FEATS_i) + fid] = result_tmp;
        }   }
        if (lid == 0) did = atomicAdd(dest, *NODE_STEP);
        did = __shfl_sync(0xffffffff, did, 0, 32);
    }
    grid.sync();
}

int main(int argc, char* argv[])
{
    string str_dataset = argv[1];
    // printf("\033[0;1;33m");
    cout << str_dataset.substr(str_dataset.rfind('/')+1) << "\n";
    // printf("\033[0;1m");
	ifstream fin;
    int _, *NUM_EDGES, SOURCE, *NODE_STEP, *NUM_BIC_EDGES, *BIC_STEP;
    long long *NUM_NODES, *NUM_FEATS, *NUM_BICS;
    cudaMallocManaged(&NODE_STEP, sizeof(int));
    cudaMallocManaged(&NUM_EDGES, sizeof(int));
    cudaMallocManaged(&NUM_NODES, sizeof(long long));
    cudaMallocManaged(&NUM_FEATS, sizeof(long long));
    cudaMallocManaged(&BIC_STEP, sizeof(int));
    cudaMallocManaged(&NUM_BICS, sizeof(long long));
    cudaMallocManaged(&NUM_BIC_EDGES, sizeof(int));

    *NUM_FEATS = (argc > 3) ? atoi(argv[3]) : FEAT_DIM;
    *NODE_STEP = (argc > 4) ? atoi(argv[4]) : DEF_STEP;
    *BIC_STEP = 1;

    fin.open(argv[5]);
    fin >> *NUM_NODES >> *NUM_EDGES;

	Node* node;
	int* edge;
    double* feat_i;
    double* feat_o;
    cudaMallocManaged(&node, sizeof(Node)*(*NUM_NODES));
    cudaMallocManaged(&edge, sizeof(int)*(*NUM_EDGES));
    cudaMallocManaged(&feat_i, sizeof(double)*((*NUM_NODES)*(*NUM_FEATS)));
    cudaMallocManaged(&feat_o, sizeof(double)*((*NUM_NODES)*(*NUM_FEATS)));

    for(int i=0;i<*NUM_NODES;i++) fin >> node[i].start >> node[i].length;
    for(int i=0;i<*NUM_EDGES;i++) fin >> edge[i];
    fin >> *NUM_BICS >> *NUM_BIC_EDGES;

    Node *node_c;
	int *edge_c;
    double* feat_c;
    cudaMallocManaged(&node_c, sizeof(Node)*(*NUM_BICS));     // biclique list (node)
    cudaMallocManaged(&edge_c, sizeof(int)*(*NUM_BIC_EDGES)); // biclique list (edge)
    cudaMallocManaged(&feat_c, sizeof(double)*((*NUM_BICS)*(*NUM_FEATS)));

    for(int i=0;i<*NUM_BICS;i++) fin >> node_c[i].start >> node_c[i].length;
    for(int i=0;i<*NUM_BIC_EDGES;i++) fin >> edge_c[i];
    fin.close();
    
    cout << "Nodes: " << *NUM_NODES << "\n";
    cout << "Edges: " << *NUM_EDGES << "\n";
    cout << "Feats: " << *NUM_FEATS << "\n";
    cout << "Step: "  << *NODE_STEP << "\n";

    int numBlocksPerSM = 1;
    int numThreads = (argc > 2) ? atoi(argv[2]) : NUM_THDS;
    int* dest;
    cudaMallocManaged(&dest, sizeof(int));
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, CUDA_GNN_GEN_FEATS, numThreads, 0);
    dim3 num_blocks_GNN_GEN_FEATS(deviceProp.multiProcessorCount * numBlocksPerSM, 1, 1);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, CUDA_CMNBRGNN_KERNEL, numThreads, 0);
    dim3 num_blocks_CMNBRGNN_KERNEL(deviceProp.multiProcessorCount * numBlocksPerSM, 1, 1);
    dim3 block_size(numThreads, 1, 1);
    void *kernelArgs_GNN_GEN_FEATS[] = {&feat_i, &NUM_NODES, &NUM_FEATS};
    void *kernelArgs_CMNBRGNN_KERNEL[] = {&node_c, &edge_c, &feat_c, &node, &edge, &feat_i, &feat_o, &dest, &BIC_STEP, &NUM_BICS, &NODE_STEP, &NUM_NODES, &NUM_FEATS, &NUM_FEATS};
    cout << "block_size: " << numThreads << "\n";
    cout << "num_blocks_GNN_GEN_FEATS:   " << num_blocks_GNN_GEN_FEATS.x   << "\n";
    cout << "num_blocks_CMNBRGNN_KERNEL: " << num_blocks_CMNBRGNN_KERNEL.x << "\n";

    // cout << "Kernel Start\n";
    cudaLaunchCooperativeKernel((void*)CUDA_GNN_GEN_FEATS, num_blocks_GNN_GEN_FEATS, block_size, kernelArgs_GNN_GEN_FEATS);
    cudaDeviceSynchronize();
    
    // cout << "Prefetch\n";
    cudaMemPrefetchAsync(node_c, sizeof(Node)*(*NUM_BICS), device, NULL);
    cudaMemPrefetchAsync(edge_c, sizeof(int)*(*NUM_BIC_EDGES), device, NULL);
    cudaMemPrefetchAsync(feat_c, sizeof(double)*((*NUM_BICS)*(*NUM_FEATS)), device, NULL);
    cudaMemPrefetchAsync(node, sizeof(Node)*(*NUM_NODES), device, NULL);
    cudaMemPrefetchAsync(edge, sizeof(int)*(*NUM_EDGES), device, NULL);
    cudaMemPrefetchAsync(feat_i, sizeof(double)*((*NUM_NODES)*(*NUM_FEATS)), device, NULL);
    cudaMemPrefetchAsync(feat_o, sizeof(double)*((*NUM_NODES)*(*NUM_FEATS)), device, NULL);
    cudaMemPrefetchAsync(dest, sizeof(int), device, NULL);
    cudaMemPrefetchAsync(BIC_STEP, sizeof(int), device, NULL);
    cudaMemPrefetchAsync(NUM_BICS, sizeof(long long), device, NULL);
    cudaMemPrefetchAsync(NODE_STEP, sizeof(int), device, NULL);
    cudaMemPrefetchAsync(NUM_NODES, sizeof(long long), device, NULL);
    cudaMemPrefetchAsync(NUM_FEATS, sizeof(long long), device, NULL);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        cudaLaunchCooperativeKernel((void*)CUDA_CMNBRGNN_KERNEL, num_blocks_CMNBRGNN_KERNEL, block_size, kernelArgs_CMNBRGNN_KERNEL);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    // cout << "Kernel End\n";

    fin.open(argv[1]);
    fin >> *NUM_NODES >> *NUM_EDGES >> SOURCE;
    for(int i=0;i<*NUM_NODES;i++) fin >> node[i].start >> node[i].length;
    cudaFree(edge);
    cudaMallocManaged(&edge, sizeof(int)*(*NUM_EDGES));
    for(int i=0;i<*NUM_EDGES;i++) fin >> edge[i] >> _;
    fin.close();

    long long num_errors = 0;
    for (long long i = 0; i < *NUM_NODES; i++)
        for (long long j = 0; j < *NUM_FEATS; j++) {
            double result_tmp = 0;
            for (int k = node[i].start, k_end = k + node[i].length; k < k_end; k++) {
                long long sid = edge[k];
                result_tmp += feat_i[sid * (*NUM_FEATS) + j];
            }
            if (result_tmp != feat_o[i * (*NUM_FEATS) + j]) {
                num_errors++;
            }
        }

    cout << "- Time  = " << time << "ms" << "\n";
    cout << "- Error = " << num_errors << '/' << (*NUM_NODES)*(*NUM_FEATS) << "\n";

    // printf("\033[0m");
    cudaFree(BIC_STEP);
    cudaFree(NODE_STEP);
    cudaFree(NUM_EDGES);
    cudaFree(NUM_NODES);
    cudaFree(NUM_FEATS);
    cudaFree(NUM_BICS);
    cudaFree(NUM_BIC_EDGES);
    cudaFree(node);
    cudaFree(edge);
    cudaFree(node_c);
    cudaFree(edge_c);
    cudaFree(feat_c);
    cudaFree(feat_i);
    cudaFree(feat_o);
    cudaFree(dest);
}
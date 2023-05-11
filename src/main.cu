#include <src/header.cuh>
#include <src/transpose.cuh>
#include <src/mbe_cpu.cuh>
#include <src/mbe_cpu_lp.cuh>
#include <src/mbe_gpu_1b.cuh>
#include <src/mbe_gpu.cuh>

void maximal_bic_enum_MineLMBC(int *NUM_L, int *NUM_R, int *NUM_EDGES, Node *node_r, int *edge_r,
                               unordered_set<int> X, unordered_set<int> gammaX, set<int> tailX) {
    unordered_set<int> gammaXprime;
    for (const auto &v: tailX) {
        gammaXprime.clear();
        for (int eid = node_r[v].start, eid_end = eid + node_r[v].length; eid < eid_end; eid++) {
            int u = edge_r[eid];
            if (gammaX.find(u) != gammaX.end())
                gammaXprime.insert(u);
        }
        if (gammaXprime.size() < MIN_SH)
            tailX.erase(v);
    }
    if (X.size() + tailX.size() < MIN_SH)
        return;
    for (const auto &v: tailX) {
        tailX.erase(v);
        unordered_set<int> Y = X;
        Y.insert(v);
        if (X.size() + tailX.size() + 1 > MIN_SH) {
            for (const auto &v_: tailX) {
                int num_N_v = 0;
                for (int eid = node_r[v_].start, eid_end = eid + node_r[v_].length; eid < eid_end; eid++) {
                    int u = edge_r[eid];
                    if (gammaXprime.find(u) != gammaXprime.end())
                        ++num_N_v;
                }
                if (num_N_v == gammaXprime.size())
                    Y.insert(v_);
            }
            
        }
    }
}

int main(int argc, char* argv[])
{
    time_t tmNow = time(0);
    string dataset = argv[1];
    dataset = dataset.substr(dataset.rfind('/')+1);

    Node *node_l, *node_r;
	int *edge_l, *edge_r, *tmp;
    int *NUM_L, *NUM_R, *NUM_EDGES, _;
    // MBE
    int *u2L, *L, *R, *P, *Q;
    int *x, *L_lp, *R_lp, *P_lp, *Q_lp;
    int *Q_rm, *L_buf;
    // MBE_82
    int *g_u2L, *g_L, *g_R, *g_P, *g_Q;
    int *g_x, *g_L_lp, *g_R_lp, *g_P_lp, *g_Q_lp;
    int *g_Q_rm, *g_L_buf, *ori_P;
    cudaMallocManaged(&NUM_EDGES, sizeof(int));
    cudaMallocManaged(&NUM_L    , sizeof(int));
    cudaMallocManaged(&NUM_R    , sizeof(int));

    ifstream fin;
    fin.open(argv[1]);
    fin >> *NUM_R >> *NUM_L >> *NUM_EDGES;
    cudaMallocManaged(&tmp   , sizeof(int )*(*NUM_L    ));
    cudaMallocManaged(&node_l, sizeof(Node)*(*NUM_L    ));
    cudaMallocManaged(&edge_l, sizeof(int )*(*NUM_EDGES));
    cudaMallocManaged(&node_r, sizeof(Node)*(*NUM_R    ));
    cudaMallocManaged(&edge_r, sizeof(int )*(*NUM_EDGES));
    for (int i = 0; i < *NUM_R    ; i++) fin >> node_r[i].start >> node_r[i].length;
    for (int i = 0; i < *NUM_EDGES; i++) fin >> edge_r[i] >> _;
    fin.close();

    int numBlocksPerSM;
    int numThreads = NUM_THDS;
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, CUDA_MBE_82, numThreads, 0);
    int numBlocks_max = deviceProp.multiProcessorCount * numBlocksPerSM;
    int numBlocks = NUM_BLKS > numBlocks_max ? numBlocks_max : \
                    NUM_BLKS == 0 ? 1 : NUM_BLKS < 0 ? 0 : NUM_BLKS;
    dim3 num_blocks_TRANSPOSE(numBlocks, 1, 1);
    dim3 num_blocks_MBE(1, 1, 1);
    dim3 num_blocks_MBE_82(numBlocks, 1, 1);
    dim3 block_size(numThreads, 1, 1);

    bool swap_RL = *NUM_R > *NUM_L;
    if (swap_RL) {
        // cout << "RETURN\n"; return 0;
        swap(*NUM_L, *NUM_R);
        swap(node_l, node_r);
        swap(edge_l, edge_r);
    }

    // MBE
    cudaMallocManaged(&u2L  , sizeof(int)*(*NUM_L)); my_memset_order(u2L, 0, *NUM_L);
    cudaMallocManaged(&L    , sizeof(int)*(*NUM_L)); my_memset_order(L  , 0, *NUM_L);
    cudaMallocManaged(&R    , sizeof(int)*(*NUM_R)); my_memset_order(R  , 0, *NUM_R);
    cudaMallocManaged(&P    , sizeof(int)*(*NUM_R)); my_memset_order(P  , 0, *NUM_R);
    cudaMallocManaged(&Q    , sizeof(int)*(*NUM_R)); my_memset_order(Q  , 0, *NUM_R);
    cudaMallocManaged(&x    , sizeof(int)*(*NUM_R)); my_memset(x   ,     -1, *NUM_R);
    cudaMallocManaged(&L_lp , sizeof(int)*(*NUM_R)); my_memset(L_lp, *NUM_L, *NUM_R);
    cudaMallocManaged(&R_lp , sizeof(int)*(*NUM_R)); my_memset(R_lp,      0, *NUM_R);
    cudaMallocManaged(&P_lp , sizeof(int)*(*NUM_R)); my_memset(P_lp, *NUM_R, *NUM_R);
    cudaMallocManaged(&Q_lp , sizeof(int)*(*NUM_R)); my_memset(Q_lp,      0, *NUM_R);
    cudaMallocManaged(&Q_rm , sizeof(int)*(*NUM_R)); my_memset(Q_rm,    INF, *NUM_R);
    cudaMallocManaged(&L_buf, sizeof(int)*(*NUM_L)); my_memset(L_buf,     0, *NUM_L);
    // MBE_82
    cudaMallocManaged(&g_u2L  , sizeof(int)*(*NUM_L)*numBlocks); for (int i = numBlocks * (*NUM_L); i-- > 0; )   g_u2L[i] =   u2L[i % (*NUM_L)];
    cudaMallocManaged(&g_L    , sizeof(int)*(*NUM_L)*numBlocks); for (int i = numBlocks * (*NUM_L); i-- > 0; )     g_L[i] =     L[i % (*NUM_L)];
    cudaMallocManaged(&g_R    , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )     g_R[i] =     R[i % (*NUM_R)];
    cudaMallocManaged(&g_P    , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )     g_P[i] =     P[i % (*NUM_R)];
    cudaMallocManaged(&g_Q    , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )     g_Q[i] =     Q[i % (*NUM_R)];
    cudaMallocManaged(&g_x    , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )     g_x[i] =     x[i % (*NUM_R)];
    cudaMallocManaged(&g_L_lp , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )  g_L_lp[i] =  L_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_R_lp , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )  g_R_lp[i] =  R_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_P_lp , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )  g_P_lp[i] =  P_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_Q_lp , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )  g_Q_lp[i] =  Q_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_Q_rm , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )  g_Q_rm[i] =  Q_rm[i % (*NUM_R)];
    cudaMallocManaged(&g_L_buf, sizeof(int)*(*NUM_L)*numBlocks); for (int i = numBlocks * (*NUM_L); i-- > 0; ) g_L_buf[i] = L_buf[i % (*NUM_L)];
    cudaMallocManaged(&ori_P, sizeof(int)*(*NUM_R));

    void *kernelArgs_CSR2CSC[] = {&tmp, &node_r, &edge_r, &node_l, &edge_l, &NUM_R, &NUM_L, &NUM_EDGES};
    void *kernelArgs_CSC2CSR[] = {&tmp, &node_l, &edge_l, &node_r, &edge_r, &NUM_L, &NUM_R, &NUM_EDGES};
    void *kernelArgs_MBE[] = {&NUM_L, &NUM_R, &NUM_EDGES, &node_r, &edge_r, &u2L, &L, &R, &P, &Q, &x, &L_lp, &R_lp, &P_lp, &Q_lp};
    void *kernelArgs_MBE_82[] = {&NUM_L, &NUM_R, &NUM_EDGES, &node_l, &edge_l, &node_r, &edge_r, &g_u2L, &g_L, &g_R, &g_P, &g_Q, &g_Q_rm, &g_x, &g_L_lp, &g_R_lp, &g_P_lp, &g_Q_lp, &g_L_buf, &ori_P};

    string algo;
    switch (NUM_BLKS) {
        case -2: algo = "CPU"   ; break;
        case -1: algo = "CPU_lp"; break;
        case  0: algo = "GPU_1B"; break;
        default: algo = "GPU"   ; break;
    }

#ifdef DEBUG
    if (algo == "GPU") {
        cout << "\33[2J\33[1;1H";
    }
#endif /* DEBUG */
    cout << "date/time: "<< ctime(&tmNow);
    cout << "algorithm: " << algo << "\n";
    cout << "dataset: " << dataset << "\n";
    cout << "|R|: " << *NUM_R << ", |L|: " << *NUM_L << ", |E|: " << *NUM_EDGES << "\n";
    cout << "grid_size: " << numBlocks << ", block_size: " << numThreads << "\n";

    // cudaLaunchCooperativeKernel((void*)CUDA_MBE_82, num_blocks_MBE_82, block_size, kernelArgs_MBE_82);
    cudaLaunchCooperativeKernel((void*)CUDA_TRANSPOSE, num_blocks_TRANSPOSE, block_size, swap_RL ? kernelArgs_CSC2CSR : kernelArgs_CSR2CSC);
    cudaDeviceSynchronize();

    my_memset_sort(ori_P, 0, *NUM_R, node_r);
    
    int stat;
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (algo == "CPU")
        maximal_bic_enum_set(NUM_L, NUM_R, NUM_EDGES, node_r, edge_r, u2L, L, R, P, Q, x, L_lp, R_lp, P_lp, Q_lp);
    else if (algo == "CPU_lp")
        maximal_bic_enum(NUM_L, NUM_R, NUM_EDGES, node_r, edge_r, u2L, L, R, P, Q, x, L_lp, R_lp, P_lp, Q_lp);
    else if (algo == "GPU_1B")
        stat = cudaLaunchCooperativeKernel((void*)CUDA_MBE, num_blocks_MBE, block_size, kernelArgs_MBE);
    else if (algo == "GPU")
        stat = cudaLaunchCooperativeKernel((void*)CUDA_MBE_82, num_blocks_MBE_82, block_size, kernelArgs_MBE_82);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // cout << "status: " << stat << ", ";
    cout << "runtime (s): " << time/1000 << "\n";

    cudaFree(tmp);
    cudaFree(node_l);
    cudaFree(edge_l);
    cudaFree(node_r);
    cudaFree(edge_r);
    cudaFree(NUM_L);
    cudaFree(NUM_R);
    cudaFree(NUM_EDGES);
    // MBE
    cudaFree(u2L);
    cudaFree(L);
    cudaFree(R);
    cudaFree(P);
    cudaFree(Q);
    cudaFree(x);
    cudaFree(L_lp);
    cudaFree(R_lp);
    cudaFree(P_lp);
    cudaFree(Q_lp);
    cudaFree(L_buf);
    // MBE_82
    cudaFree(g_u2L);
    cudaFree(g_L);
    cudaFree(g_R);
    cudaFree(g_P);
    cudaFree(g_Q);
    cudaFree(g_x);
    cudaFree(g_L_lp);
    cudaFree(g_R_lp);
    cudaFree(g_P_lp);
    cudaFree(g_Q_lp);
    cudaFree(g_L_buf);
    cudaFree(ori_P);
}
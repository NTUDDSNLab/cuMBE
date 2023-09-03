#include <src/header.cuh>
#include <src/transpose.cuh>
#include <src/mbe_cuMBE.cuh>
#include <src/mbe_noES.cuh>
#include <src/mbe_noRS.cuh>
#include <src/mbe_noWS.cuh>

int main(int argc, char* argv[])
{
    time_t tmNow = time(0);
    string dataset = argv[1];
    dataset = dataset.substr(dataset.rfind('/')+1);

    Node *node_l, *node_r;
	int *edge_l, *edge_r, *tmp;
    int *NUM_L, *NUM_R, *NUM_EDGES, _;
    int *num_mb;
    long long *time_section;
    // MBE
    int *u2L, *v2P, *v2Q, *L, *R, *P, *Q;
    int *x, *L_lp, *R_lp, *P_lp, *Q_lp;
    int *L_buf, *num_N_u, *pre_min;
    // MBE_82
    int *g_u2L, *g_v2P, *g_v2Q, *g_L, *g_R, *g_P, *g_Q;
    int *g_x, *g_L_lp, *g_R_lp, *g_P_lp, *g_Q_lp;
    int *g_L_buf, *g_num_N_u, *g_pre_min;
    int *ori_P, *ori_P1, *ori_Q1, *ori_L1;
    int *P_ptr1, *fix_P_ptr1, *fix_Q_ptr1;
    cudaMallocManaged(&NUM_EDGES   , sizeof(int));
    cudaMallocManaged(&NUM_L       , sizeof(int));
    cudaMallocManaged(&NUM_R       , sizeof(int));
    cudaMallocManaged(&num_mb      , sizeof(int));
    cudaMallocManaged(&time_section, sizeof(long long)*NUM_CLK);
    *num_mb = 0;
    my_memset(time_section, (long long)0, NUM_CLK);

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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, CUDA_MBE_cuMBE, numThreads, 0);
    int numBlocks_max = deviceProp.multiProcessorCount * numBlocksPerSM;
    int numBlocks = NUM_BLKS > numBlocks_max ? numBlocks_max : \
                    NUM_BLKS > 0 ? NUM_BLKS : 1;
    dim3 num_blocks_TRANSPOSE(numBlocks, 1, 1);
    dim3 num_blocks_MBE(numBlocks, 1, 1);
    dim3 block_size(numThreads, 1, 1);

    bool swap_RL = *NUM_R > *NUM_L;
    if (swap_RL) {
        swap(*NUM_L, *NUM_R);
        swap(node_l, node_r);
        swap(edge_l, edge_r);
    }

    // MBE
    cudaMallocManaged(&u2L    , sizeof(int)*(*NUM_L)); my_memset_order(u2L, 0, *NUM_L);
    cudaMallocManaged(&v2P    , sizeof(int)*(*NUM_R)); my_memset_order(v2P, 0, *NUM_R);
    cudaMallocManaged(&v2Q    , sizeof(int)*(*NUM_R)); my_memset_order(v2Q, 0, *NUM_R);
    cudaMallocManaged(&L      , sizeof(int)*(*NUM_L)); my_memset_order(L  , 0, *NUM_L);
    cudaMallocManaged(&R      , sizeof(int)*(*NUM_R)); my_memset_order(R  , 0, *NUM_R);
    cudaMallocManaged(&P      , sizeof(int)*(*NUM_R)); my_memset_order(P  , 0, *NUM_R);
    cudaMallocManaged(&Q      , sizeof(int)*(*NUM_R)); my_memset_order(Q  , 0, *NUM_R);
    cudaMallocManaged(&x      , sizeof(int)*(*NUM_R)); my_memset(x   ,     -1, *NUM_R);
    cudaMallocManaged(&L_lp   , sizeof(int)*(*NUM_R)); my_memset(L_lp, *NUM_L, *NUM_R);
    cudaMallocManaged(&R_lp   , sizeof(int)*(*NUM_R)); my_memset(R_lp,      0, *NUM_R);
    cudaMallocManaged(&P_lp   , sizeof(int)*(*NUM_R)); my_memset(P_lp, *NUM_R, *NUM_R);
    cudaMallocManaged(&Q_lp   , sizeof(int)*(*NUM_R)); my_memset(Q_lp,      0, *NUM_R);
    cudaMallocManaged(&L_buf  , sizeof(int)*(*NUM_L)); my_memset(L_buf,     0, *NUM_L);
    cudaMallocManaged(&num_N_u, sizeof(int)*(*NUM_R)); my_memset(num_N_u,   0, *NUM_R);
    cudaMallocManaged(&pre_min, sizeof(int)*(*NUM_R)); my_memset(pre_min,   1, *NUM_R);
    // MBE_82
    cudaMallocManaged(&g_u2L    , sizeof(int)*(*NUM_L)*numBlocks); for (int i = numBlocks * (*NUM_L); i-- > 0; )     g_u2L[i] =     u2L[i % (*NUM_L)];
    cudaMallocManaged(&g_v2P    , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )     g_v2P[i] =     v2P[i % (*NUM_R)];
    cudaMallocManaged(&g_v2Q    , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )     g_v2Q[i] =     v2Q[i % (*NUM_R)];
    cudaMallocManaged(&g_L      , sizeof(int)*(*NUM_L)*numBlocks); for (int i = numBlocks * (*NUM_L); i-- > 0; )       g_L[i] =       L[i % (*NUM_L)];
    cudaMallocManaged(&g_R      , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )       g_R[i] =       R[i % (*NUM_R)];
    cudaMallocManaged(&g_P      , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )       g_P[i] =       P[i % (*NUM_R)];
    cudaMallocManaged(&g_Q      , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )       g_Q[i] =       Q[i % (*NUM_R)];
    cudaMallocManaged(&g_x      , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )       g_x[i] =       x[i % (*NUM_R)];
    cudaMallocManaged(&g_L_lp   , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )    g_L_lp[i] =    L_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_R_lp   , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )    g_R_lp[i] =    R_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_P_lp   , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )    g_P_lp[i] =    P_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_Q_lp   , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )    g_Q_lp[i] =    Q_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_L_buf  , sizeof(int)*(*NUM_L)*numBlocks); for (int i = numBlocks * (*NUM_L); i-- > 0; )   g_L_buf[i] =   L_buf[i % (*NUM_L)];
    cudaMallocManaged(&g_num_N_u, sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; ) g_num_N_u[i] = num_N_u[i % (*NUM_R)];
    cudaMallocManaged(&g_pre_min, sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; ) g_pre_min[i] = pre_min[i % (*NUM_R)];
    cudaMallocManaged(&ori_P     , sizeof(int)*(*NUM_R));
    cudaMallocManaged(&ori_P1    , sizeof(int)*(*NUM_R)*numBlocks);
    cudaMallocManaged(&ori_Q1    , sizeof(int)*(*NUM_R)*numBlocks);
    cudaMallocManaged(&ori_L1    , sizeof(int)*(*NUM_L)*numBlocks);
    cudaMallocManaged(&P_ptr1    , sizeof(int)         *numBlocks);
    cudaMallocManaged(&fix_P_ptr1, sizeof(int)         *numBlocks);
    cudaMallocManaged(&fix_Q_ptr1, sizeof(int)         *numBlocks);

    void *kernelArgs_CSR2CSC[] = {&tmp, &node_r, &edge_r, &node_l, &edge_l, &NUM_R, &NUM_L, &NUM_EDGES};
    void *kernelArgs_CSC2CSR[] = {&tmp, &node_l, &edge_l, &node_r, &edge_r, &NUM_L, &NUM_R, &NUM_EDGES};
    void *kernelArgs_MBE[] = {&NUM_L, &NUM_R, &NUM_EDGES, &node_l, &edge_l, &node_r, &edge_r,
                              &g_u2L, &g_v2P, &g_v2Q, &g_L, &g_R, &g_P, &g_Q, &g_x, &g_L_lp, &g_R_lp, &g_P_lp, &g_Q_lp, &g_L_buf, &g_num_N_u, &g_pre_min,
                              &ori_P, &ori_P1, &ori_Q1, &ori_L1, &P_ptr1, &fix_P_ptr1, &fix_Q_ptr1, &num_mb, &time_section};

    string algo = ALGORITHM;
    algo = algo == "noRS" || algo == "noES" || algo == "noWS" ? algo : "cuMBE";
    string filename = dataset.substr(0, dataset.rfind('.')) + "_" + algo;
    
    ofstream fout;
    fout.open("result/"+filename);

#ifdef DEBUG
    cout << "\33[2J\33[1;1H";
#endif /* DEBUG */
    cout << "date/time: "<< ctime(&tmNow);
    cout << "algorithm: " << algo << endl;
    cout << "dataset: " << dataset << endl;
    cout << "|R|: " << *NUM_R << ", |L|: " << *NUM_L << ", |E|: " << *NUM_EDGES << endl;
    cout << "grid_size: " << numBlocks << ", block_size: " << numThreads << endl;

    fout << "date/time: "<< ctime(&tmNow);
    fout << "algorithm: " << algo << endl;
    fout << "dataset: " << dataset << endl;
    fout << "|R|: " << *NUM_R << ", |L|: " << *NUM_L << ", |E|: " << *NUM_EDGES << endl;
    fout << "grid_size: " << numBlocks << ", block_size: " << numThreads << endl;

    cudaLaunchCooperativeKernel((void*)CUDA_TRANSPOSE, num_blocks_TRANSPOSE, block_size, swap_RL ? kernelArgs_CSC2CSR : kernelArgs_CSR2CSC);
    cudaDeviceSynchronize();
    my_memset_sort(ori_P, 0, *NUM_R, node_r);
    
    cudaMemPrefetchAsync(&NUM_EDGES   , sizeof(int), device, NULL);
    cudaMemPrefetchAsync(&NUM_L       , sizeof(int), device, NULL);
    cudaMemPrefetchAsync(&NUM_R       , sizeof(int), device, NULL);
    cudaMemPrefetchAsync(&num_mb      , sizeof(int), device, NULL);
    cudaMemPrefetchAsync(&time_section, sizeof(long long)*NUM_CLK, device, NULL);

    cudaMemPrefetchAsync(&node_l, sizeof(Node)*(*NUM_L    ), device, NULL);
    cudaMemPrefetchAsync(&edge_l, sizeof(int )*(*NUM_EDGES), device, NULL);
    cudaMemPrefetchAsync(&node_r, sizeof(Node)*(*NUM_R    ), device, NULL);
    cudaMemPrefetchAsync(&edge_r, sizeof(int )*(*NUM_EDGES), device, NULL);
    
    cudaMemPrefetchAsync(g_u2L     , sizeof(int)*(*NUM_L)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_v2P     , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_v2Q     , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_L       , sizeof(int)*(*NUM_L)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_R       , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_P       , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_Q       , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_x       , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_L_lp    , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_R_lp    , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_P_lp    , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_Q_lp    , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_L_buf   , sizeof(int)*(*NUM_L)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_num_N_u , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(g_pre_min , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(ori_P     , sizeof(int)*(*NUM_R)          , device, NULL);
    cudaMemPrefetchAsync(ori_P1    , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(ori_Q1    , sizeof(int)*(*NUM_R)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(ori_L1    , sizeof(int)*(*NUM_L)*numBlocks, device, NULL);
    cudaMemPrefetchAsync(P_ptr1    , sizeof(int)         *numBlocks, device, NULL);
    cudaMemPrefetchAsync(fix_P_ptr1, sizeof(int)         *numBlocks, device, NULL);
    cudaMemPrefetchAsync(fix_Q_ptr1, sizeof(int)         *numBlocks, device, NULL);

    cudaDeviceSynchronize();

    int stat;
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if      (algo == "noRS")
        stat = cudaLaunchCooperativeKernel((void*)CUDA_MBE_noRS , num_blocks_MBE, block_size, kernelArgs_MBE);
    else if (algo == "noES")
        stat = cudaLaunchCooperativeKernel((void*)CUDA_MBE_noES , num_blocks_MBE, block_size, kernelArgs_MBE);
    else if (algo == "noWS")
        stat = cudaLaunchCooperativeKernel((void*)CUDA_MBE_noWS , num_blocks_MBE, block_size, kernelArgs_MBE);
    else if (algo == "cuMBE")
        stat = cudaLaunchCooperativeKernel((void*)CUDA_MBE_cuMBE, num_blocks_MBE, block_size, kernelArgs_MBE);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // cout << "status: " << stat << endl;
    cout << "maximal bicliques: " << *num_mb << endl;
#ifdef DESECTION
    long long sum_time_section = 0;
    for (int i = 0; i < NUM_CLK; i++)
        sum_time_section += time_section[i];
    cout << "time percentage:" << fixed << setprecision(4);
    for (int i = 0; i < NUM_CLK; i++)
        cout << " " << (double)time_section[i] * 100 / sum_time_section;
    cout << setprecision(6) << endl;
    // cout << " " << (double)time_section[8] / 1932735283.2 / 246 << "\n";
    // cout << "runtime (s) (no L): " << (time * ((double)(sum_time_section - time_section[10])) / sum_time_section)/1000 << endl;
#endif /* DESECTION */
#ifdef DEBUG
    cout << "runtime (s): " << time/1000 << endl;
    cout << "\33[" << (numBlocks-1) / WORDS_1ROW + 9 << ";1H";
#else  /* DEBUG */
    cout << "runtime (s): " << time/1000 << endl;
#endif /* DEBUG */

    fout << "maximal bicliques: " << *num_mb << endl;
#ifdef DESECTION
    fout << "time percentage:" << fixed << setprecision(4);
    for (int i = 0; i < NUM_CLK; i++)
        fout << " " << (double)time_section[i] * 100 / sum_time_section;
    fout << setprecision(6) << endl;
    // fout << "runtime (s) (no L): " << (time * ((double)(sum_time_section - time_section[10])) / sum_time_section)/1000 << endl;
#endif /* DESECTION */
    fout << "runtime (s): " << time/1000 << endl;

    cudaFree(tmp);
    cudaFree(node_l);
    cudaFree(edge_l);
    cudaFree(node_r);
    cudaFree(edge_r);
    cudaFree(NUM_L);
    cudaFree(NUM_R);
    cudaFree(NUM_EDGES);
    cudaFree(num_mb);
    cudaFree(time_section);
    // MBE
    cudaFree(u2L);
    cudaFree(v2P);
    cudaFree(v2Q);
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
    cudaFree(num_N_u);
    cudaFree(pre_min);
    // MBE_82
    cudaFree(g_u2L);
    cudaFree(g_v2P);
    cudaFree(g_v2Q);
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
    cudaFree(g_num_N_u);
    cudaFree(g_pre_min);
    cudaFree(ori_P);
    cudaFree(ori_P1);
    cudaFree(ori_Q1);
    cudaFree(ori_L1);
    cudaFree(P_ptr1);
    cudaFree(fix_P_ptr1);
    cudaFree(fix_Q_ptr1);
}
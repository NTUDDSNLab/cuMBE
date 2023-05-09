__device__ int g_clk[NUM_CLK], g_clk_scale;
__device__ int total_bic;
__device__ int P_ptr;

__global__ void CUDA_MBE_82(int *NUM_L, int *NUM_R, int *NUM_EDGES, Node *node, int *edge,
                            int *g_u2L, int *g_L, int *g_R, int *g_P, int *g_Q, int *g_Q_rm,
                            int *g_x, int *g_L_lp, int *g_R_lp, int *g_P_lp, int *g_Q_lp, int *ori_P) {

    int *u2L  = g_u2L  + blockIdx.x * (*NUM_L);
    int *L    = g_L    + blockIdx.x * (*NUM_L);
    int *R    = g_R    + blockIdx.x * (*NUM_R);
    int *P    = g_P    + blockIdx.x * (*NUM_R);
    int *Q    = g_Q    + blockIdx.x * (*NUM_R);
    int *x    = g_x    + blockIdx.x * (*NUM_R);
    int *L_lp = g_L_lp + blockIdx.x * (*NUM_R);
    int *R_lp = g_R_lp + blockIdx.x * (*NUM_R);
    int *P_lp = g_P_lp + blockIdx.x * (*NUM_R);
    int *Q_lp = g_Q_lp + blockIdx.x * (*NUM_R);
    int *Q_rm = g_Q_rm + blockIdx.x * (*NUM_R);
    grid_group grid = this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_total_thds = gridDim.x * blockDim.x;
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 0x1f;
    int num_warps = blockDim.x >> 5;
    int num_maximal_bicliques = 0;
    __shared__ int lvl;
    __shared__ int *x_cur;
    __shared__ int *L_lp_cur, *L_lp_nxt;
    __shared__ int *R_lp_cur, *R_lp_nxt;
    __shared__ int *P_lp_cur, *P_lp_nxt;
    __shared__ int *Q_lp_cur, *Q_lp_nxt;
    __shared__ bool is_recursive;
    __shared__ bool is_maximal;
    __shared__ int num_L_nxt, num_N_v[NUM_THDS >> 5];

    __shared__ long long clk[NUM_CLK], clk_;
    if (!threadIdx.x) {
        clk_ = clock();
        for (int i = 0; i < NUM_CLK; i++)
            clk[i] = 0;
        if (!blockIdx.x)
            for (int i = 0; i < NUM_CLK; i++)
                g_clk[i] = 0;
    }

    if (!tid) {
        P_ptr = *NUM_R - 1;
        total_bic = 0;
    }

    grid.sync();

    if (!threadIdx.x) {
        lvl = 0;
        // P_lp[0] = *NUM_R + blockIdx.x;
        P_lp[0] = *NUM_R;
    }

    __syncthreads();

    for (; lvl >= 0; ) {

        if (!threadIdx.x) {
            x_cur    = &(   x[lvl]);
            L_lp_cur = &(L_lp[lvl]); L_lp_nxt = &(L_lp[lvl+1]);
            R_lp_cur = &(R_lp[lvl]); R_lp_nxt = &(R_lp[lvl+1]);
            P_lp_cur = &(P_lp[lvl]); P_lp_nxt = &(P_lp[lvl+1]);
            Q_lp_cur = &(Q_lp[lvl]); Q_lp_nxt = &(Q_lp[lvl+1]);
            is_recursive = false;
        }
        
        __syncthreads();

        if (lvl == 0)
        // while P ≠ ∅ do
        while (*P_lp_cur >= gridDim.x || 1) {
            
            __syncthreads();

            CLK(0);

            if (!threadIdx.x) {
                // atomically get a new 1-level sub-tree
                // Q <--- Q ∪ {x before P_ptr};
                //// for (int i = *P_lp_cur, i_end = *P_lp_cur -= gridDim.x; --i > i_end; ) {
                ////     if (i < *NUM_R) {
                ////         Q_rm[*Q_lp_cur] = INF;
                ////         Q[(*Q_lp_cur)++] = i;
                ////     }
                //// }
                for (int i = *P_lp_cur, i_end = *P_lp_cur = atomicAdd(&P_ptr, -1); --i > i_end; ) {
                    if (i >= 0) {
                        Q_rm[*Q_lp_cur] = INF;
                        Q[(*Q_lp_cur)++] = ori_P[i];
                    }
                }

                // reset P to ordered
                for (int i = 0; i < *NUM_R; i++)
                    P[i] = ori_P[i];

                // Select x from P;
                // P <--- P \ {x before P_ptr and x_cur};
                *x_cur = ori_P[*P_lp_cur];
                
                // R' <--- R ∪ {x};
                *R_lp_nxt = *R_lp_cur;
                R[(*R_lp_nxt)++] = *x_cur;

                //// *L_lp_nxt = 0;
                num_L_nxt = 0;
            
                // P' ← ∅; Q' ← ∅;
                *P_lp_nxt = 0; *Q_lp_nxt = *Q_lp_cur;
                is_maximal = true;
            }
            
            __syncthreads();

            CLK(1);

            if (*P_lp_cur < 0) break;

            // |L'|
            for (int l = threadIdx.x; l < *NUM_L; l += blockDim.x)
                L[l] = L[l] > lvl ? lvl : L[l];

            __syncthreads();

            CLK(2);
            
            // L' <--- {u ∈ L | (u, x) ∈ E(G)};
            for (int eid = node[*x_cur].start + threadIdx.x, eid_end = node[*x_cur].start + node[*x_cur].length; eid < eid_end; eid += blockDim.x) {
                int l = edge[eid];
                if (L[l] == lvl) {
                    L[l]++;
                    atomicAdd(&num_L_nxt, 1);
                }
            }

            __syncthreads();

            CLK(3);

            /*TODO  atomic*/

            // foreach v ∈ Q
                    // WARNING // *Q_lp_cur // WARNING //
            for (int i = wid; i < *Q_lp_nxt; i += num_warps) {

                if (Q_rm[i] < lvl) {
                    __syncwarp();
                    continue;
                }

                int v = Q[i];

                if (!lid)
                    num_N_v[wid] = 0; // |N[v]|

                __syncwarp();

                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node[v].start + lid, eid_end = node[v].start + node[v].length; eid < eid_end; eid += WARP_SIZE) {
                    int l = edge[eid];
                    if (L[l] > lvl)
                        atomicAdd(&(num_N_v[wid]), 1);
                }

                __syncwarp();

                if (!lid) {
                    Q_rm[i] = INF;

                    // if |N[v]| = |L'| then
                    if (num_N_v[wid] == num_L_nxt) {
                        is_maximal = false;
                        *Q_lp_nxt = 0;
                    }

                    // else if |N[v]| > 0 then
                    else if (num_N_v[wid] == 0)
                        // Q' ← Q' ∪ {v};
                        // swap(Q[(*Q_ls_nxt)++], Q[i]);
                        Q_rm[i] = lvl;
                }

                __syncwarp();
                
            }
            
            CLK(4);

            __syncthreads();

            // if is_maximal = TRUE then
            if (is_maximal == true) {

                // foreach v ∈ P do
                for (int align_i = 0; align_i < *P_lp_cur; align_i += num_warps) {
                    int i = align_i + wid;

                    if (i < *P_lp_cur) {

                        int v = P[i];

                        if (!lid)
                            num_N_v[wid] = 0; // |N[v]|

                        __syncwarp();

                        // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                        for (int eid = node[v].start + lid, eid_end = node[v].start + node[v].length; eid < eid_end; eid += WARP_SIZE) {
                            int l = edge[eid];
                            if (L[l] > lvl)
                                atomicAdd(&(num_N_v[wid]), 1);
                        }

                        __syncwarp();
                        
                        if (!lid) {
                            // if |N[v]| = |L'| then
                            if (num_N_v[wid] == num_L_nxt)
                                // R' ← R' ∪ {v};
                                R[atomicAdd(R_lp_nxt, 1)] = v;
                        }
                    }
                    
                    __syncthreads();

                    // serial maintain P
                    if (!threadIdx.x) {
                        for (int j = 0; j < num_warps; j++) {
                            i = align_i + j;
                            if (i == *P_lp_cur) break;
                            int v = P[i];
                            // else if |N[v]| > 0 then
                            if (num_N_v[j] != num_L_nxt && num_N_v[j] > 0/* && node[v].length >= node[*x_cur].length*//* && (node[v].length > node[*x_cur].length || (node[v].length == node[*x_cur].length && v > *x_cur))*/) {
                                // P' ← P' ∪ {v};
                                int P_tmp = P[*P_lp_nxt];
                                P[(*P_lp_nxt)++] = P[i];
                                P[i] = P_tmp;
                            }
                        }
                    }
                    
                    __syncthreads();

                }
            
                CLK(5);

                if (!threadIdx.x) {
#ifdef DEBUG

                    //// if (blockIdx.x == LOG_BLK_ID) {
                    ////     // PRINT(L', R');
                    ////     printf("\n> Find maximal biclique (No. %d)", num_maximal_bicliques);
                    ////     printf("\nL':");
                    ////     for (int i = 0; i < *NUM_L; i++)
                    ////         if (L[i] > lvl)
                    ////             printf(" %d", i);
                    ////     printf("\nR':");
                    ////     for (int i = 0; i < *R_lp_nxt; i++)
                    ////         printf(" %d", R[i]);
                    ////     printf("\n");
                    //// }

                    // save maximal bicliques
                    //// Biclique new_maximal_bicliques;
                    //// for (int i = 0; i < *L_lp_nxt; i++)
                    ////     new_maximal_bicliques.L.insert(L[i]);
                    //// for (int i = 0; i < *R_lp_nxt; i++)
                    ////     new_maximal_bicliques.R.insert(R[i]);
                    //// maximal_bicliques.push_back(new_maximal_bicliques);

                    // if (++num_maximal_bicliques > 0)
                    printf("\33[%d;%dH%d\n", blockIdx.x / 10 + 9, (blockIdx.x % 10) * 10 + 1, ++num_maximal_bicliques);
#else  /* DEBUG */
                    ++num_maximal_bicliques;
#endif /* DEBUG */
                    // if P' ≠ ∅ then
                    if (*P_lp_nxt != 0) {
                        // biclique_find(G, L', R', P', Q');
                        //// printf("\n往 下 安安");
                        lvl++;
                        is_recursive = true;
                    }

                }

                __syncthreads();
            
                CLK(6);

                if (is_recursive)
                    break;

            }
            else {
                //// printf("\n不安安");
            }

            if (!threadIdx.x) {

                // Q ← Q ∪ {x};
                Q_rm[*Q_lp_cur] = INF;
                Q[(*Q_lp_cur)++] = *x_cur;
                //// printf("\n往 右 安安");

            }

        }

        else // lvl >= 1
        // while P ≠ ∅ do
        while (*P_lp_cur != 0) {
            
            __syncthreads();

            CLK(0);

            // 5/8 revised start //
            
            if (!threadIdx.x) {
                num_L_nxt = INF;
            }

            __syncthreads();

            // foreach v ∈ P do
            for (int i = *P_lp_cur; i-- > 0; ) {

                int v = P[i];

                if (!threadIdx.x)
                    num_N_v[0] = 0; // |N[v]|

                __syncthreads();

                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node[v].start + threadIdx.x, eid_end = node[v].start + node[v].length; eid < eid_end; eid += blockDim.x) {
                    int l = edge[eid];
                    if (L[l] >= lvl && atomicAdd(&(num_N_v[0]), 1) == num_L_nxt)
                        break;
                }

                __syncthreads();

                if (!threadIdx.x && num_N_v[0] < num_L_nxt) {
                    num_L_nxt = num_N_v[0];
                    int P_tmp = P[*P_lp_cur - 1];
                    P[*P_lp_cur - 1] = P[i];
                    P[i] = P_tmp;
                }

            }
            
            __syncthreads();

            CLK(7);

            // 5/8 revised end //

            if (!threadIdx.x) {

                // Select x from P;
                // P <--- P \ {x};
                *x_cur = P[--(*P_lp_cur)];
                
                // R' <--- R ∪ {x};
                *R_lp_nxt = *R_lp_cur;
                R[(*R_lp_nxt)++] = *x_cur;

                //// *L_lp_nxt = 0;
                num_L_nxt = 0;
                
                // P' ← ∅; Q' ← ∅;
                *P_lp_nxt = 0; *Q_lp_nxt = *Q_lp_cur;
                is_maximal = true;
            }
            
            __syncthreads();

            CLK(1);

            // |L'|
            for (int l = threadIdx.x; l < *NUM_L; l += blockDim.x)
                L[l] = L[l] > lvl ? lvl : L[l];

            __syncthreads();

            CLK(2);
            
            // L' <--- {u ∈ L | (u, x) ∈ E(G)};
            for (int eid = node[*x_cur].start + threadIdx.x, eid_end = node[*x_cur].start + node[*x_cur].length; eid < eid_end; eid += blockDim.x) {
                int l = edge[eid];
                if (L[l] == lvl) {
                    L[l]++;
                    atomicAdd(&num_L_nxt, 1);
                }
            }

            __syncthreads();

            CLK(3);

            // foreach v ∈ Q
                    // WARNING // *Q_lp_cur // WARNING //
            for (int i = wid; i < *Q_lp_nxt; i += num_warps) {

                if (Q_rm[i] < lvl) {
                    __syncwarp();
                    continue;
                }

                int v = Q[i];

                if (!lid)
                    num_N_v[wid] = 0; // |N[v]|

                __syncwarp();

                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node[v].start + lid, eid_end = node[v].start + node[v].length; eid < eid_end; eid += WARP_SIZE) {
                    int l = edge[eid];
                    if (L[l] > lvl)
                        atomicAdd(&(num_N_v[wid]), 1);
                }

                __syncwarp();

                if (!lid) {
                    Q_rm[i] = INF;

                    // if |N[v]| = |L'| then
                    if (num_N_v[wid] == num_L_nxt) {
                        is_maximal = false;
                        *Q_lp_nxt = 0;
                    }

                    // else if |N[v]| > 0 then
                    else if (num_N_v[wid] == 0)
                        // Q' ← Q' ∪ {v};
                        // swap(Q[(*Q_ls_nxt)++], Q[i]);
                        Q_rm[i] = lvl;
                }

                __syncwarp();
                
            }
            
            CLK(4);

            __syncthreads();

            // if is_maximal = TRUE then
            if (is_maximal == true) {

                // foreach v ∈ P do
                for (int align_i = 0; align_i < *P_lp_cur; align_i += num_warps) {
                    int i = align_i + wid;

                    if (i < *P_lp_cur) {

                        int v = P[i];

                        if (!lid)
                            num_N_v[wid] = 0; // |N[v]|

                        __syncwarp();

                        // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                        for (int eid = node[v].start + lid, eid_end = node[v].start + node[v].length; eid < eid_end; eid += WARP_SIZE) {
                            int l = edge[eid];
                            if (L[l] > lvl)
                                atomicAdd(&(num_N_v[wid]), 1);
                        }

                        __syncwarp();
                        
                        if (!lid) {
                            // if |N[v]| = |L'| then
                            if (num_N_v[wid] == num_L_nxt)
                                // R' ← R' ∪ {v};
                                R[atomicAdd(R_lp_nxt, 1)] = v;
                        }
                    }
                    
                    __syncthreads();
                    
                    // serial maintain P
                    if (!threadIdx.x) {
                        for (int j = 0; j < num_warps; j++) {
                            i = align_i + j;
                            if (i == *P_lp_cur) break;
                            int v = P[i];
                            // else if |N[v]| > 0 then
                            if (num_N_v[j] != num_L_nxt && num_N_v[j] > 0) {
                                // P' ← P' ∪ {v};
                                int P_tmp = P[*P_lp_nxt];
                                P[(*P_lp_nxt)++] = P[i];
                                P[i] = P_tmp;
                            }
                        }
                    }
                    
                    __syncthreads();

                }
            
                CLK(5);

                if (!threadIdx.x) {
#ifdef DEBUG

                    //// if (blockIdx.x == LOG_BLK_ID) {
                    ////     // PRINT(L', R');
                    ////     printf("\n> Find maximal biclique (No. %d)", num_maximal_bicliques);
                    ////     printf("\nL':");
                    ////     for (int i = 0; i < *NUM_L; i++)
                    ////         if (L[i] > lvl)
                    ////             printf(" %d", i);
                    ////     printf("\nR':");
                    ////     for (int i = 0; i < *R_lp_nxt; i++)
                    ////         printf(" %d", R[i]);
                    ////     printf("\n");
                    //// }

                    // save maximal bicliques
                    //// Biclique new_maximal_bicliques;
                    //// for (int i = 0; i < *L_lp_nxt; i++)
                    ////     new_maximal_bicliques.L.insert(L[i]);
                    //// for (int i = 0; i < *R_lp_nxt; i++)
                    ////     new_maximal_bicliques.R.insert(R[i]);
                    //// maximal_bicliques.push_back(new_maximal_bicliques);

                    // if (++num_maximal_bicliques > 0)
                    printf("\33[%d;%dH%d\n", blockIdx.x / 10 + 9, (blockIdx.x % 10) * 10 + 1, ++num_maximal_bicliques);
#else  /* DEBUG */
                    ++num_maximal_bicliques;
#endif /* DEBUG */

                    // if P' ≠ ∅ then
                    if (*P_lp_nxt != 0) {
                        // biclique_find(G, L', R', P', Q');
                        //// printf("\n往 下 安安");
                        lvl++;
                        is_recursive = true;
                    }

                }

                __syncthreads();
            
                CLK(6);

                if (is_recursive)
                    break;

            }
            else {
                //// printf("\n不安安");
            }

            if (!threadIdx.x) {

                // Q ← Q ∪ {x};
                Q_rm[*Q_lp_cur] = INF;
                Q[(*Q_lp_cur)++] = *x_cur;
                //// printf("\n往 右 安安");

            }

        }
        
        __syncthreads();

        if (!threadIdx.x) {

            if (!is_recursive) {
                if (lvl--) {
                    Q_rm[Q_lp[lvl]] = INF;
                    Q[Q_lp[lvl]++] = x[lvl];
                }
                //// printf("\n往 上 安安");
                //// printf("\n往 右 安安");
            }

        }

        __syncthreads();

    }

    grid.sync();
    
    if (!threadIdx.x)
        atomicAdd(&total_bic, num_maximal_bicliques);
    grid.sync();
    if (!tid) {
        g_clk_scale = 0;
        for (int num = total_bic >> 8; num >>= 1; g_clk_scale += 2) ;
    }
    grid.sync();
    if (!threadIdx.x)
        for (int i = 0; i < NUM_CLK; i++) {
            clk[i] >>= g_clk_scale;
            atomicAdd(&(g_clk[i]), (int)clk[i]);
        }
    grid.sync();
    if (!tid) {
#ifdef DEBUG
        printf("\33[6;1H");
#endif /* DEBUG */
        printf("maximal bicliques: %d\n", total_bic);
        printf("time:");
        for (int i = 0; i < NUM_CLK; i++)
            printf(" %d", g_clk[i]);
        printf("\n");
    }
    grid.sync();
}
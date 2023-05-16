__device__ int g_clk[NUM_CLK], g_clk_scale;
__device__ int total_bic;
__device__ int P_ptr;

__global__ void CUDA_MBE_82(int *NUM_L, int *NUM_R, int *NUM_EDGES,
                            Node *node_l, int *edge_l, Node *node_r, int *edge_r,
                            int *g_u2L, int *g_v2Q, int *g_L, int *g_R, int *g_P, int *g_Q, int *g_Q_rm,
                            int *g_x, int *g_L_lp, int *g_R_lp, int *g_P_lp, int *g_Q_lp,
                            int *g_L_buf, int *g_num_N_u, int *ori_P) {

    int *u2L     = g_u2L     + blockIdx.x * (*NUM_L);
    int *v2Q     = g_v2Q     + blockIdx.x * (*NUM_R);
    int *L       = g_L       + blockIdx.x * (*NUM_L);
    int *R       = g_R       + blockIdx.x * (*NUM_R);
    int *P       = g_P       + blockIdx.x * (*NUM_R);
    int *Q       = g_Q       + blockIdx.x * (*NUM_R);
    int *x       = g_x       + blockIdx.x * (*NUM_R);
    int *L_lp    = g_L_lp    + blockIdx.x * (*NUM_R);
    int *R_lp    = g_R_lp    + blockIdx.x * (*NUM_R);
    int *P_lp    = g_P_lp    + blockIdx.x * (*NUM_R);
    int *Q_lp    = g_Q_lp    + blockIdx.x * (*NUM_R);
    int *Q_rm    = g_Q_rm    + blockIdx.x * (*NUM_R);
    int *L_buf   = g_L_buf   + blockIdx.x * (*NUM_L);
    int *num_N_u = g_num_N_u + blockIdx.x * (*NUM_R);
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
    __shared__ int num_L_nxt, num_N_v[NUM_THDS >> 5], num_N_L;

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

            // atomically get a new 1-level sub-tree

            if (!threadIdx.x) {
                *P_lp_nxt = *P_lp_cur;
                *P_lp_cur = atomicAdd(&P_ptr, -1);
                *P_lp_cur = *P_lp_cur >= 0 ? *P_lp_cur : -1;
            }
            
            __syncthreads();

            if (*P_lp_cur == -1) break;

            if (!threadIdx.x) {
                for (int i = *P_lp_cur + 1; i < *P_lp_nxt; i++) {
                    
                    Q_rm[*Q_lp_cur] = INF;

                    int v = ori_P[i];
                    int q = v2Q[v];

                    // swap Q
                    int Q_tmp = Q[*Q_lp_cur];
                    Q[*Q_lp_cur] = v;
                    Q[q] = Q_tmp;
                    // maintain v2Q
                    v2Q[v] = (*Q_lp_cur)++;;
                    v2Q[Q_tmp] = q;

                }
            }
            

            // for (int i = *P_lp_cur, i_end = *P_lp_cur = atomicAdd(&P_ptr, -1); --i > i_end; ) {
            //     if (i >= 0) {
            //         Q_rm[*Q_lp_cur] = INF;
            //         Q[(*Q_lp_cur)++] = ori_P[i];
            //     }
            // }

            // reset P to ordered
            for (int i = threadIdx.x; i < *P_lp_cur; i += blockDim.x)
                P[i] = ori_P[i];
            
            __syncthreads();

            CLK(8);

            if (!threadIdx.x) {

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

                // |L'|
                *L_lp_nxt = 0;

                // scan from L
                num_N_L = 0;
            }
            
            __syncthreads();

            CLK(1);

            // for (int l = threadIdx.x; l < *NUM_L; l += blockDim.x)
            //     L[l] = L[l] > lvl ? lvl : L[l];

            __syncthreads();

            CLK(2);
            
            // L' <--- {u ∈ L | (u, x) ∈ E(G)};
            for (int eid = node_r[*x_cur].start + threadIdx.x, eid_end = node_r[*x_cur].start + node_r[*x_cur].length; eid < eid_end; eid += blockDim.x) {
                int u = edge_r[eid];
                int l = u2L[u];
                if (l < *L_lp_cur)
                    L_buf[atomicAdd(&num_L_nxt, 1)] = u;
            }

            __syncthreads();

            if (!threadIdx.x)
                for (int i = 0; i < num_L_nxt; i++) {
                    int u = L_buf[i];
                    int l = u2L[u];
                    // swap(L[(*L_lp_nxt)++], L[l]);
                    int L_tmp = L[*L_lp_nxt];
                    L[(*L_lp_nxt)++] = L[l];
                    L[l] = L_tmp;
                    // swap(u2L[L[l]], u2L[u]);
                    int u2L_tmp = u2L[L[l]];
                    u2L[L[l]] = u2L[u];
                    u2L[u] = u2L_tmp;
                }

            __syncthreads();

            CLK(3);

            // foreach u ∈ L'
            for (int i = wid; i < *L_lp_nxt; i += num_warps) {

                int u = L[i];

                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node_l[u].start + lid, eid_end = node_l[u].start + node_l[u].length; eid < eid_end; eid += WARP_SIZE) {
                    int v = edge_l[eid];
                    int q = v2Q[v];
                    if (q < *Q_lp_cur && atomicAdd(&(num_N_u[v]), 1) == 0)
                        L_buf[atomicAdd(&num_N_L, 1)] = v;
                }
            }

            __syncthreads();

            for (int i = threadIdx.x; i < num_N_L; i += blockDim.x) {
                int v = L_buf[i];
                if (num_N_u[v] == num_L_nxt)
                    is_maximal = false;
                num_N_u[v] = 0;
            }

            // foreach v ∈ Q
                    // WARNING // *Q_lp_cur // WARNING //
            // for (int i = wid; i < *Q_lp_nxt; i += num_warps) {

            //     if (Q_rm[i] < lvl) {
            //         __syncwarp();
            //         continue;
            //     }

            //     int v = Q[i];

            //     if (!lid)
            //         num_N_v[wid] = 0; // |N[v]|

            //     __syncwarp();

            //     // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
            //     for (int eid = node_r[v].start + lid, eid_end = node_r[v].start + node_r[v].length; eid < eid_end; eid += WARP_SIZE) {
            //         int u = edge_r[eid];
            //         int l = u2L[u];
            //         if (l < *L_lp_nxt)
            //             atomicAdd(&(num_N_v[wid]), 1);
            //     }

            //     __syncwarp();

            //     if (!lid) {
            //         Q_rm[i] = INF;

            //         // if |N[v]| = |L'| then
            //         if (num_N_v[wid] == num_L_nxt) {
            //             is_maximal = false;
            //             *Q_lp_nxt = 0;
            //         }

            //         // else if |N[v]| > 0 then
            //         else if (num_N_v[wid] == 0)
            //             // Q' ← Q' ∪ {v};
            //             // swap(Q[(*Q_ls_nxt)++], Q[i]);
            //             Q_rm[i] = lvl;
            //     }

            //     __syncwarp();
                
            // }

            __syncthreads();
            
            CLK(4);

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
                        for (int eid = node_r[v].start + lid, eid_end = node_r[v].start + node_r[v].length; eid < eid_end; eid += WARP_SIZE) {
                            int u = edge_r[eid];
                            int l = u2L[u];
                            if (l < *L_lp_nxt)
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
                            if (num_N_v[j] != num_L_nxt && num_N_v[j] > 0/* && node_r[v].length >= node_r[*x_cur].length*//* && (node_r[v].length > node_r[*x_cur].length || (node_r[v].length == node_r[*x_cur].length && v > *x_cur))*/) {
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
                    printf("\33[%d;%dH%*d\n", blockIdx.x / WORDS_1ROW + 9, (blockIdx.x % WORDS_1ROW) * WORD_WIDTH + 1, WORD_WIDTH, ++num_maximal_bicliques);
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

                Q_rm[*Q_lp_cur] = INF;

                // Q ← Q ∪ {x};
                int q = v2Q[*x_cur];

                // swap Q
                int Q_tmp = Q[*Q_lp_cur];
                Q[*Q_lp_cur] = *x_cur;
                Q[q] = Q_tmp;
                // maintain v2Q
                v2Q[*x_cur] = (*Q_lp_cur)++;;
                v2Q[Q_tmp] = q;

                

                //// printf("\n往 右 安安");

            }

        }

        else // lvl >= 1
        // while P ≠ ∅ do
        while (*P_lp_cur != 0) {
            
            __syncthreads();

            CLK(0);

            // 5/8 revised start //
            
            // find v in P to minimize num_L_nxt
            if (!threadIdx.x)
                num_L_nxt = INF;

            __syncthreads();

            // foreach v ∈ P do
            for (int i = *P_lp_cur; i-- > 0; ) {

                int v = P[i];

                if (!threadIdx.x)
                    num_N_v[0] = 0; // |N[v]|

                __syncthreads();

                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node_r[v].start + threadIdx.x, eid_end = node_r[v].start + node_r[v].length; eid < eid_end; eid += blockDim.x) {
                    int u = edge_r[eid];
                    int l = u2L[u];
                    if (l < *L_lp_cur && atomicAdd(&(num_N_v[0]), 1) == num_L_nxt)
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
                
                // |L'|
                *L_lp_nxt = 0;

                // scan from L
                num_N_L = 0;
            }
            
            __syncthreads();

            CLK(1);

            // for (int l = threadIdx.x; l < *NUM_L; l += blockDim.x)
            //     L[l] = L[l] > lvl ? lvl : L[l];

            __syncthreads();

            CLK(2);
            
            // L' <--- {u ∈ L | (u, x) ∈ E(G)};
            for (int eid = node_r[*x_cur].start + threadIdx.x, eid_end = node_r[*x_cur].start + node_r[*x_cur].length; eid < eid_end; eid += blockDim.x) {
                int u = edge_r[eid];
                int l = u2L[u];
                if (l < *L_lp_cur)
                    L_buf[atomicAdd(&num_L_nxt, 1)] = u;
            }

            __syncthreads();

            if (!threadIdx.x)
                for (int i = 0; i < num_L_nxt; i++) {
                    int u = L_buf[i];
                    int l = u2L[u];
                    // swap(L[(*L_lp_nxt)++], L[l]);
                    int L_tmp = L[*L_lp_nxt];
                    L[(*L_lp_nxt)++] = L[l];
                    L[l] = L_tmp;
                    // swap(u2L[L[l]], u2L[u]);
                    int u2L_tmp = u2L[L[l]];
                    u2L[L[l]] = u2L[u];
                    u2L[u] = u2L_tmp;
                }

            __syncthreads();

            CLK(3);

            // foreach u ∈ L'
            for (int i = wid; i < *L_lp_nxt; i += num_warps) {

                int u = L[i];

                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node_l[u].start + lid, eid_end = node_l[u].start + node_l[u].length; eid < eid_end; eid += WARP_SIZE) {
                    int v = edge_l[eid];
                    int q = v2Q[v];
                    if (q < *Q_lp_cur && atomicAdd(&(num_N_u[v]), 1) == 0)
                        L_buf[atomicAdd(&num_N_L, 1)] = v;
                }
            }

            __syncthreads();

            for (int i = threadIdx.x; i < num_N_L; i += blockDim.x) {
                int v = L_buf[i];
                if (num_N_u[v] == num_L_nxt)
                    is_maximal = false;
                num_N_u[v] = 0;
            }

            // foreach v ∈ Q
                    // WARNING // *Q_lp_cur // WARNING //
            // for (int i = wid; i < *Q_lp_nxt; i += num_warps) {

            //     if (Q_rm[i] < lvl) {
            //         __syncwarp();
            //         continue;
            //     }

            //     int v = Q[i];

            //     if (!lid)
            //         num_N_v[wid] = 0; // |N[v]|

            //     __syncwarp();

            //     // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
            //     for (int eid = node_r[v].start + lid, eid_end = node_r[v].start + node_r[v].length; eid < eid_end; eid += WARP_SIZE) {
            //         int u = edge_r[eid];
            //         int l = u2L[u];
            //         if (l < *L_lp_nxt)
            //             atomicAdd(&(num_N_v[wid]), 1);
            //     }

            //     __syncwarp();

            //     if (!lid) {
            //         Q_rm[i] = INF;

            //         // if |N[v]| = |L'| then
            //         if (num_N_v[wid] == num_L_nxt) {
            //             is_maximal = false;
            //             *Q_lp_nxt = 0;
            //         }

            //         // else if |N[v]| > 0 then
            //         else if (num_N_v[wid] == 0)
            //             // Q' ← Q' ∪ {v};
            //             // swap(Q[(*Q_ls_nxt)++], Q[i]);
            //             Q_rm[i] = lvl;
            //     }

            //     __syncwarp();
                
            // }

            __syncthreads();
            
            CLK(4);

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
                        for (int eid = node_r[v].start + lid, eid_end = node_r[v].start + node_r[v].length; eid < eid_end; eid += WARP_SIZE) {
                            int u = edge_r[eid];
                            int l = u2L[u];
                            if (l < *L_lp_nxt)
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
                    printf("\33[%d;%dH%*d\n", blockIdx.x / WORDS_1ROW + 9, (blockIdx.x % WORDS_1ROW) * WORD_WIDTH + 1, WORD_WIDTH, ++num_maximal_bicliques);
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
                
                Q_rm[*Q_lp_cur] = INF;

                // Q ← Q ∪ {x};
                int q = v2Q[*x_cur];

                // swap Q
                int Q_tmp = Q[*Q_lp_cur];
                Q[*Q_lp_cur] = *x_cur;
                Q[q] = Q_tmp;
                // maintain v2Q
                v2Q[*x_cur] = (*Q_lp_cur)++;;
                v2Q[Q_tmp] = q;


                
                
                //// printf("\n往 右 安安");

            }

        }
        
        __syncthreads();

        if (!threadIdx.x) {

            if (!is_recursive) {
                if (lvl--) {
                    Q_rm[Q_lp[lvl]] = INF;

                    int q = v2Q[x[lvl]];

                    // swap Q
                    int Q_tmp = Q[Q_lp[lvl]];
                    Q[Q_lp[lvl]] = Q[q];
                    Q[q] = Q_tmp;
                    // maintain v2Q
                    v2Q[x[lvl]] = Q_lp[lvl]++;
                    v2Q[Q_tmp] = q;

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
        for (int num = total_bic >> 7; num >>= 1; g_clk_scale += 2) ;
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
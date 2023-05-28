__device__ int P_ptr;

__global__ void CUDA_MBE_82(int *NUM_L, int *NUM_R, int *NUM_EDGES,
                            Node *node_l, int *edge_l, Node *node_r, int *edge_r,
                            int *g_u2L, int *g_v2P, int *g_v2Q, int *g_L, int *g_R, int *g_P, int *g_Q,
                            int *g_x, int *g_L_lp, int *g_R_lp, int *g_P_lp, int *g_Q_lp,
                            int *g_L_buf, int *g_num_N_u, int *g_pre_min,
                            int *ori_P, int *g_ori_P1, int *g_ori_Q1, int *g_ori_L1,
                            int *g_P_ptr1, int *g_fix_P_ptr1, int *g_fix_Q_ptr1,
                            int *num_mb, long long *time_section) {

    __shared__ int *u2L, *v2P, *v2Q;
    __shared__ int *L, *R, *P, *Q, *x;
    __shared__ int *L_lp, *R_lp, *P_lp, *Q_lp;
    __shared__ int *L_buf;
    __shared__ int *num_N_u;
    __shared__ int *pre_min;
    __shared__ int *ori_P1, *ori_Q1, *ori_L1, *ori_R1;
    __shared__ int *P_ptr1, *fix_P_ptr1, *fix_Q_ptr1;
    grid_group grid = this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = threadIdx.x >> LOG_WARP_SIZE;
    int lid = threadIdx.x % WARP_SIZE;
    __shared__ int num_total_thds;
    __shared__ int num_warps;
    __shared__ int num_maximal_bicliques;
    __shared__ int lvl;
    __shared__ int *x_cur;
    __shared__ int *L_lp_cur, *L_lp_nxt;
    __shared__ int *R_lp_cur, *R_lp_nxt;
    __shared__ int *P_lp_cur, *P_lp_nxt;
    __shared__ int *Q_lp_cur, *Q_lp_nxt;
    __shared__ int *pre_min_cur;
    __shared__ bool is_recursive, is_up;
    __shared__ bool is_maximal, is_pause;
    __shared__ int num_L_nxt, num_N_v[NUM_THDS >> LOG_WARP_SIZE], num_N_L;
    __shared__ int i_min[NUM_THDS >> LOG_WARP_SIZE];
    __shared__ int P_lp_cur_before, P_ptr0;

#ifdef DEBUG
    __shared__ long long clk[NUM_CLK], clk_;
    if (!threadIdx.x) {
        clk_ = clock();
        for (int i = 0; i < NUM_CLK; i++)
            clk[i] = 0;
    }
#endif /* DEBUG */

    for (int i = *NUM_R - tid - 1; i >= 0; i -= gridDim.x * blockDim.x)
        if (node_r[ori_P[i]].length) {
            if (i == *NUM_R - 1 || !node_r[ori_P[i + 1]].length)
                P_ptr = i;
            break;
        }
    
    grid.sync();

    if (!threadIdx.x) {
        u2L        = g_u2L        + blockIdx.x * (*NUM_L);
        v2P        = g_v2P        + blockIdx.x * (*NUM_R);
        v2Q        = g_v2Q        + blockIdx.x * (*NUM_R);
        L          = g_L          + blockIdx.x * (*NUM_L);
        R          = g_R          + blockIdx.x * (*NUM_R);
        P          = g_P          + blockIdx.x * (*NUM_R);
        Q          = g_Q          + blockIdx.x * (*NUM_R);
        x          = g_x          + blockIdx.x * (*NUM_R);
        L_lp       = g_L_lp       + blockIdx.x * (*NUM_R);
        R_lp       = g_R_lp       + blockIdx.x * (*NUM_R);
        P_lp       = g_P_lp       + blockIdx.x * (*NUM_R);
        Q_lp       = g_Q_lp       + blockIdx.x * (*NUM_R);
        L_buf      = g_L_buf      + blockIdx.x * (*NUM_L);
        num_N_u    = g_num_N_u    + blockIdx.x * (*NUM_R);
        pre_min    = g_pre_min    + blockIdx.x * (*NUM_R);
        ori_P1     = g_ori_P1     + blockIdx.x * (*NUM_R);
        ori_Q1     = g_ori_Q1     + blockIdx.x * (*NUM_R);
        ori_L1     = g_ori_L1     + blockIdx.x * (*NUM_L);
        P_ptr1     = g_P_ptr1     + blockIdx.x;
        fix_P_ptr1 = g_fix_P_ptr1 + blockIdx.x;
        fix_Q_ptr1 = g_fix_Q_ptr1 + blockIdx.x;
        num_total_thds = gridDim.x * blockDim.x;
        num_warps = blockDim.x >> LOG_WARP_SIZE;
        num_maximal_bicliques = 0;
        lvl = 0;
        P_ptr0 = blockIdx.x;
        // P_lp[0] = *NUM_R + blockIdx.x;
        P_lp[0] = *NUM_R;
        is_pause = false;
        is_up = true;
    }

    __syncthreads();

    while (lvl >= 0) {

        if (!threadIdx.x) {
            x_cur    = &(   x[lvl]);
            L_lp_cur = &(L_lp[lvl]); L_lp_nxt = &(L_lp[lvl+1]);
            R_lp_cur = &(R_lp[lvl]); R_lp_nxt = &(R_lp[lvl+1]);
            P_lp_cur = &(P_lp[lvl]); P_lp_nxt = &(P_lp[lvl+1]);
            Q_lp_cur = &(Q_lp[lvl]); Q_lp_nxt = &(Q_lp[lvl+1]);
            pre_min_cur = &(pre_min[lvl]);
            is_recursive = false;
        }
        
        __syncthreads();

        if (lvl == 0)
        // while P ≠ ∅ do
        while (true) {
            
            __syncthreads();

            CLK(0);

            // atomically get a new 1-level sub-tree

            if (!threadIdx.x) {
                P_lp_cur_before = *P_lp_cur;
                *P_lp_cur = atomicAdd(&P_ptr, -1);
                *P_lp_cur = *P_lp_cur >= 0 ? *P_lp_cur : -1;
            }
            
            __syncthreads();

            if (*P_lp_cur == -1) break;

            if (!threadIdx.x) {
                for (int i = *P_lp_cur + 1; i < P_lp_cur_before; i++) {

                    int v = ori_P[i];
                    int q = v2Q[v];

                    // swap Q
                    int Q_tmp = Q[*Q_lp_cur];
                    Q[*Q_lp_cur] = v;
                    Q[q] = Q_tmp;
                    // maintain v2Q
                    v2Q[v] = (*Q_lp_cur)++;
                    v2Q[Q_tmp] = q;

                }
            }

            // reset P to ordered
            for (int i = threadIdx.x; i < P_lp_cur_before; i += blockDim.x)
                v2P[P[i] = ori_P[i]] = i;
            
            __syncthreads();

            CLK(8);

            if (!threadIdx.x) {

                // Select x from P;
                // P <--- P \ {x before P_ptr and x_cur};
                *x_cur = ori_P[*P_lp_cur];
                
                // R' <--- R ∪ {x};
                *R_lp_nxt = *R_lp_cur;
                R[(*R_lp_nxt)++] = *x_cur;

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

            __syncthreads();
            
            CLK(4);

            // if is_maximal = TRUE then
            if (is_maximal == true) {

                if (!threadIdx.x)
                    num_N_L = 0;

                __syncthreads();

                // foreach u ∈ L'
                for (int i = wid; i < *L_lp_nxt; i += num_warps) {

                    int u = L[i];

                    // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                    for (int eid = node_l[u].start + lid, eid_end = node_l[u].start + node_l[u].length; eid < eid_end; eid += WARP_SIZE) {
                        int v = edge_l[eid];
                        int p = v2P[v];
                        if (p < *P_lp_cur && atomicAdd(&(num_N_u[v]), 1) == 0)
                            L_buf[atomicAdd(&num_N_L, 1)] = v;
                    }

                }

                __syncthreads();

                for (int i = threadIdx.x; i < num_N_L; i += blockDim.x) {
                    int v = L_buf[i];
                    if (num_N_u[v] == num_L_nxt)
                        R[atomicAdd(R_lp_nxt, 1)] = v;
                }

                __syncthreads();

                if (!threadIdx.x)
                    for (int i = 0; i < num_N_L; i++) {
                        int v = L_buf[i];
                        int p = v2P[v];
                        if (num_N_u[v] != num_L_nxt) {
                            // P' ← P' ∪ {v};
                            int P_tmp = P[*P_lp_nxt];
                            P[*P_lp_nxt] = v;
                            P[p] = P_tmp;
                            // maintain v2P
                            v2P[v] = (*P_lp_nxt)++;
                            v2P[P_tmp] = p;
                        }
                        num_N_u[v] = 0;
                    }

                __syncthreads();
            
                CLK(5);

                if (!threadIdx.x) {
#ifdef DEBUG

                    // if (++num_maximal_bicliques > 0)
                    printf("\33[%d;%dH%*d\n", blockIdx.x / WORDS_1ROW + 9, (blockIdx.x % WORDS_1ROW) * WORD_WIDTH + 1, WORD_WIDTH, ++num_maximal_bicliques);
#else  /* DEBUG */
                    ++num_maximal_bicliques;
#endif /* DEBUG */
                    // if P' ≠ ∅ then
                    if (*P_lp_nxt != 0) {
                        // biclique_find(G, L', R', P', Q');
                        is_recursive = true;
                    }

                }

                __syncthreads();
            
                CLK(6);

                if (is_recursive)
                    break;

            }

            if (!threadIdx.x) {

                // tree traversal to the right one

                // Q ← Q ∪ {x};
                int q = v2Q[*x_cur];

                // swap Q
                int Q_tmp = Q[*Q_lp_cur];
                Q[*Q_lp_cur] = *x_cur;
                Q[q] = Q_tmp;
                // maintain v2Q
                v2Q[*x_cur] = (*Q_lp_cur)++;
                v2Q[Q_tmp] = q;

            }

        }

        else // lvl >= 1
        // while P ≠ ∅ do
        while (*P_lp_cur != 0 && !is_pause) {
            
            __syncthreads();

            CLK(0);
            
            // find v in P to minimize num_L_nxt
            if (!threadIdx.x)
                num_L_nxt = INF;

            __syncthreads();

            // foreach v ∈ P do
            for (int i = *P_lp_cur; i-- > 0 && num_L_nxt != *pre_min_cur; ) {

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
                    i_min[0] = i;
                    num_L_nxt = num_N_v[0];
                }

                __syncthreads();

            }

            if (!threadIdx.x) {
                *pre_min_cur = num_L_nxt;
                // swap choosed *x_cur to P[*P_lp_cur - 1]
                int idx = i_min[0];
                int P_tmp = P[*P_lp_cur - 1];
                P[*P_lp_cur - 1] = P[idx];
                P[idx] = P_tmp;
                // maintain v2P
                v2P[P[*P_lp_cur - 1]] = *P_lp_cur - 1;
                v2P[P_tmp] = idx;
            }
                
            __syncthreads();

            CLK(7);

            if (!threadIdx.x) {

                // Select x from P;
                // P <--- P \ {x};
                *x_cur = P[--(*P_lp_cur)];
                
                // R' <--- R ∪ {x};
                *R_lp_nxt = *R_lp_cur;
                R[(*R_lp_nxt)++] = *x_cur;

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

            __syncthreads();

            CLK(4);

            // if is_maximal = TRUE then
            if (is_maximal == true) {
                
                if (!threadIdx.x)
                    num_N_L = 0;

                __syncthreads();

                // foreach u ∈ L'
                for (int i = wid; i < *L_lp_nxt; i += num_warps) {

                    int u = L[i];

                    // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                    for (int eid = node_l[u].start + lid, eid_end = node_l[u].start + node_l[u].length; eid < eid_end; eid += WARP_SIZE) {
                        int v = edge_l[eid];
                        int p = v2P[v];
                        if (p < *P_lp_cur && atomicAdd(&(num_N_u[v]), 1) == 0)
                            L_buf[atomicAdd(&num_N_L, 1)] = v;
                    }

                }

                __syncthreads();

                for (int i = threadIdx.x; i < num_N_L; i += blockDim.x) {
                    int v = L_buf[i];
                    if (num_N_u[v] == num_L_nxt)
                        R[atomicAdd(R_lp_nxt, 1)] = v;
                }

                __syncthreads();

                if (!threadIdx.x)
                    for (int i = 0; i < num_N_L; i++) {
                        int v = L_buf[i];
                        int p = v2P[v];
                        if (num_N_u[v] != num_L_nxt) {
                            // P' ← P' ∪ {v};
                            int P_tmp = P[*P_lp_nxt];
                            P[*P_lp_nxt] = v;
                            P[p] = P_tmp;
                            // maintain v2P
                            v2P[v] = (*P_lp_nxt)++;
                            v2P[P_tmp] = p;
                        }
                        num_N_u[v] = 0;
                    }

                __syncthreads();
            
                CLK(5);

                if (!threadIdx.x) {
#ifdef DEBUG

                    // if (++num_maximal_bicliques > 0)
                    printf("\33[%d;%dH%*d\n", blockIdx.x / WORDS_1ROW + 9, (blockIdx.x % WORDS_1ROW) * WORD_WIDTH + 1, WORD_WIDTH, ++num_maximal_bicliques);
#else  /* DEBUG */
                    ++num_maximal_bicliques;
#endif /* DEBUG */

                    // if P' ≠ ∅ then
                    if (*P_lp_nxt != 0) {
                        // biclique_find(G, L', R', P', Q');
                        is_recursive = true;
                    }

                }

                __syncthreads();
            
                CLK(6);

                if (is_recursive)
                    break;

            }

            if (!threadIdx.x) {

                // tree traversal to the right one

                // Q ← Q ∪ {x};
                int q = v2Q[*x_cur];

                // swap Q
                int Q_tmp = Q[*Q_lp_cur];
                Q[*Q_lp_cur] = *x_cur;
                Q[q] = Q_tmp;
                // maintain v2Q
                v2Q[*x_cur] = (*Q_lp_cur)++;
                v2Q[Q_tmp] = q;

                // up or right
                is_up = *P_lp_cur == 0;

                // pause and goto gsync();
                if (P_ptr < 0 && lvl - is_up == 1)
                    is_pause = true;

            }
            __syncthreads();

        }
        
        __syncthreads();

        if (!threadIdx.x) {

            // tree traversal to the deeper one
            if (is_recursive)
                lvl++;
            // tree traversal to the up one and right one
            else if (is_up && lvl--) {

                int q = v2Q[x[lvl]];

                // swap Q
                int Q_tmp = Q[Q_lp[lvl]];
                Q[Q_lp[lvl]] = Q[q];
                Q[q] = Q_tmp;
                // maintain v2Q
                v2Q[x[lvl]] = Q_lp[lvl]++;
                v2Q[Q_tmp] = q;
            }
        
            // pause and goto gsync();
            if (is_pause) {
                *fix_P_ptr1 = P_lp[1];
                *P_ptr1 = P_lp[1] - 1;
                *fix_Q_ptr1 = Q_lp[1];
            }
        }
        if (is_pause)
            break;

        __syncthreads();

    }
    
    if (!threadIdx.x && !is_pause) {
        *fix_P_ptr1 = P_lp[1];
        *P_ptr1 = -1;
        *fix_Q_ptr1 = Q_lp[1];
    }

    grid.sync();

    CLK(0);

    for (int i = threadIdx.x; i < *NUM_R; i += blockDim.x)
        ori_P1[i] = P[i];

    int *deg = L_buf;

    // foreach v ∈ P do
    for (int i = P_lp[1]; i-- > 0; ) {

        int v = P[i];

        if (!threadIdx.x)
            num_N_v[0] = 0; // |N[v]|

        __syncthreads();

        // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
        for (int eid = node_r[v].start + threadIdx.x, eid_end = node_r[v].start + node_r[v].length; eid < eid_end; eid += blockDim.x) {
            int u = edge_r[eid];
            int l = u2L[u];
            if (l < L_lp[1])
                atomicAdd(&(num_N_v[0]), 1);
        }

        __syncthreads();

        deg[P[i]] = num_N_v[0];

        __syncthreads();

    }

    grid.sync();

    for (int i = 0; i < gridDim.x; i++) {
        PARALLEL_BUBBLE_SORT(&(g_ori_P1[i*(*NUM_R)]), &(g_P_lp[i*(*NUM_R)+1]), &(g_L_buf[i*(*NUM_L)]));
        grid.sync();
    }

#ifdef DEBUG

    for (int i = 1 + threadIdx.x; i < P_lp[1]; i += blockDim.x)
        if (deg[ori_P1[i]] > deg[ori_P1[i-1]])
            printf("sorting failed\n");

#endif /* DEBUG */

    for (int i = threadIdx.x; i < *NUM_R; i += blockDim.x) {
        ori_L1[i] = L[i];
        ori_Q1[i] = Q[i];
    }

    for (int i = *NUM_R + threadIdx.x; i < *NUM_L; i += blockDim.x)
        ori_L1[i] = L[i];

    grid.sync();

    CLK(9);


    while (true) {    

        while (lvl >= 1) {

            if (!threadIdx.x) {
                x_cur    = &(   x[lvl]);
                L_lp_cur = &(L_lp[lvl]); L_lp_nxt = &(L_lp[lvl+1]);
                R_lp_cur = &(R_lp[lvl]); R_lp_nxt = &(R_lp[lvl+1]);
                P_lp_cur = &(P_lp[lvl]); P_lp_nxt = &(P_lp[lvl+1]);
                Q_lp_cur = &(Q_lp[lvl]); Q_lp_nxt = &(Q_lp[lvl+1]);
                pre_min_cur = &(pre_min[lvl]);
                is_recursive = false;
            }
            
            __syncthreads();

            if (lvl == 1)
            // while P ≠ ∅ do
            while (true) {
                
                __syncthreads();

                CLK(0);

                // atomically get a new 1-level sub-tree

                if (!threadIdx.x) {
                    P_lp_cur_before = *P_lp_cur;
                    *P_lp_cur = atomicAdd(P_ptr1, -1);
                    *P_lp_cur = *P_lp_cur >= 0 ? *P_lp_cur : -1;
                }
                
                __syncthreads();

                if (*P_lp_cur == -1) break;

                if (!threadIdx.x) {
                    for (int i = *P_lp_cur + 1; i < P_lp_cur_before; i++) {

                        int v = ori_P1[i];
                        int q = v2Q[v];

                        // swap Q
                        int Q_tmp = Q[*Q_lp_cur];
                        Q[*Q_lp_cur] = v;
                        Q[q] = Q_tmp;
                        // maintain v2Q
                        v2Q[v] = (*Q_lp_cur)++;
                        v2Q[Q_tmp] = q;

                    }
                }

                // reset P to ordered
                for (int i = threadIdx.x; i < P_lp_cur_before; i += blockDim.x)
                    v2P[P[i] = ori_P1[i]] = i;
                
                __syncthreads();

                CLK(8);

                if (!threadIdx.x) {

                    // Select x from P;
                    // P <--- P \ {x before P_ptr and x_cur};
                    *x_cur = ori_P1[*P_lp_cur];
                    
                    // R' <--- R ∪ {x};
                    *R_lp_nxt = *R_lp_cur;
                    R[(*R_lp_nxt)++] = *x_cur;

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

                __syncthreads();
                
                CLK(4);

                // if is_maximal = TRUE then
                if (is_maximal == true) {

                    if (!threadIdx.x)
                        num_N_L = 0;

                    __syncthreads();

                    // foreach u ∈ L'
                    for (int i = wid; i < *L_lp_nxt; i += num_warps) {

                        int u = L[i];

                        // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                        for (int eid = node_l[u].start + lid, eid_end = node_l[u].start + node_l[u].length; eid < eid_end; eid += WARP_SIZE) {
                            int v = edge_l[eid];
                            int p = v2P[v];
                            if (p < *P_lp_cur && atomicAdd(&(num_N_u[v]), 1) == 0)
                                L_buf[atomicAdd(&num_N_L, 1)] = v;
                        }

                    }

                    __syncthreads();

                    for (int i = threadIdx.x; i < num_N_L; i += blockDim.x) {
                        int v = L_buf[i];
                        if (num_N_u[v] == num_L_nxt)
                            R[atomicAdd(R_lp_nxt, 1)] = v;
                    }

                    __syncthreads();

                    if (!threadIdx.x)
                        for (int i = 0; i < num_N_L; i++) {
                            int v = L_buf[i];
                            int p = v2P[v];
                            if (num_N_u[v] != num_L_nxt) {
                                // P' ← P' ∪ {v};
                                int P_tmp = P[*P_lp_nxt];
                                P[*P_lp_nxt] = v;
                                P[p] = P_tmp;
                                // maintain v2P
                                v2P[v] = (*P_lp_nxt)++;
                                v2P[P_tmp] = p;
                            }
                            num_N_u[v] = 0;
                        }

                    __syncthreads();
                
                    CLK(5);

                    if (!threadIdx.x) {
#ifdef DEBUG
                        // if (++num_maximal_bicliques > 0)
                        printf("\33[%d;%dH%*d\n", blockIdx.x / WORDS_1ROW + 9, (blockIdx.x % WORDS_1ROW) * WORD_WIDTH + 1, WORD_WIDTH, ++num_maximal_bicliques);
#else  /* DEBUG */
                        ++num_maximal_bicliques;
#endif /* DEBUG */
                        // if P' ≠ ∅ then
                        if (*P_lp_nxt != 0) {
                            // biclique_find(G, L', R', P', Q');
                            is_recursive = true;
                        }

                    }

                    __syncthreads();
                
                    CLK(6);

                    if (is_recursive)
                        break;

                }

                if (!threadIdx.x) {

                    // tree traversal to the right one

                    // Q ← Q ∪ {x};
                    int q = v2Q[*x_cur];

                    // swap Q
                    int Q_tmp = Q[*Q_lp_cur];
                    Q[*Q_lp_cur] = *x_cur;
                    Q[q] = Q_tmp;
                    // maintain v2Q
                    v2Q[*x_cur] = (*Q_lp_cur)++;
                    v2Q[Q_tmp] = q;

                }

            }

            else // lvl >= 2
            // while P ≠ ∅ do
            while (*P_lp_cur != 0) {
                
                __syncthreads();

                CLK(0);
                
                // find v in P to minimize num_L_nxt
                if (!threadIdx.x)
                    num_L_nxt = INF;

                __syncthreads();

                // foreach v ∈ P do
                for (int i = *P_lp_cur; i-- > 0 && num_L_nxt != *pre_min_cur; ) {

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
                        i_min[0] = i;
                        num_L_nxt = num_N_v[0];
                    }

                    __syncthreads();

                }

                if (!threadIdx.x) {
                    *pre_min_cur = num_L_nxt;
                    // swap choosed *x_cur to P[*P_lp_cur - 1]
                    int idx = i_min[0];
                    int P_tmp = P[*P_lp_cur - 1];
                    P[*P_lp_cur - 1] = P[idx];
                    P[idx] = P_tmp;
                    // maintain v2P
                    v2P[P[*P_lp_cur - 1]] = *P_lp_cur - 1;
                    v2P[P_tmp] = idx;
                }
                    
                __syncthreads();

                CLK(7);

                if (!threadIdx.x) {

                    // Select x from P;
                    // P <--- P \ {x};
                    *x_cur = P[--(*P_lp_cur)];
                    
                    // R' <--- R ∪ {x};
                    *R_lp_nxt = *R_lp_cur;
                    R[(*R_lp_nxt)++] = *x_cur;

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

                __syncthreads();

                CLK(4);

                // if is_maximal = TRUE then
                if (is_maximal == true) {
                    
                    if (!threadIdx.x)
                        num_N_L = 0;

                    __syncthreads();

                    // foreach u ∈ L'
                    for (int i = wid; i < *L_lp_nxt; i += num_warps) {

                        int u = L[i];

                        // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                        for (int eid = node_l[u].start + lid, eid_end = node_l[u].start + node_l[u].length; eid < eid_end; eid += WARP_SIZE) {
                            int v = edge_l[eid];
                            int p = v2P[v];
                            if (p < *P_lp_cur && atomicAdd(&(num_N_u[v]), 1) == 0)
                                L_buf[atomicAdd(&num_N_L, 1)] = v;
                        }

                    }

                    __syncthreads();

                    for (int i = threadIdx.x; i < num_N_L; i += blockDim.x) {
                        int v = L_buf[i];
                        if (num_N_u[v] == num_L_nxt)
                            R[atomicAdd(R_lp_nxt, 1)] = v;
                    }

                    __syncthreads();

                    if (!threadIdx.x)
                        for (int i = 0; i < num_N_L; i++) {
                            int v = L_buf[i];
                            int p = v2P[v];
                            if (num_N_u[v] != num_L_nxt) {
                                // P' ← P' ∪ {v};
                                int P_tmp = P[*P_lp_nxt];
                                P[*P_lp_nxt] = v;
                                P[p] = P_tmp;
                                // maintain v2P
                                v2P[v] = (*P_lp_nxt)++;
                                v2P[P_tmp] = p;
                            }
                            num_N_u[v] = 0;
                        }

                    __syncthreads();
                
                    CLK(5);

                    if (!threadIdx.x) {
#ifdef DEBUG
                        // if (++num_maximal_bicliques > 0)
                        printf("\33[%d;%dH%*d\n", blockIdx.x / WORDS_1ROW + 9, (blockIdx.x % WORDS_1ROW) * WORD_WIDTH + 1, WORD_WIDTH, ++num_maximal_bicliques);
#else  /* DEBUG */
                        ++num_maximal_bicliques;
#endif /* DEBUG */

                        // if P' ≠ ∅ then
                        if (*P_lp_nxt != 0) {
                            // biclique_find(G, L', R', P', Q');
                            is_recursive = true;
                        }

                    }

                    __syncthreads();
                
                    CLK(6);

                    if (is_recursive)
                        break;

                }

                if (!threadIdx.x) {

                    // tree traversal to the right one

                    // Q ← Q ∪ {x};
                    int q = v2Q[*x_cur];

                    // swap Q
                    int Q_tmp = Q[*Q_lp_cur];
                    Q[*Q_lp_cur] = *x_cur;
                    Q[q] = Q_tmp;
                    // maintain v2Q
                    v2Q[*x_cur] = (*Q_lp_cur)++;
                    v2Q[Q_tmp] = q;

                }
                __syncthreads();

            }
            
            __syncthreads();

            if (!threadIdx.x) {

                // tree traversal to the deeper one
                if (is_recursive)
                    lvl++;
                // tree traversal to the up one and right one
                else if (lvl--) {

                    int q = v2Q[x[lvl]];

                    // swap Q
                    int Q_tmp = Q[Q_lp[lvl]];
                    Q[Q_lp[lvl]] = Q[q];
                    Q[q] = Q_tmp;
                    // maintain v2Q
                    v2Q[x[lvl]] = Q_lp[lvl]++;
                    v2Q[Q_tmp] = q;
                }
            }

            __syncthreads();

        }


        __syncthreads();

        CLK(0);

        // enable work stealing
        if (!threadIdx.x)
            lvl = 1;
        
        __syncthreads();

        if (!threadIdx.x) do {
            ++P_ptr0 %= gridDim.x;
            if (P_ptr0 == blockIdx.x) {
                lvl = 0;
                break;
            }
        } while (g_P_ptr1[P_ptr0] < 0);

        __syncthreads();

        if (lvl <= 0) break;
        
        if (!threadIdx.x) {
            ori_P1     = g_ori_P1     + P_ptr0 * (*NUM_R);
            ori_Q1     = g_ori_Q1     + P_ptr0 * (*NUM_R);
            ori_L1     = g_ori_L1     + P_ptr0 * (*NUM_L);
            ori_R1     = g_R          + P_ptr0 * (*NUM_R);
            P_ptr1     = g_P_ptr1     + P_ptr0;
            fix_P_ptr1 = g_fix_P_ptr1 + P_ptr0;
            fix_Q_ptr1 = g_fix_Q_ptr1 + P_ptr0;

            P_lp[1] = *fix_P_ptr1;
            Q_lp[1] = *fix_Q_ptr1;
            L_lp[1] = g_L_lp[P_ptr0 * (*NUM_R) + 1];
            R_lp[1] = g_R_lp[P_ptr0 * (*NUM_R) + 1];
        }

        __syncthreads();
        
        for (int i = threadIdx.x; i < R_lp[1]; i += blockDim.x)
            R[i] = ori_R1[i];
        
        for (int i = threadIdx.x; i < *NUM_R; i += blockDim.x) {
            v2P[P[i] = ori_P1[i]] = i;
            v2Q[Q[i] = ori_Q1[i]] = i;
        }

        for (int i = threadIdx.x; i < *NUM_L; i += blockDim.x)
            u2L[L[i] = ori_L1[i]] = i;

        __syncthreads();

        CLK(2);
    }



    grid.sync();
    
    if (!threadIdx.x)
        atomicAdd(num_mb, num_maximal_bicliques);

    grid.sync();

#ifdef DEBUG
    
    for (int blockid = 0; blockid < gridDim.x; blockid++) {
        if (blockIdx.x == blockid)
            for (int i = threadIdx.x; i < NUM_CLK; i += blockDim.x)
                time_section[i] += clk[i];
        grid.sync();
    }
    
    grid.sync();

    if (!tid)
        printf("\33[6;1H");
    grid.sync();

#endif /* DEBUG */

}
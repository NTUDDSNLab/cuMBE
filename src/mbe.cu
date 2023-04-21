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
#define NUM_THDS 256
#define NUM_BLKS atoi(argv[2])
#define LOG_BLK_ID 1
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

void maximal_bic_enum_set(int *NUM_L, int *NUM_R, int *NUM_EDGES, Node *node, int *edge,
                          int *u2L, int *L, int *R, int *P, int *Q,
                          int *x, int *L_lp, int *R_lp, int *P_lp, int *Q_lp) {
    vector<Biclique> maximal_bicliques;
    int num_maximal_bicliques = 0;
    vector< unordered_set<int> > Q_set(*NUM_R);
    Q_set[0].clear();

    long long clk[10] = { 0 }, clk_ = clock();
    
    for (int lvl = 0; lvl >= 0; ) {

        // printf("lvl: %d\n", lvl);

        int *x_cur    = &(   x[lvl]);
        int *L_lp_cur = &(L_lp[lvl]), *L_lp_nxt = &(L_lp[lvl+1]);
        int *R_lp_cur = &(R_lp[lvl]), *R_lp_nxt = &(R_lp[lvl+1]);
        int *P_lp_cur = &(P_lp[lvl]), *P_lp_nxt = &(P_lp[lvl+1]);
        unordered_set<int> *Q_cur = &(Q_set[lvl]), *Q_nxt = &(Q_set[lvl+1]);
        bool is_recursive = false;

        // while P ≠ ∅ do
        while (*P_lp_cur != 0) {

            CLK_CPU(0);

            //string tab_level(lvl << 3, ' ');
            //printf("\n%sL:", tab_level.c_str());
            //for (int i = 0; i < *L_lp_cur; i++)
            //    printf(" %d", L[i]);
            //printf("\n%sR:", tab_level.c_str());
            //for (int i = 0; i < *R_lp_cur; i++)
            //    printf(" %d", R[i]);
            //printf("\n%sP:", tab_level.c_str());
            //for (int i = 0; i < *P_lp_cur; i++)
            //    printf(" %d", P[i]);
            //printf("\n%sQ:", tab_level.c_str());
            //for (int i = 0; i < *Q_lp_cur; i++)
            //    printf(" %d", Q[i]);

            // Select x from P;
            // P <--- P \ {x};
            *x_cur = P[--(*P_lp_cur)];
            //// printf("x: %d\n", *x_cur);
            
            // R' <--- R ∪ {x};
            *R_lp_nxt = *R_lp_cur;
            R[(*R_lp_nxt)++] = *x_cur;

            CLK_CPU(1);

            *L_lp_nxt = 0; // |L'|

            CLK_CPU(2);

            // L' <--- {u ∈ L | (u, x) ∈ E(G)};
            for (int eid = node[*x_cur].start, eid_end = eid + node[*x_cur].length; eid < eid_end; eid++) {
                int u = edge[eid];
                int l = u2L[u];
                if (l < *L_lp_cur) {
                    swap(L[(*L_lp_nxt)++], L[l]);
                    swap(u2L[L[l]], u2L[u]);
                }
            }

            CLK_CPU(3);

            //// printf("L':");
            //// for (int i = 0; i < *L_lp_nxt; i++)
            ////     printf(" %d", L[i]);
            //// printf("\n");
            
            // P' ← ∅; Q' ← ∅;
            *P_lp_nxt = 0; (*Q_nxt).clear();

            bool is_maximal = true;

            // foreach v ∈ Q
            for (const auto &v : *Q_cur) {

                int num_N_v = 0; // |N[v]|
                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node[v].start, eid_end = eid + node[v].length; eid < eid_end; eid++) {
                    int u = edge[eid];
                    int l = u2L[u];
                    if (l < *L_lp_nxt)
                        num_N_v++;
                }
                
                // if |N[v]| = |L'| then
                if (num_N_v == *L_lp_nxt) {
                    is_maximal = false;
                    break;
                }
                // else if |N[v]| > 0 then
                else if (num_N_v > 0)
                    // Q' ← Q' ∪ {v};
                    (*Q_nxt).insert(v);
                
            }

            CLK_CPU(4);

            //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

            // if is_maximal = TRUE then
            if (is_maximal == true) {

                // foreach v ∈ P do
                for (int i = 0; i < *P_lp_cur; i++) {
                    int v = P[i];

                    int num_N_v = 0; // |N[v]|
                    // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                    for (int eid = node[v].start, eid_end = eid + node[v].length; eid < eid_end; eid++) {
                        int u = edge[eid];
                        int l = u2L[u];
                        if (l < *L_lp_nxt)
                            num_N_v++;
                    }
                    
                    // if |N[v]| = |L'| then
                    if (num_N_v == *L_lp_nxt)
                        // R' ← R' ∪ {v};
                        R[(*R_lp_nxt)++] = v;
                    // else if |N[v]| > 0 then
                    else if (num_N_v > 0)
                        // P' ← P' ∪ {v};
                        swap(P[(*P_lp_nxt)++], P[i]);

                }

                CLK_CPU(5);
                
                //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

                // PRINT(L', R');
                //                printf("\n> Find maximal biclique (No. %d)", num_maximal_bicliques++);
                //                printf("\nL':");
                //                for (int i = 0; i < *L_lp_nxt; i++)
                //                    printf(" %d", L[i]);
                //                printf("\nR':");
                //                for (int i = 0; i < *R_lp_nxt; i++)
                //                    printf(" %d", R[i]);
                //                printf("\n");
                //
                // save maximal bicliques
                //// Biclique new_maximal_bicliques;
                //// for (int i = 0; i < *L_lp_nxt; i++)
                ////     new_maximal_bicliques.L.insert(L[i]);
                //// for (int i = 0; i < *R_lp_nxt; i++)
                ////     new_maximal_bicliques.R.insert(R[i]);
                //// maximal_bicliques.push_back(new_maximal_bicliques);
                if (++num_maximal_bicliques << 22 == 0)
                    printf("%d\n", num_maximal_bicliques);

                CLK_CPU(6);

                // if P' ≠ ∅ then
                if (*P_lp_nxt != 0) {
                    // biclique_find(G, L', R', P', Q');
                    //// printf("\n往 下 安安");
                    lvl++;
                    is_recursive = true;
                    break;
                }

            }
            else {
                //// printf("\n不安安");
            }

            // Q ← Q ∪ {x};
            (*Q_cur).insert(*x_cur);
            //// printf("\n往 右 安安");
        }

        if (!is_recursive) {
            if (lvl--)
                Q_set[lvl].insert(x[lvl]);
            //// printf("\n往 上 安安");
            //// printf("\n往 右 安安");
        }

    }

    printf("\nFind %d maximal bicliques.\n", num_maximal_bicliques);
    printf("time:");
    for (int i = 0; i < 10; i++) {
        // clk[i] >>= 21;
        printf(" %lld", clk[i]);
    }
    printf("\n");

    if (*NUM_R > 20 || *NUM_L > 20) return;

    string _ = "";
    printf("\33[2J\33[0;0H");

    printf("  ");
    for (int i = 0; i < *NUM_L; i++)
        printf(" %d", i / 10);
    printf("\n  ");
    for (int i = 0; i < *NUM_L; i++)
        printf(" %d", i % 10);
    printf("\n");
    for (int i = 0; i < *NUM_R; i++) {
        bool adj_vec[*NUM_L] = { false };
        for (int j = node[i].start, j_end = j + node[i].length; j < j_end; j++)
            adj_vec[edge[j]] = true;
        printf("%d%d", i / 10, i % 10);
        for (int j = 0; j < *NUM_L; j++)
            printf(" %c", adj_vec[j] ? '#' : '-');
        printf("\n");
    }

    for (int i = 0, i_end = maximal_bicliques.size(); i < i_end; i++) {
        printf("\33[7m");
        for (const auto &r: maximal_bicliques[i].R)
            for (const auto &l: maximal_bicliques[i].L)
                printf("\33[%d;%dH#\n", 3 + r, 4 + l * 2);
        printf("\33[0m\n\33[%d;0H\n", 3 + (*NUM_R));
        if      (_ == "auto") usleep(800000);
        else if (_ != "exit") cin >> _;
        for (const auto &r: maximal_bicliques[i].R)
            for (const auto &l: maximal_bicliques[i].L)
                printf("\33[%d;%dH#\n", 3 + r, 4 + l * 2);
    }
    printf("\33[%d;0H\n", 3 + (*NUM_R));
}

void maximal_bic_enum(int *NUM_L, int *NUM_R, int *NUM_EDGES, Node *node, int *edge,
                      int *u2L, int *L, int *R, int *P, int *Q,
                      int *x, int *L_lp, int *R_lp, int *P_lp, int *Q_lp) {
    vector<Biclique> maximal_bicliques;
    int num_maximal_bicliques = 0;

    long long clk[10] = { 0 }, clk_ = clock();
    
    for (int lvl = 0; lvl >= 0; ) {

        // printf("lvl: %d\n", lvl);

        int *x_cur    = &(   x[lvl]);
        int *L_lp_cur = &(L_lp[lvl]), *L_lp_nxt = &(L_lp[lvl+1]);
        int *R_lp_cur = &(R_lp[lvl]), *R_lp_nxt = &(R_lp[lvl+1]);
        int *P_lp_cur = &(P_lp[lvl]), *P_lp_nxt = &(P_lp[lvl+1]);
        int *Q_lp_cur = &(Q_lp[lvl]), *Q_lp_nxt = &(Q_lp[lvl+1]);
        bool is_recursive = false;

        // while P ≠ ∅ do
        while (*P_lp_cur != 0) {

            CLK_CPU(0);

            //string tab_level(lvl << 3, ' ');
            //printf("\n%sL:", tab_level.c_str());
            //for (int i = 0; i < *L_lp_cur; i++)
            //    printf(" %d", L[i]);
            //printf("\n%sR:", tab_level.c_str());
            //for (int i = 0; i < *R_lp_cur; i++)
            //    printf(" %d", R[i]);
            //printf("\n%sP:", tab_level.c_str());
            //for (int i = 0; i < *P_lp_cur; i++)
            //    printf(" %d", P[i]);
            //printf("\n%sQ:", tab_level.c_str());
            //for (int i = 0; i < *Q_lp_cur; i++)
            //    printf(" %d", Q[i]);

            // Select x from P;
            // P <--- P \ {x};
            *x_cur = P[--(*P_lp_cur)];
            //// printf("x: %d\n", *x_cur);
            
            // R' <--- R ∪ {x};
            *R_lp_nxt = *R_lp_cur;
            R[(*R_lp_nxt)++] = *x_cur;

            CLK_CPU(1);

            *L_lp_nxt = 0; // |L'|

            CLK_CPU(2);

            // L' <--- {u ∈ L | (u, x) ∈ E(G)};
            for (int eid = node[*x_cur].start, eid_end = eid + node[*x_cur].length; eid < eid_end; eid++) {
                int u = edge[eid];
                int l = u2L[u];
                if (l < *L_lp_cur) {
                    swap(L[(*L_lp_nxt)++], L[l]);
                    swap(u2L[L[l]], u2L[u]);
                }
            }

            CLK_CPU(3);

            //// printf("L':");
            //// for (int i = 0; i < *L_lp_nxt; i++)
            ////     printf(" %d", L[i]);
            //// printf("\n");
            
            // P' ← ∅; Q' ← ∅;
            *P_lp_nxt = 0; *Q_lp_nxt = *Q_lp_cur;

            bool is_maximal = true;

            // foreach v ∈ Q
            for (int i = 0; i < *Q_lp_cur; i++) {

                int v = Q[i];

                int num_N_v = 0; // |N[v]|
                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node[v].start, eid_end = eid + node[v].length; eid < eid_end; eid++) {
                    int u = edge[eid];
                    int l = u2L[u];
                    if (l < *L_lp_nxt)
                        num_N_v++;
                }
                
                // if |N[v]| = |L'| then
                if (num_N_v == *L_lp_nxt) {
                    is_maximal = false;
                    break;
                }
                // // else if |N[v]| > 0 then
                // else if (num_N_v == 0)
                //     // Q' ← Q' ∪ {v};
                //     swap(Q[(*Q_ls_nxt)++], Q[i]);
                
            }

            CLK_CPU(4);

            //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

            // if is_maximal = TRUE then
            if (is_maximal == true) {

                // foreach v ∈ P do
                for (int i = 0; i < *P_lp_cur; i++) {
                    int v = P[i];

                    int num_N_v = 0; // |N[v]|
                    // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                    for (int eid = node[v].start, eid_end = eid + node[v].length; eid < eid_end; eid++) {
                        int u = edge[eid];
                        int l = u2L[u];
                        if (l < *L_lp_nxt)
                            num_N_v++;
                    }
                    
                    // if |N[v]| = |L'| then
                    if (num_N_v == *L_lp_nxt)
                        // R' ← R' ∪ {v};
                        R[(*R_lp_nxt)++] = v;
                    // else if |N[v]| > 0 then
                    else if (num_N_v > 0)
                        // P' ← P' ∪ {v};
                        swap(P[(*P_lp_nxt)++], P[i]);

                }

                CLK_CPU(5);
                
                //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

                // PRINT(L', R');
                //                printf("\n> Find maximal biclique (No. %d)", num_maximal_bicliques++);
                //                printf("\nL':");
                //                for (int i = 0; i < *L_lp_nxt; i++)
                //                    printf(" %d", L[i]);
                //                printf("\nR':");
                //                for (int i = 0; i < *R_lp_nxt; i++)
                //                    printf(" %d", R[i]);
                //                printf("\n");
                //
                // save maximal bicliques
                //// Biclique new_maximal_bicliques;
                //// for (int i = 0; i < *L_lp_nxt; i++)
                ////     new_maximal_bicliques.L.insert(L[i]);
                //// for (int i = 0; i < *R_lp_nxt; i++)
                ////     new_maximal_bicliques.R.insert(R[i]);
                //// maximal_bicliques.push_back(new_maximal_bicliques);
                if (++num_maximal_bicliques << 22 == 0)
                    printf("%d\n", num_maximal_bicliques);

                CLK_CPU(6);

                // if P' ≠ ∅ then
                if (*P_lp_nxt != 0) {
                    // biclique_find(G, L', R', P', Q');
                    //// printf("\n往 下 安安");
                    lvl++;
                    is_recursive = true;
                    break;
                }

            }
            else {
                //// printf("\n不安安");
            }

            // Q ← Q ∪ {x};
            Q[(*Q_lp_cur)++] = *x_cur;
            //// printf("\n往 右 安安");
        }

        if (!is_recursive) {
            lvl--;
            Q[Q_lp[lvl]++] = x[lvl];
            //// printf("\n往 上 安安");
            //// printf("\n往 右 安安");
        }

    }

    printf("\nFind %d maximal bicliques.\n", num_maximal_bicliques);
    printf("time:");
    for (int i = 0; i < 10; i++) {
        // clk[i] >>= 21;
        printf(" %lld", clk[i]);
    }
    printf("\n");

    if (*NUM_R > 20 || *NUM_L > 20) return;

    string _ = "";
    printf("\33[2J\33[0;0H");

    printf("  ");
    for (int i = 0; i < *NUM_L; i++)
        printf(" %d", i / 10);
    printf("\n  ");
    for (int i = 0; i < *NUM_L; i++)
        printf(" %d", i % 10);
    printf("\n");
    for (int i = 0; i < *NUM_R; i++) {
        bool adj_vec[*NUM_L] = { false };
        for (int j = node[i].start, j_end = j + node[i].length; j < j_end; j++)
            adj_vec[edge[j]] = true;
        printf("%d%d", i / 10, i % 10);
        for (int j = 0; j < *NUM_L; j++)
            printf(" %c", adj_vec[j] ? '#' : '-');
        printf("\n");
    }

    for (int i = 0, i_end = maximal_bicliques.size(); i < i_end; i++) {
        printf("\33[7m");
        for (const auto &r: maximal_bicliques[i].R)
            for (const auto &l: maximal_bicliques[i].L)
                printf("\33[%d;%dH#\n", 3 + r, 4 + l * 2);
        printf("\33[0m\n\33[%d;0H\n", 3 + (*NUM_R));
        if      (_ == "auto") usleep(800000);
        else if (_ != "exit") cin >> _;
        for (const auto &r: maximal_bicliques[i].R)
            for (const auto &l: maximal_bicliques[i].L)
                printf("\33[%d;%dH#\n", 3 + r, 4 + l * 2);
    }
    printf("\33[%d;0H\n", 3 + (*NUM_R));
}

__global__ void CUDA_MBE(int *NUM_L, int *NUM_R, int *NUM_EDGES, Node *node, int *edge,
                         int *u2L, int *L, int *R, int *P, int *Q,
                         int *x, int *L_lp, int *R_lp, int *P_lp, int *Q_lp) {

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
    __shared__ int num_L_nxt, num_N_v;
    // __shared__ int num_N_v[32];

    __shared__ long long clk[10], clk_;

    if (!threadIdx.x) {
        clk_ = clock();
        for (int i = 0; i < 10; i++)
            clk[i] = 0;
    }

    if (!threadIdx.x)
        lvl = 0;

    __syncthreads();

    for (; lvl >= 0; ) {

        // if (!threadIdx.x)
        //     printf("\nlvl: %d", lvl);

        x_cur    = &(   x[lvl]);
        L_lp_cur = &(L_lp[lvl]); L_lp_nxt = &(L_lp[lvl+1]);
        R_lp_cur = &(R_lp[lvl]); R_lp_nxt = &(R_lp[lvl+1]);
        P_lp_cur = &(P_lp[lvl]); P_lp_nxt = &(P_lp[lvl+1]);
        Q_lp_cur = &(Q_lp[lvl]); Q_lp_nxt = &(Q_lp[lvl+1]);

        if (!threadIdx.x)
            is_recursive = false;
        
        __syncthreads();

        // while P ≠ ∅ do
        while (*P_lp_cur != 0) {

            CLK(0);

            if (!threadIdx.x) {

                // printf("\n");
                // for (int i = 0; i < lvl; i++) printf("        ");
                // printf("L:");
                // for (int i = 0; i < *NUM_L; i++)
                //     printf(" %d", L[i]);
                // printf("\n");
                // for (int i = 0; i < lvl; i++) printf("        ");
                // printf("R:");
                // for (int i = 0; i < *R_lp_cur; i++)
                //     printf(" %d", R[i]);
                // printf("\n");
                // for (int i = 0; i < lvl; i++) printf("        ");
                // printf("P:");
                // for (int i = 0; i < *P_lp_cur; i++)
                //     printf(" %d", P[i]);
                // printf("\n");
                // for (int i = 0; i < lvl; i++) printf("        ");
                // printf("Q:");
                // for (int i = 0; i < *Q_lp_cur; i++)
                //     printf(" %d", Q[i]);

                // Select x from P;
                // P <--- P \ {x};
                *x_cur = P[--(*P_lp_cur)];
                //// printf("x: %d\n", *x_cur);
                
                // R' <--- R ∪ {x};
                *R_lp_nxt = *R_lp_cur;
                R[(*R_lp_nxt)++] = *x_cur;

                //// *L_lp_nxt = 0;
                num_L_nxt = 0;
            }
            
            __syncthreads();

            CLK(1);

            // |L'|
            for (int l = tid; l < *NUM_L; l += num_total_thds)
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

            //// printf("L':");
            //// for (int i = 0; i < *L_lp_nxt; i++)
            ////     printf(" %d", L[i]);
            //// printf("\n");
            
            if (!threadIdx.x) {
                
                // P' ← ∅; Q' ← ∅;
                *P_lp_nxt = 0; *Q_lp_nxt = *Q_lp_cur;
                is_maximal = true;

            }

            // foreach v ∈ Q
            for (int i = 0; i < *Q_lp_cur; i++) {

                int v = Q[i];

                if (!threadIdx.x)
                    num_N_v = 0; // |N[v]|

                __syncthreads();

                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node[v].start + threadIdx.x, eid_end = node[v].start + node[v].length; eid < eid_end; eid += blockDim.x) {
                    int l = edge[eid];
                    if (L[l] > lvl)
                        atomicAdd(&num_N_v, 1);
                }

                __syncthreads();
                
                // if |N[v]| = |L'| then
                if (num_N_v == num_L_nxt) {
                    is_maximal = false;
                    break;
                }
                // // else if |N[v]| > 0 then
                // else if (num_N_v == 0)
                //     // Q' ← Q' ∪ {v};
                //     swap(Q[(*Q_ls_nxt)++], Q[i]);

                __syncthreads();
                
            }
            
            CLK(4);

            __syncthreads();

            //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

            // if is_maximal = TRUE then
            if (is_maximal == true) {

                // foreach v ∈ P do
                for (int i = 0; i < *P_lp_cur; i++) {
                    int v = P[i];

                    if (!threadIdx.x)
                        num_N_v = 0; // |N[v]|

                    __syncthreads();

                    // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                    for (int eid = node[v].start + threadIdx.x, eid_end = node[v].start + node[v].length; eid < eid_end; eid += blockDim.x) {
                        int l = edge[eid];
                        if (L[l] > lvl)
                            atomicAdd(&num_N_v, 1);
                    }

                    __syncthreads();
                    
                    if (!threadIdx.x) {

                        // if |N[v]| = |L'| then
                        if (num_N_v == num_L_nxt)
                            // R' ← R' ∪ {v};
                            R[(*R_lp_nxt)++] = v;
                        // else if |N[v]| > 0 then
                        else if (num_N_v > 0) {
                            // P' ← P' ∪ {v};
                            int P_tmp = P[*P_lp_nxt];
                            P[(*P_lp_nxt)++] = P[i];
                            P[i] = P_tmp;
                        }

                    }

                    __syncthreads();

                }
            
                CLK(5);

                if (!threadIdx.x) {
                
                    //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

                    // PRINT(L', R');
                    // printf("\n> Find maximal biclique (No. %d)", num_maximal_bicliques++);
                    // printf("\nL':");
                    // for (int i = 0; i < *NUM_L; i++)
                    //     if (L[i] > lvl)
                    //         printf(" %d", i);
                    // printf("\nR':");
                    // for (int i = 0; i < *R_lp_nxt; i++)
                    //     printf(" %d", R[i]);
                    // printf("\n");

                    // save maximal bicliques
                    //// Biclique new_maximal_bicliques;
                    //// for (int i = 0; i < *L_lp_nxt; i++)
                    ////     new_maximal_bicliques.L.insert(L[i]);
                    //// for (int i = 0; i < *R_lp_nxt; i++)
                    ////     new_maximal_bicliques.R.insert(R[i]);
                    //// maximal_bicliques.push_back(new_maximal_bicliques);
                    if (++num_maximal_bicliques << 22 == 0)
                        printf("%d\n", num_maximal_bicliques);

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
                Q[(*Q_lp_cur)++] = *x_cur;
                //// printf("\n往 右 安安");

            }

        }
        
        __syncthreads();

        // printf("tid: %d, lvl: %d\n", tid, lvl);

        if (!threadIdx.x) {

            if (!is_recursive) {
                lvl--;
                Q[Q_lp[lvl]++] = x[lvl];
                //// printf("\n往 上 安安");
                //// printf("\n往 右 安安");
            }

        }

        __syncthreads();

    }

    if (!threadIdx.x) {
        printf("\nFind %d maximal bicliques.\n", num_maximal_bicliques);
        printf("time:");
        for (int i = 0; i < 10; i++) {
            // clk[i] >>= 21;
            printf(" %lld", clk[i]);
        }
        printf("\n");
    }
}

__device__ int g_clk[10];
__device__ int total_bic;
__device__ int P_ptr;
__global__ void CUDA_MBE_82(int *NUM_L, int *NUM_R, int *NUM_EDGES, Node *node, int *edge,
                            int *g_u2L, int *g_L, int *g_R, int *g_P, int *g_Q, int *g_Q_rm,
                            int *g_x, int *g_L_lp, int *g_R_lp, int *g_P_lp, int *g_Q_lp) {

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
    // int wid = threadIdx.x >> 5;
    // int lid = threadIdx.x & 0x1f;
    // int num_warps = blockDim.x >> 5;
    int num_maximal_bicliques = 0;
    __shared__ int lvl;
    __shared__ int *x_cur;
    __shared__ int *L_lp_cur, *L_lp_nxt;
    __shared__ int *R_lp_cur, *R_lp_nxt;
    __shared__ int *P_lp_cur, *P_lp_nxt;
    __shared__ int *Q_lp_cur, *Q_lp_nxt;
    __shared__ bool is_recursive;
    __shared__ bool is_maximal;
    __shared__ int num_L_nxt, num_N_v;

    __shared__ long long clk[10], clk_;
    if (!threadIdx.x) {
        clk_ = clock();
        for (int i = 0; i < 10; i++)
            clk[i] = 0;
        if (!blockIdx.x)
            for (int i = 0; i < 10; i++)
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
        //// printf("blk %d, u2L : %p\n", blockIdx.x, u2L );
        //// printf("blk %d, L   : %p\n", blockIdx.x, L   );
        //// printf("blk %d, R   : %p\n", blockIdx.x, R   );
        //// printf("blk %d, P   : %p\n", blockIdx.x, P   );
        //// printf("blk %d, Q   : %p\n", blockIdx.x, Q   );
        //// printf("blk %d, x   : %p\n", blockIdx.x, x   );
        //// printf("blk %d, L_lp: %p\n", blockIdx.x, L_lp);
        //// printf("blk %d, R_lp: %p\n", blockIdx.x, R_lp);
        //// printf("blk %d, P_lp: %p\n", blockIdx.x, P_lp);
        //// printf("blk %d, Q_lp: %p\n", blockIdx.x, Q_lp);
    }

    __syncthreads();

    for (; lvl >= 0; ) {

        // if (!threadIdx.x)
        //     printf("\nlvl: %d", lvl);

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

            //// if (!threadIdx.x && blockIdx.x == LOG_BLK_ID) {
            ////     printf("\nblock_%d subtree_%d has found %d maximal bicliques now", blockIdx.x, *P_lp_cur - gridDim.x, num_maximal_bicliques);
            ////     // num_maximal_bicliques = 0;
            //// }
            CLK(0);

            if (!threadIdx.x) {

                //// if (blockIdx.x == LOG_BLK_ID) {
                ////     printf("\n");
                ////     for (int i = 0; i < lvl; i++) printf("        ");
                ////     printf("L:");
                ////     for (int i = 0; i < *NUM_L; i++)
                ////         printf(" %d", L[i]);
                ////     printf("\n");
                ////     for (int i = 0; i < lvl; i++) printf("        ");
                ////     printf("R:");
                ////     for (int i = 0; i < *R_lp_cur; i++)
                ////         printf(" %d", R[i]);
                ////     printf("\n");
                ////     for (int i = 0; i < lvl; i++) printf("        ");
                ////     printf("P:");
                ////     for (int i = 0; i < *P_lp_cur; i++)
                ////         printf(" %d", P[i]);
                ////     printf("\n");
                ////     for (int i = 0; i < lvl; i++) printf("        ");
                ////     printf("Q:");
                ////     for (int i = 0; i < *Q_lp_cur; i++)
                ////         printf(" %d", Q[i]);
                //// }

                // atomically get a new 1-level sub-tree
                // Q <--- Q ∪ {x before P_ptr};
                // for (int i = *P_lp_cur, i_end = *P_lp_cur -= gridDim.x; --i > i_end; ) {
                //     if (i < *NUM_R) {
                //         Q_rm[*Q_lp_cur] = INF;
                //         Q[(*Q_lp_cur)++] = i;
                //     }
                // }
                for (int i = *P_lp_cur, i_end = *P_lp_cur = atomicAdd(&P_ptr, -1); --i > i_end; ) {
                    if (i >= 0) {
                        Q_rm[*Q_lp_cur] = INF;
                        Q[(*Q_lp_cur)++] = i;
                    }
                }

                // printf("blk %d, P_lp_cur: %d\n", blockIdx.x, *P_lp_cur);

                // reset P to ordered
                for (int i = 0; i < *NUM_R; i++)
                    P[i] = i;

                // Select x from P;
                // P <--- P \ {x before P_ptr and x_cur};
                *x_cur = *P_lp_cur;
                //// printf("x: %d\n", *x_cur);
                
                // R' <--- R ∪ {x};
                *R_lp_nxt = *R_lp_cur;
                R[(*R_lp_nxt)++] = *x_cur;

                //// *L_lp_nxt = 0;
                num_L_nxt = 0;
            }
            
            __syncthreads();

            CLK(1);

            if (*P_lp_cur < 0) break;

            // |L'|
            // for (int l = tid; l < *NUM_L; l += num_total_thds)
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

            //// printf("L':");
            //// for (int i = 0; i < *L_lp_nxt; i++)
            ////     printf(" %d", L[i]);
            //// printf("\n");
            
            if (!threadIdx.x) {
                
                // P' ← ∅; Q' ← ∅;
                *P_lp_nxt = 0; *Q_lp_nxt = *Q_lp_cur;
                is_maximal = true;

            }

            // foreach v ∈ Q
            for (int i = 0; i < *Q_lp_cur; i++) {

                if (Q_rm[i] < lvl) {
                    __syncthreads();
                    continue;
                }

                int v = Q[i];

                if (!threadIdx.x)
                    num_N_v = 0; // |N[v]|

                __syncthreads();

                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node[v].start + threadIdx.x, eid_end = node[v].start + node[v].length; eid < eid_end; eid += blockDim.x) {
                    int l = edge[eid];
                    if (L[l] > lvl)
                        atomicAdd(&num_N_v, 1);
                }

                __syncthreads();

                if (!threadIdx.x)
                    Q_rm[i] = INF;
                
                // if |N[v]| = |L'| then
                if (num_N_v == num_L_nxt) {
                    is_maximal = false;
                    break;
                }
                // else if |N[v]| > 0 then
                else if (num_N_v == 0)
                    // Q' ← Q' ∪ {v};
                    // swap(Q[(*Q_ls_nxt)++], Q[i]);
                    if (!threadIdx.x)
                        Q_rm[i] = lvl;

                __syncthreads();
                
            }
            
            CLK(4);

            __syncthreads();

            //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

            // if is_maximal = TRUE then
            if (is_maximal == true) {

                // foreach v ∈ P do
                for (int i = 0; i < *P_lp_cur; i++) {
                    int v = P[i];

                    if (!threadIdx.x)
                        num_N_v = 0; // |N[v]|

                    __syncthreads();

                    // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                    for (int eid = node[v].start + threadIdx.x, eid_end = node[v].start + node[v].length; eid < eid_end; eid += blockDim.x) {
                        int l = edge[eid];
                        if (L[l] > lvl)
                            atomicAdd(&num_N_v, 1);
                    }

                    __syncthreads();
                    
                    if (!threadIdx.x) {

                        // if |N[v]| = |L'| then
                        if (num_N_v == num_L_nxt)
                            // R' ← R' ∪ {v};
                            R[(*R_lp_nxt)++] = v;
                        // else if |N[v]| > 0 then
                        else if (num_N_v > 0) {
                            // P' ← P' ∪ {v};
                            int P_tmp = P[*P_lp_nxt];
                            P[(*P_lp_nxt)++] = P[i];
                            P[i] = P_tmp;
                        }

                    }

                    __syncthreads();

                }
            
                CLK(5);

                if (!threadIdx.x) {
                
                    //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

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
                    //     printf("blk %d : %d\n", blockIdx.x, num_maximal_bicliques);
                    printf("\33[%d;%dH%d\n", blockIdx.x / 10 + 1, (blockIdx.x % 10) * 10 + 1, ++num_maximal_bicliques);

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

            if (!threadIdx.x) {

                //// if (blockIdx.x == LOG_BLK_ID) {
                ////     printf("\n");
                ////     for (int i = 0; i < lvl; i++) printf("        ");
                ////     printf("L:");
                ////     for (int i = 0; i < *NUM_L; i++)
                ////         printf(" %d", L[i]);
                ////     printf("\n");
                ////     for (int i = 0; i < lvl; i++) printf("        ");
                ////     printf("R:");
                ////     for (int i = 0; i < *R_lp_cur; i++)
                ////         printf(" %d", R[i]);
                ////     printf("\n");
                ////     for (int i = 0; i < lvl; i++) printf("        ");
                ////     printf("P:");
                ////     for (int i = 0; i < *P_lp_cur; i++)
                ////         printf(" %d", P[i]);
                ////     printf("\n");
                ////     for (int i = 0; i < lvl; i++) printf("        ");
                ////     printf("Q:");
                ////     for (int i = 0; i < *Q_lp_cur; i++)
                ////         printf(" %d", Q[i]);
                //// }

                // Select x from P;
                // P <--- P \ {x};
                *x_cur = P[--(*P_lp_cur)];
                //// printf("x: %d\n", *x_cur);
                
                // R' <--- R ∪ {x};
                *R_lp_nxt = *R_lp_cur;
                R[(*R_lp_nxt)++] = *x_cur;

                //// *L_lp_nxt = 0;
                num_L_nxt = 0;
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

            //// printf("L':");
            //// for (int i = 0; i < *L_lp_nxt; i++)
            ////     printf(" %d", L[i]);
            //// printf("\n");
            
            if (!threadIdx.x) {
                
                // P' ← ∅; Q' ← ∅;
                *P_lp_nxt = 0; *Q_lp_nxt = *Q_lp_cur;
                is_maximal = true;

            }

            // foreach v ∈ Q
            for (int i = 0; i < *Q_lp_cur; i++) {
                if (Q_rm[i] < lvl) {
                    __syncthreads();
                    continue;
                }

                int v = Q[i];

                if (!threadIdx.x)
                    num_N_v = 0; // |N[v]|

                __syncthreads();

                // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                for (int eid = node[v].start + threadIdx.x, eid_end = node[v].start + node[v].length; eid < eid_end; eid += blockDim.x) {
                    int l = edge[eid];
                    if (L[l] > lvl)
                        atomicAdd(&num_N_v, 1);
                }
                
                __syncthreads();
                
                if (!threadIdx.x)
                    Q_rm[i] = INF;

                // if |N[v]| = |L'| then
                if (num_N_v == num_L_nxt) {
                    is_maximal = false;
                    break;
                }
                // else if |N[v]| > 0 then
                else if (num_N_v == 0)
                    // Q' ← Q' ∪ {v};
                    // swap(Q[(*Q_ls_nxt)++], Q[i]);
                    if (!threadIdx.x)
                        Q_rm[i] = lvl;

                __syncthreads();
                
            }
            
            CLK(4);

            __syncthreads();

            //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

            // if is_maximal = TRUE then
            if (is_maximal == true) {

                // foreach v ∈ P do
                for (int i = 0; i < *P_lp_cur; i++) {
                    int v = P[i];

                    if (!threadIdx.x)
                        num_N_v = 0; // |N[v]|

                    __syncthreads();

                    // N[v] ← {u ∈ L' | (u, v) ∈ E(G)};
                    for (int eid = node[v].start + threadIdx.x, eid_end = node[v].start + node[v].length; eid < eid_end; eid += blockDim.x) {
                        int l = edge[eid];
                        if (L[l] > lvl)
                            atomicAdd(&num_N_v, 1);
                    }

                    __syncthreads();
                    
                    if (!threadIdx.x) {

                        // if |N[v]| = |L'| then
                        if (num_N_v == num_L_nxt)
                            // R' ← R' ∪ {v};
                            R[(*R_lp_nxt)++] = v;
                        // else if |N[v]| > 0 then
                        else if (num_N_v > 0) {
                            // P' ← P' ∪ {v};
                            int P_tmp = P[*P_lp_nxt];
                            P[(*P_lp_nxt)++] = P[i];
                            P[i] = P_tmp;
                        }

                    }

                    __syncthreads();

                }
            
                CLK(5);

                if (!threadIdx.x) {
                
                    //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

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
                    //     printf("blk %d : %d\n", blockIdx.x, num_maximal_bicliques);
                    printf("\33[%d;%dH%d\n", blockIdx.x / 10 + 1, (blockIdx.x % 10) * 10 + 1, ++num_maximal_bicliques);

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

        // printf("tid: %d, lvl: %d\n", tid, lvl);
        // 感覺這邊break完之後不用做?
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
    
    if (!threadIdx.x) {
        //// printf("\nBlock: %d find %d maximal bicliques.\n", blockIdx.x, num_maximal_bicliques);
        atomicAdd(&total_bic, num_maximal_bicliques);
        for (int i = 0; i < 10; i++) {
            clk[i] >>= 21;
            atomicAdd(&(g_clk[i]), (int)clk[i]);
        }
    }
    grid.sync();
    if (!tid) {
        printf("\33[%d;1Htotal maximal bicliques : %d\n", (gridDim.x + (10-1)) / 10 + 1, total_bic);
        printf("Time:");
        for (int i = 0; i < 10; i++)
            printf(" %d", g_clk[i]);
        printf("\n");
    }
    grid.sync();
}

int main(int argc, char* argv[])
{
    string str_dataset = argv[1];
    cout << str_dataset.substr(str_dataset.rfind('/')+1) << "\n";

    Node *node;
	int *edge;
    int *NUM_L, *NUM_R, *NUM_EDGES, _;
    // MBE
    int *u2L, *L, *R, *P, *Q;
    int *x, *L_lp, *R_lp, *P_lp, *Q_lp;
    int *Q_rm;
    // MBE_82
    int *g_u2L, *g_L, *g_R, *g_P, *g_Q;
    int *g_x, *g_L_lp, *g_R_lp, *g_P_lp, *g_Q_lp;
    int *g_Q_rm;
    cudaMallocManaged(&NUM_EDGES, sizeof(int));
    cudaMallocManaged(&NUM_L    , sizeof(int));
    cudaMallocManaged(&NUM_R    , sizeof(int));

    ifstream fin;
    fin.open(argv[1]);
    fin >> *NUM_R >> *NUM_L >> *NUM_EDGES;
    cudaMallocManaged(&node, sizeof(Node)*(*NUM_R    ));
    cudaMallocManaged(&edge, sizeof(int )*(*NUM_EDGES));
    for (int i = 0; i < *NUM_R    ; i++) fin >> node[i].start >> node[i].length;
    for (int i = 0; i < *NUM_EDGES; i++) fin >> edge[i] >> _;
    fin.close();

    int numBlocksPerSM;
    int numThreads = NUM_THDS;
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, CUDA_MBE, numThreads, 0);
    int numBlocks_max = deviceProp.multiProcessorCount * numBlocksPerSM;
    int numBlocks = NUM_BLKS <= numBlocks_max ? NUM_BLKS : numBlocks_max;
    dim3 num_blocks_MBE(1, 1, 1);
    dim3 num_blocks_MBE_82(numBlocks, 1, 1);
    dim3 block_size(numThreads, 1, 1);

    // MBE
    cudaMallocManaged(&u2L , sizeof(int)*(*NUM_L)); my_memset_order(u2L, 0, *NUM_L);
    cudaMallocManaged(&L   , sizeof(int)*(*NUM_L)); my_memset_order(L  , 0, *NUM_L);
    cudaMallocManaged(&R   , sizeof(int)*(*NUM_R)); my_memset_order(R  , 0, *NUM_R);
    cudaMallocManaged(&P   , sizeof(int)*(*NUM_R)); my_memset_order(P  , 0, *NUM_R);
    cudaMallocManaged(&Q   , sizeof(int)*(*NUM_R)); my_memset_order(Q  , 0, *NUM_R);
    cudaMallocManaged(&x   , sizeof(int)*(*NUM_R)); my_memset(x   ,     -1, *NUM_R);
    cudaMallocManaged(&L_lp, sizeof(int)*(*NUM_R)); my_memset(L_lp, *NUM_L, *NUM_R);
    cudaMallocManaged(&R_lp, sizeof(int)*(*NUM_R)); my_memset(R_lp,      0, *NUM_R);
    cudaMallocManaged(&P_lp, sizeof(int)*(*NUM_R)); my_memset(P_lp, *NUM_R, *NUM_R);
    cudaMallocManaged(&Q_lp, sizeof(int)*(*NUM_R)); my_memset(Q_lp,      0, *NUM_R);
    cudaMallocManaged(&Q_rm, sizeof(int)*(*NUM_R)); my_memset(Q_rm,    INF, *NUM_R);
    // MBE_82
    cudaMallocManaged(&g_u2L , sizeof(int)*(*NUM_L)*numBlocks); for (int i = numBlocks * (*NUM_L); i-- > 0; )  g_u2L[i] =  u2L[i % (*NUM_L)];
    cudaMallocManaged(&g_L   , sizeof(int)*(*NUM_L)*numBlocks); for (int i = numBlocks * (*NUM_L); i-- > 0; )    g_L[i] =    L[i % (*NUM_L)];
    cudaMallocManaged(&g_R   , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )    g_R[i] =    R[i % (*NUM_R)];
    cudaMallocManaged(&g_P   , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )    g_P[i] =    P[i % (*NUM_R)];
    cudaMallocManaged(&g_Q   , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )    g_Q[i] =    Q[i % (*NUM_R)];
    cudaMallocManaged(&g_x   , sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; )    g_x[i] =    x[i % (*NUM_R)];
    cudaMallocManaged(&g_L_lp, sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; ) g_L_lp[i] = L_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_R_lp, sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; ) g_R_lp[i] = R_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_P_lp, sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; ) g_P_lp[i] = P_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_Q_lp, sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; ) g_Q_lp[i] = Q_lp[i % (*NUM_R)];
    cudaMallocManaged(&g_Q_rm, sizeof(int)*(*NUM_R)*numBlocks); for (int i = numBlocks * (*NUM_R); i-- > 0; ) g_Q_rm[i] = Q_rm[i % (*NUM_R)];

    void *kernelArgs_MBE[] = {&NUM_L, &NUM_R, &NUM_EDGES, &node, &edge, &u2L, &L, &R, &P, &Q, &x, &L_lp, &R_lp, &P_lp, &Q_lp};
    void *kernelArgs_MBE_82[] = {&NUM_L, &NUM_R, &NUM_EDGES, &node, &edge, &g_u2L, &g_L, &g_R, &g_P, &g_Q, &g_Q_rm, &g_x, &g_L_lp, &g_R_lp, &g_P_lp, &g_Q_lp};

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int stat;
    int mode = NUM_BLKS;
    if (mode == -2)
        maximal_bic_enum_set(NUM_L, NUM_R, NUM_EDGES, node, edge, u2L, L, R, P, Q, x, L_lp, R_lp, P_lp, Q_lp);
    else if (mode == -1)
        maximal_bic_enum(NUM_L, NUM_R, NUM_EDGES, node, edge, u2L, L, R, P, Q, x, L_lp, R_lp, P_lp, Q_lp);
    else if (mode == 0) {
        cudaLaunchCooperativeKernel((void*)CUDA_MBE, num_blocks_MBE, block_size, kernelArgs_MBE);
    }
    else {
        cout << "\33[2J\n";
        stat = cudaLaunchCooperativeKernel((void*)CUDA_MBE_82, num_blocks_MBE_82, block_size, kernelArgs_MBE_82);
    }
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cout << "status: " << stat << "\n";
    printf("Running time: %f secs\n", time/1000);

    cudaFree(node);
    cudaFree(edge);
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
}
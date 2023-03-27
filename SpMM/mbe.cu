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

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

typedef struct
{
	unordered_set<int> L;
	unordered_set<int> R;
} Biclique;

void my_memset(int *SA, int val, int len) {
    for (int i = 0; i < len; i++)
        SA[i] = val;
}

void my_memset_order(int *SA, int val_start, int val_end) {
    for (int i = val_start; i < val_end; i++)
        SA[i - val_start] = i;
}

void maximal_bic_enum(int *NUM_NODES, int *NUM_EDGES, Node *node, int *edge,
                      int *L, int *R, int *P, int *Q, int lvl) {
    // int    v[*NUM_NODES]; my_memset(&v   ,         -1, *NUM_NODES);
    int    x[*NUM_NODES]; my_memset(x   ,         -1, *NUM_NODES);
    int  u2L[*NUM_NODES]; my_memset_order(u2L, 0, *NUM_NODES);
    int L_lp[*NUM_NODES]; my_memset(L_lp, *NUM_NODES, *NUM_NODES);
    int R_lp[*NUM_NODES]; my_memset(R_lp,          0, *NUM_NODES);
    int P_lp[*NUM_NODES]; my_memset(P_lp, *NUM_NODES, *NUM_NODES);
    int Q_lp[*NUM_NODES]; my_memset(Q_lp,          0, *NUM_NODES);
    vector<Biclique> maximal_bicliques;
    
    for (lvl = 0; lvl >= 0; ) {

        //// printf("lvl: %d\n", lvl);

        // int *v_cur    = &(   v[lvl]);
        int *x_cur    = &(   x[lvl]);
        int *L_lp_cur = &(L_lp[lvl]), *L_lp_nxt = &(L_lp[lvl+1]);
        int *R_lp_cur = &(R_lp[lvl]), *R_lp_nxt = &(R_lp[lvl+1]);
        int *P_lp_cur = &(P_lp[lvl]), *P_lp_nxt = &(P_lp[lvl+1]);
        int *Q_lp_cur = &(Q_lp[lvl]), *Q_lp_nxt = &(Q_lp[lvl+1]);
        bool is_recursive = false;

        // while P ≠ ∅ do
        while (*P_lp_cur != 0) {

            //// string tab_level(lvl << 3, ' ');
            //// printf("\n%sL:", tab_level.c_str());
            //// for (int i = 0; i < *L_lp_cur; i++)
            ////     printf(" %d", L[i]);
            //// printf("\n%sR:", tab_level.c_str());
            //// for (int i = 0; i < *R_lp_cur; i++)
            ////     printf(" %d", R[i]);
            //// printf("\n%sP:", tab_level.c_str());
            //// for (int i = 0; i < *P_lp_cur; i++)
            ////     printf(" %d", P[i]);
            //// printf("\n%sQ:", tab_level.c_str());
            //// for (int i = 0; i < *Q_lp_cur; i++)
            ////     printf(" %d", Q[i]);

            // Select x from P;
            // P <--- P \ {x};
            *x_cur = P[--(*P_lp_cur)];
            //// printf("x: %d\n", *x_cur);
            
            // R' <--- R ∪ {x};
            *R_lp_nxt = *R_lp_cur;
            R[(*R_lp_nxt)++] = *x_cur;

            *L_lp_nxt = 0; // |L'|
            // L' <--- {u ∈ L | (u, x) ∈ E(G)};
            for (int eid = node[*x_cur].start, eid_end = eid + node[*x_cur].length; eid < eid_end; eid++) {
                int u = edge[eid];
                int l = u2L[u];
                if (l < *L_lp_cur) {
                    swap(L[(*L_lp_nxt)++], L[l]);
                    swap(u2L[L[l]], u2L[u]);
                }
            }

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
                // else if (num_N_v > 0)
                //     // Q' ← Q' ∪ {v};
                //     (*Q_nxt).insert(v);
                
            }

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
                
                //// printf("R_lp_nxt: %d\n", *R_lp_nxt);

                // PRINT(L', R');
                //// printf("\n-------------- find maximal biclique --------------");
                //// printf("\nL':");
                //// for (int i = 0; i < *L_lp_nxt; i++)
                ////     printf(" %d", L[i]);
                //// printf("\nR':");
                //// for (int i = 0; i < *R_lp_nxt; i++)
                ////     printf(" %d", R[i]);
                //// printf("\n");
                //// printf("---------------------------------------------------\n");

                // save maximal bicliques
                Biclique new_maximal_bicliques;
                for (int i = 0; i < *L_lp_nxt; i++)
                    new_maximal_bicliques.L.insert(L[i]);
                for (int i = 0; i < *R_lp_nxt; i++)
                    new_maximal_bicliques.R.insert(R[i]);
                maximal_bicliques.push_back(new_maximal_bicliques);
                if (rand() % 100000 == 0) printf(".\n");

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

    printf("Find %d maximal bicliques.\n", maximal_bicliques.size());
    if (*NUM_NODES > 100) return;

    // string _ = "";
    // printf("\33[2J\33[0;0H");

    // printf("  ");
    // for (int i = 0; i < *NUM_NODES; i++)
    //     printf(" %d", i / 10);
    // printf("\n  ");
    // for (int i = 0; i < *NUM_NODES; i++)
    //     printf(" %d", i % 10);
    // printf("\n");
    // for (int i = 0; i < *NUM_NODES; i++) {
    //     bool adj_vec[*NUM_NODES] = { false };
    //     for (int j = node[i].start, j_end = j + node[i].length; j < j_end; j++)
    //         adj_vec[edge[j]] = true;
    //     printf("%d%d", i / 10, i % 10);
    //     for (int j = 0; j < *NUM_NODES; j++)
    //         printf(" %c", adj_vec[j] ? '#' : '-');
    //     printf("\n");
    // }

    // for (int i = 0, i_end = maximal_bicliques.size(); i < i_end; i++) {
    //     printf("\33[7m");
    //     for (const auto &r: maximal_bicliques[i].R)
    //         for (const auto &l: maximal_bicliques[i].L)
    //             printf("\33[%d;%dH#\n", 3 + r, 4 + l * 2);
    //     printf("\33[0m\n\33[%d;0H\n", 3 + (*NUM_NODES));
    //     if      (_ == "auto") usleep(800000);
    //     else if (_ != "exit") cin >> _;
    //     for (const auto &r: maximal_bicliques[i].R)
    //         for (const auto &l: maximal_bicliques[i].L)
    //             printf("\33[%d;%dH#\n", 3 + r, 4 + l * 2);
    // }
    // printf("\33[%d;0H\n", 3 + (*NUM_NODES));
}

int main(int argc, char* argv[])
{
    string str_dataset = argv[1];
    cout << str_dataset.substr(str_dataset.rfind('/')+1) << "\n";

    Node *node;
	int *edge;
    int *NUM_NODES, *NUM_EDGES, SOURCE, _;
    int *L, *R, *P, *Q;
    cudaMallocManaged(&NUM_EDGES, sizeof(int));
    cudaMallocManaged(&NUM_NODES, sizeof(int));

    ifstream fin;
    fin.open(argv[1]);
    fin >> *NUM_NODES >> *NUM_EDGES >> SOURCE;
    cudaMallocManaged(&node, sizeof(Node)*(*NUM_NODES));
    cudaMallocManaged(&edge, sizeof(int)*(*NUM_EDGES));
    for(int i=0;i<*NUM_NODES;i++) fin >> node[i].start >> node[i].length;
    for(int i=0;i<*NUM_EDGES;i++) fin >> edge[i] >> _;
    fin.close();

    cudaMallocManaged(&L, sizeof(int)*(*NUM_NODES)); my_memset_order(L, 0, *NUM_NODES);
    cudaMallocManaged(&R, sizeof(int)*(*NUM_NODES)); my_memset_order(R, 0, *NUM_NODES);
    cudaMallocManaged(&P, sizeof(int)*(*NUM_NODES)); my_memset_order(P, 0, *NUM_NODES);
    cudaMallocManaged(&Q, sizeof(int)*(*NUM_NODES)); my_memset_order(Q, 0, *NUM_NODES);

    maximal_bic_enum(NUM_NODES, NUM_EDGES, node, edge, L, R, P, Q, ONE);

    cudaFree(node);
    cudaFree(edge);
    cudaFree(NUM_EDGES);
    cudaFree(NUM_NODES);
}
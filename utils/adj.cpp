#include <bits/stdc++.h>
using namespace std;

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

int main(int argc, char* argv[])
{
    string str_dataset = argv[1];
    cout << str_dataset.substr(str_dataset.rfind('/')+1) << "\n";

    int NUM_R, NUM_L, NUM_EDGES, _;

    ifstream fin;
    fin.open(argv[1]);
    fin >> NUM_R >> NUM_L >> NUM_EDGES;
    vector<Node> node(NUM_R);
    vector<int> edge(NUM_EDGES);
    for (int i = 0; i < NUM_R    ; i++) fin >> node[i].start >> node[i].length;
    for (int i = 0; i < NUM_EDGES; i++) fin >> edge[i] >> _;
    fin.close();
    
    ofstream fout;
    fout.open(argv[2]);
    for (int i = 0; i < NUM_R; i++) {
        vector<bool> adj_vec(NUM_L, 0);
        for (int j = node[i].start, j_end = j + node[i].length; j < j_end; j++)
            adj_vec[edge[j]] = true;
        for (int j = 0; j < NUM_L; j++)
            fout << (j ? " " : i ? "\n" : "") << (adj_vec[j] ? '1' : '0');
    }
}
#include <bits/stdc++.h>
using namespace std;

// -------------------------------- //
// Use: Convert to bipartite graph  //
//      (from edge-pair to CSR)     //
// ---------- How to use ---------- //
// g++ -O3 gen_bi.cpp -o gen_bi.out //
// ./gen_bi.out [f_edge] [f_CSR]    //
// -------------------------------- //

int main(int argc, char* argv[])
{
	if (!argv[2]) { cout << "Wrong Dataset :(\n"; return 0; }

    ifstream fin;
    fin.open(argv[1]);

	string _;
	int num_r = 0, num_l = 0, num_edges = 0, source = 0;
	int num_passed_words, num_passed_words_per_edge;
	
	// ----- Something needed to be passed ----- //
	cout << "Number of passed words: ";
	if (argv[3])
		cout << (num_passed_words = atoi(argv[3])) << "\n";
	else
		cin >> num_passed_words;
	cout << "Number of passed words per edge: ";
	if (argv[4])
		cout << (num_passed_words_per_edge = atoi(argv[4])) << "\n";
	else
		cin >> num_passed_words_per_edge;
	for (int i = 0; i < num_passed_words; i++) fin >> _;

	map<int, int> row_id, col_id;
	vector<int> node_length;
	vector< unordered_set<int> > edge;
	
	for (int node_r, node_l, i_edge = 0, progress_reporter = -1, progress_clk = -1; fin >> node_r >> node_l/*fin >> node_l >> node_r*/; i_edge++) {
		for (int i = 0; i < num_passed_words_per_edge; i++) fin >> _;
		if (row_id.insert(pair<int, int>(node_r, num_r)).second) {
			edge.resize(++num_r);
			node_length.resize(num_r);
		}
		if (col_id.insert(pair<int, int>(node_l, num_l)).second) {
			num_l++;
		}
		int node_r_id = row_id[node_r];
		int node_l_id = col_id[node_l];
		edge[node_r_id].insert(node_l_id);
		int percent = (double)i_edge / num_edges * 100;
		int runtime = clock() >> 19;
		if (percent > progress_reporter && runtime > progress_clk) {
			progress_reporter = percent; progress_clk = runtime;
			cout << "Loading..." << setw(3) << progress_reporter << " %\n";
		}
	}

	for (int i = 0; i < num_r; i++) {
		node_length[i] = edge[i].size();
		num_edges += node_length[i];
	}

	fin.close();
	
	cout << "Loading... Done\nWriting...";

	ofstream fout;
	fout.open(argv[2]);

	fout << num_r << ' ' << num_l << ' ' << num_edges << "\n\n";

	for (int sum = 0, i = 0; i < num_r; sum += node_length[i++])
		fout << sum << " " << node_length[i] << "\n";
	
	for (int i = 0; i < num_r; i++)
		for (const auto &edge_ : edge[i])
			fout << "\n" << edge_ << " 1";

	cout << " Done\n";

	fout.close();
}
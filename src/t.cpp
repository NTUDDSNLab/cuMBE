#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[])
{
    int    L[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int utoL[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    int lp = 0, u, l;
    int u_list[3] = {3, 6, 1};

    cout << "\n";
    for (int ui = -1; ++ui < 3; cout << "\n") {
        u = u_list[ui];
        l = utoL[u];
        cout << "lp: " << lp << " -> " << lp+1 << ", u: " << u << "\n";
        cout << "   L: "; for (int i = 0; i < 10; i++) cout <<    L[i] << ' '; cout << "\n";
        cout << "utoL: "; for (int i = 0; i < 10; i++) cout << utoL[i] << ' '; cout << "\n";
        swap(L[lp++], L[l]);
        swap(utoL[L[l]], utoL[u]);
        cout << "   L: "; for (int i = 0; i < 10; i++) cout <<    L[i] << ' '; cout << "\n";
        cout << "utoL: "; for (int i = 0; i < 10; i++) cout << utoL[i] << ' '; cout << "\n";
    }

    return 0;
}
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;


vector<string> split(const string& str, const char regx, vector<string>& strs) {
    strs.clear();
    int left = 0;
    int right = 0;
    int size = str.size();
    
    while (right < size) {
        if (str[right] != regx)
            right++;
        else {
            strs.push_back(str.substr(left, right - left));
            right++;
            left = right;
        }
    }
    if (right != left)
        strs.push_back(str.substr(left, right - left));

    return strs;
}


void load_ply(const char* filename, Mat& pc) {
    fstream ifs(filename);
    string str;
    vector<string> strs;
    int num_vertex = 0;
    int num_col = 0;
    while (getline(ifs, str)) {
        if (str.find("element vertex") != string::npos) {
            split(str, ' ', strs);
            num_vertex = stoi(strs[2]);
        } else if (str == "end_header")
            break;
        else if (str.substr(0, 8) == "property")
            num_col++;
    }
    num_col--;
    pc.create(num_vertex, 6, CV_32FC1);

    for (int i = 0; i < num_vertex; i++) {
        float* data = pc.ptr<float>(i);
        int col = 0;
        for (; col < 6; col++)
            ifs >> data[col];
        for (; col < num_col; col++) {
            int tmp;
            ifs >> tmp;
        }
        float norm = sqrt(data[3] * data[3] + data[4] * data[4] + data[5] * data[5]);
        data[3] /= norm;
        data[4] /= norm;
        data[5] /= norm;
    }
}


int main() {
    Mat pc;
    load_ply("./models/obj_01.ply", pc);
    for (int i = 0; i < 100; i++) {
        float* data = pc.ptr<float>(i);
        for (int j = 0; j < 6; j++)
            cout << data[j] << " ";
        cout << endl;
    }
    return 0;
}

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <cassert>
#include <sstream>
#include <fstream>

using namespace std;

const double NMS_THR = 0.5;
const double EPS = 1e-8;

int sgn(double x) {
    return (x > EPS) - (x < -EPS);
}

struct Proposal {
    vector<double> box;
    double score;
    Proposal(const vector<double>& v): box(v.begin() + 2, v.begin() + 6), score(v[6]) {
    }

    void print(ostream& out, int idx) const {
        vector<double> pbox(box.begin(), box.end());
        pbox[2] -= pbox[0];
        pbox[3] -= pbox[1];
        pbox[0] += 1.0;
        pbox[1] += 1.0;
        out << idx << ",-1," << fixed << pbox[0] << "," << pbox[1] << "," << pbox[2] << ","
            << pbox[3] << "," << score << ",1,-1,-1" << endl;
    }

    double get_iou(const Proposal& pro) const {
        double x_min = max(box[0], pro.box[0]);
        double y_min = max(box[1], pro.box[1]);
        double x_max = min(box[2], pro.box[2]);
        double y_max = min(box[3], pro.box[3]);

        double w = max(0., x_max - x_min + 1.);
        double h = max(0., y_max - y_min + 1.);

        double inter = w * h;
        double uni = (box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) + (pro.box[2] - pro.box[0] + 1.) * (pro.box[3] - pro.box[1] + 1.) - inter;

        return inter / uni;
    }
};

struct DpNode {
    double val;
    vector<int> prev, next;
    int route;
    bool used;

    DpNode(): val(0), route(-1), used(false) {
        prev.clear();
        next.clear();
    }
};

struct QNode {
    int i, j;
    double val;
    QNode(int _i, int _j, double _val): i(_i), j(_j), val(_val) {}
    bool operator < (const QNode& node) const {
        return val < node.val;
    }
};

void dynamic_programming(const vector<vector<Proposal>>& proposals, vector<vector<DpNode>>& dp, double thr) {
    // ---- Build Graph ----
    dp.resize(proposals.size());
    for (int i = 0; i < proposals.size(); ++i) {
        dp[i].clear();
        dp[i].resize(proposals[i].size());
        for (int j = 0; j < proposals[i].size(); ++j) {
            if (i > 0) {
                for (int k = 0; k < proposals[i - 1].size(); ++k) {
                    if (proposals[i][j].get_iou(proposals[i - 1][k]) > thr) {
                        dp[i][j].prev.push_back(k);
                        dp[i - 1][k].next.push_back(j);
                        //cerr << "Graph Edge: (" << i - 1 << "," << k << "), (" << i << "," << j << ")" << endl;
                    }
                }
            }
        }
    }

    // ---- DP ----
    for (int i = 0; i < dp.size(); ++i) {
        for (int j = 0; j < dp[i].size(); ++j) {
            for (int k: dp[i][j].prev) {
                if (dp[i][j].val <= dp[i - 1][k].val) {
                    dp[i][j].val = dp[i - 1][k].val;
                    dp[i][j].route = k;
                }
            }
            dp[i][j].val += proposals[i][j].score;
            //cerr << "dp: " << i << " " << j << ": " << dp[i][j].val << " " << dp[i][j].route << endl;
        }
    }
}

pair<vector<int>, int> find_route(const vector<vector<DpNode>>& dp, int ei, int ej) {
    vector<int> routes;
    while (ej != -1) {
        routes.push_back(ej);
        ej = dp[ei][ej].route;
        --ei;
    }
    reverse(routes.begin(), routes.end());
    return make_pair(routes, ei + 1);
}

vector<vector<int>> seq_nms(vector<vector<Proposal>>& proposals, double thr = 0.5) {
    vector<vector<DpNode>> dp;

    // ---- Init DP && Priority Queue ----
    dynamic_programming(proposals, dp, 1);
    priority_queue<QNode> heap;
    int tot = 0; // 用来计数
    for (int i = 0; i < dp.size(); ++i) {
        for (int j = 0; j < dp[i].size(); ++j) {
            heap.push(QNode(i, j, dp[i][j].val));
            ++tot;
        }
    }

    vector<vector<int>> res(proposals.size());
    while (!heap.empty()) {
        cerr << "Remaining Box: " << tot << endl;
        QNode node(1, 0, 0);
        do {
            //cerr << node.val << "haha" << dp[node.i][node.j].val << endl;
            node = heap.top();
            heap.pop();
            //cerr << "in: " << node.i << " " << node.j << " " << node.val << "used " << dp[node.i][node.j].used << "val: " << dp[node.i][node.j].val << endl;
        } while (!heap.empty() && (dp[node.i][node.j].used || sgn(node.val - dp[node.i][node.j].val) != 0));
        if (heap.empty() && (dp[node.i][node.j].used || sgn(node.val - dp[node.i][node.j].val) != 0)) break;

        // ---- Best Seq ----
        pair<vector<int>, int> mid_res = find_route(dp, node.i, node.j);
        vector<int>& best_seq = mid_res.first;
        int st = mid_res.second;

        // ---- rescore ----
        double sum = dp[st + best_seq.size() - 1][best_seq.back()].val;
        for (int i = 0; i < best_seq.size(); ++i) {
            //cerr << "Rescore: " << st + i << " " << best_seq[i] << endl;
            proposals[st + i][best_seq[i]].score = sum / best_seq.size();
            //cerr << "Assign True: " << st + i << " " << best_seq[i] << " Route: " << dp[st + i][best_seq[i]].route << endl;
            dp[st + i][best_seq[i]].used = true;
            res[st + i].push_back(best_seq[i]);
        }

        // ---- Suppress && Update the dp value ----
        vector<int> nows;
        for (int i = 0; i < best_seq.size(); ++i) {
            vector<int> nexts;
            for (int j = 0; j < proposals[st + i].size(); ++j) {
                if (j != best_seq[i] && dp[st + i][j].used) continue;
                double x = proposals[st + i][best_seq[i]].get_iou(proposals[st + i][j]);
                if (x > NMS_THR) {
                    //cerr << "Suppress: " << st + i << " " << j << endl;
                    --tot;
                    dp[st + i][j].used = true; // Supress
                    for (int prev: dp[st + i][j].prev) {
                        auto it = find(dp[st + i - 1][prev].next.begin(), dp[st + i - 1][prev].next.end(), j);
                        dp[st + i - 1][prev].next.erase(it);
                    }
                    for (int next: dp[st + i][j].next) {
                        auto it = find(dp[st + i + 1][next].prev.begin(), dp[st + i + 1][next].prev.end(), j);
                        dp[st + i + 1][next].prev.erase(it);
                        if (!dp[st + i + 1][next].used && dp[st + i + 1][next].route == j) {
                            //cerr << "push next1: " << st + i << "," << j << " ---> " << st + i + 1 << " " << next << " Route: " << dp[st + i + 1][next].route << endl;
                            nexts.push_back(next);
                        }
                    }
                    dp[st + i][j].prev.clear();
                    dp[st + i][j].next.clear();
                }
            }
            for (int now: nows) {
                for (int next: dp[st + i][now].next) {
                    if (dp[st + i + 1][next].route == now) {
                        //cerr << "push next2: " << st + i << "," << now << " ---> " << st + i + 1 << "," << next << " Route: " << dp[st + i + 1][next].route << endl;
                        nexts.push_back(next);
                    }
                }
            }
            sort(nexts.begin(), nexts.end());
            nexts.erase(unique(nexts.begin(), nexts.end()), nexts.end());
            for (int next: nexts) {
                //cerr << "calc next: " << st + i + 1 << " " << next << endl;
                assert(dp[st + i + 1][next].used == false);
                dp[st + i + 1][next].val = 0;
                dp[st + i + 1][next].route = -1;
                for (int now: dp[st + i + 1][next].prev) {
                    if (dp[st + i + 1][next].val <= dp[st + i][now].val) {
                        dp[st + i + 1][next].val = dp[st + i][now].val;
                        dp[st + i + 1][next].route = now;
                    }
                }
                dp[st + i + 1][next].val += proposals[st + i + 1][next].score;
                heap.push(QNode(st + i + 1, next, dp[st + i + 1][next].val));
            }
            nows.swap(nexts);
            nexts.clear();
        }
    }
    return res;
}

vector<vector<double>> load(const string& path) {
    string line;
    vector<vector<double>> res;
    ifstream fin(path);
    while (getline(fin, line)) {
        vector<double> vec;
        istringstream sin(line);
        double f;
        char c;
        while (sin >> f) {
            vec.push_back(f);
            sin >> c;
        }
        res.push_back(vec);
    }
    return res;
}

int main() {
    string vid_res = "/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_model0_2015_proposal_300/mot_model0_2015_without_nms";
    string vid_list[] = {"PETS09-S2L1", "TUD-Campus", "TUD-Stadtmitte", "ETH-Bahnhof", "ETH-Sunnyday", "KITTI-13", "KITTI-17"};
    // string vid_list[] = {""};

    string out_dir = "/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_model0_2015_proposal_300/mot_model0_2015_without_nms";

    for (string vid_name: vid_list) {
        
        // ---- Read ----
        string path = vid_res + "/" + vid_name + "/" + "det.txt";
        // string path = "det.txt";
        vector<vector<double>> seq_det = load(path);
        vector<vector<vector<double>>> val;
        for (auto& seq: seq_det) {
            seq[2] -= 1;
            seq[3] -= 1;
            seq[4] += seq[2];
            seq[5] += seq[3];
            if (seq[6] <= 1e-3) continue;
            int idx = seq[0];
            while (val.size() <= idx) val.push_back(vector<vector<double>>());
            val[idx].push_back(seq);
        }
        cerr << "processing " << vid_name << ", " << val.size() << " images" << endl;

        // ---- Build Proposals ----
        vector<vector<Proposal>> proposals;
        for (auto& one_val: val) {
            sort(one_val.begin(), one_val.end(), [](const vector<double>& v1, const vector<double>& v2) {
                    for (int i = 0; i < v1.size(); ++i) {
                        if (v1[i] != v2[i]) return v1[i] > v2[i];
                    }
                    return false;
                }
            );
            proposals.push_back(vector<Proposal>());
            for (auto& vec: one_val) {
                proposals.back().push_back(Proposal(vec));
            }
        }

        // ---- SEQ_NMS ----
        vector<vector<int>> res_seq_nms = seq_nms(proposals);

        // ---- OUTPUT ----
        string out_path = out_dir + "/" + vid_name + "/" + "det_seq.txt";
        // string out_path = "det_seq.txt";
        ofstream fout(out_path);
        for (int fr_idx = 0; fr_idx < res_seq_nms.size(); ++fr_idx) {
            for (int box_idx: res_seq_nms[fr_idx]) {
                //cerr << fr_idx << " " << box_idx << endl;
                proposals[fr_idx][box_idx].print(fout, fr_idx);
            }
        }
        fout.close();
    }

    return 0;
}

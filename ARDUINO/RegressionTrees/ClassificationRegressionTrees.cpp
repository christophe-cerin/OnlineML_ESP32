/* mainco2t.cpp
CART (Classification and Regression Trees)

https://github.com/magikerwin/ML-From-Scratch-With-CPP/tree/main/RegressionTrees

Regression:
The cost function that is minimized to choose split points is the sum squared error across all training samples that fall within the rectangle.

Classification:
The Gini cost function is used which provides an indication of how pure the node are, where node purity refers to how mixed the training data assigned to each node is.


1. Gini Index (cost function to evaluate splits in the dataset)
2. Create Split
3. Build a Tree
    3.1 Terminal Nodes (Maximum Tree Depth, Minimum Node Records)
    3.2 Recursive Splitting
    3.3 Building a Tree
4. Make a Prediction

*/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <cfloat>
#include <algorithm>

using namespace std;

double giniIndex(const vector<vector<vector<double>>> &groups, vector<double> classes) {
    
    // count all samples as split point
    int numOfInstances = 0;
    for (auto group : groups) {
        numOfInstances += group.size();
    }
    
    // sum weighted Gini index for each group
    double gini = 0;
    for (auto group : groups) {
        double size = group.size();
        if (size == 0) continue; // avoid divide by zero
        
        double score = 1.0; // 1 - p_1^2 - p_2^2 - ... - - p_N^2
        for (int classIdx : classes) {
            double count = 0;
            for (auto instance : group) {
                double label = instance[instance.size() - 1];
                if (label == classIdx) count++;
            }
            double p = count / size;
            score -= p * p;
        }
        gini += (size / numOfInstances) * score;
    }
    
    return gini;
}

vector<vector<vector<double>>> splitGroups(
    int featureIdx, double value, 
    const vector<vector<double>> &dataset) {
    
    vector<vector<double>> lefts;
    vector<vector<double>> rights;
    for (auto data : dataset) {
        if (data[featureIdx] < value) {
            lefts.push_back(data);
        } else {
            rights.push_back(data);
        }
    }
    
    return {lefts, rights};
}

struct Node {
    int featureIdx;
    double featureValue;
    double gini;
    vector<vector<vector<double>>> groups;
    
    Node* left = nullptr;
    Node* right = nullptr;
    double label = -1;
};

Node* getSplit(const vector<vector<double>> &dataset) {
    int numOfFeatures = dataset[0].size() - 1;
    
    // get labels lists
    unordered_set<double> bucket;
    for (auto data : dataset) {
        double classIdx = data[numOfFeatures];
        bucket.insert(classIdx);
    }
    vector<double> labels;
    for (auto label : bucket) labels.push_back(label);
    sort(labels.begin(), labels.end());
    
    // split groups by min gini
    double minGini = DBL_MAX;
    Node* info = new Node;
    for (int featureIdx=0; featureIdx<numOfFeatures; featureIdx++) {
        for (auto data : dataset) {
            auto groups = splitGroups(featureIdx, data[featureIdx], dataset);
            auto gini = giniIndex(groups, labels);
            // cout << "X1 < " << data[featureIdx] << ", gini = " << gini << endl;
            if (gini < minGini) {
                minGini = gini;
                info->featureIdx = featureIdx;
                info->featureValue = data[featureIdx];
                info->gini = gini;
                info->groups = groups;
            }
        }
    }
    return info;
}

// Create a terminal node value, and it will return most common output value
double toTerminal(const vector<vector<double>> &group) {
    unordered_map<double, int> counter;
    for (auto data : group) {
        double label = data[data.size()-1];
        if (counter.count(label) == 0) {
            counter[label] = 1;
        } else {
            counter[label] += 1;
        }
    }
    
    int maxCount = 0;
    double targetLabel;
    for (auto item : counter) {
        if (item.second > maxCount) {
            maxCount = item.second;
            targetLabel = item.first;
        }
    }
    return targetLabel;
}

// Create child splits for a node or make terminal
void split(Node* currNode, int maxDepth, int minSize, int depth) {
    auto leftGroup = currNode->groups[0];
    auto rightGroup = currNode->groups[1];
    currNode->groups.clear();
    
    // check for a no split
    if (leftGroup.empty() || rightGroup.empty()) {
        if (leftGroup.empty()) {
            currNode->right = new Node;
            currNode->right->label = toTerminal(rightGroup);
        } else {
            currNode->left = new Node;
            currNode->left->label = toTerminal(leftGroup);
        }
        return;
    }
    // check for max depth
    if (depth >= maxDepth) {
        currNode->left = new Node;
        currNode->left->label = toTerminal(leftGroup);
        currNode->right = new Node;
        currNode->right->label = toTerminal(rightGroup);
        return;
    }
    // process left child
    if (leftGroup.size() <= minSize) {
        currNode->left = new Node;
        currNode->left->label = toTerminal(leftGroup);
    } else {
        currNode->left = getSplit(leftGroup);
        split(currNode->left, maxDepth, minSize, depth+1);
    }
    // process right child
    if (rightGroup.size() <= minSize) {
        currNode->right = new Node;
        currNode->right->label = toTerminal(rightGroup);
    } else {
        currNode->right = getSplit(rightGroup);
        split(currNode->right, maxDepth, minSize, depth+1);
    }
}

Node* buildTree(
    const vector<vector<double>> &dataset, 
    int maxDepth, int minSize) {
    
    Node* root = getSplit(dataset);
    split(root, maxDepth, minSize, 1);
    return root;
}

void printTree(Node* root, int depth) {
    if (root == nullptr) return;
    
    if (root->label != -1) {
        cout << "depth: " << depth
            << ", label: " << root->label << endl;
    } else {
        cout << "depth: " << depth
            << ", featureIdx: " << root->featureIdx 
            << ", featureValue: " << root->featureValue << endl;
    }
    
    printTree(root->left, depth+1);
    printTree(root->right, depth+1);
}

double predict(Node* currNode, vector<double> data) {
    
    if (currNode->label != -1) return currNode->label;
    
    double featureValue = data[currNode->featureIdx];
    if (featureValue < currNode->featureValue) {
        if (currNode->left != nullptr) {
            return predict(currNode->left, data);
        }
    } else {
        if (currNode->right != nullptr) {
            return predict(currNode->right, data);
        }
    }
    return -1;
}

int main() {
	

	vector<vector<double>> dataset = {
		{264,20,2},
		{269,21.7,2},
		{262,20.1,1},
		{266,21.6,2},
		{252,20,1},
		{276,21.6,3},
		{259,20,1},
		{270,21.6,2},
		{261,20,1},
		{264,21.6,2},
		{261,21.6,1},
		{266,20,2},
		{262,21.6,1},
		{256,20,1},
		{263,21.5,2},
		{250,20,1},
		{253,20,1},
		{256,21.6,1},
		{267,20,2},
		{276,21.6,3},
		{263,20,2},
		{268,20,2},
		{269,21.5,2},
		{251,20.1,1},
		{281,21.6,0},
		{273,21.7,3},
		{257,20.3,1},
		{277,21.8,3},
		{257,20.4,1},
		{262,21.8,1},
		{260,20.3,1},
		{265,21.9,2},
		{259,20.4,1},
		{278,21.9,3},
		{257,20.4,1},
		{271,21.9,2},
		{256,20.5,1},
		{258,20.5,1},
		{268,22,2},
		{268,20.5,2},
		{277,22.1,3},
		{249,20.5,1},
		{267,22.1,2},
		{253,20.5,1},
		{268,22.1,2},
		{259,20.5,1},
		{266,22.1,2},
		{267,22.1,2},
		{266,20.5,2},
		{282,22.2,0},
		{258,20.6,1},
		{271,22.2,2},
		{258,20.5,1},
		{274,22.1,3},
		{258,20.6,1},
		{269,22.2,2},
		{265,20.6,2},
		{275,22.2,3},
		{255,20.6,1},
		{266,22.2,2},
		{252,20.6,1},
		{268,22.2,2},
		{256,20.5,1},
		{277,22.2,3},
		{263,20.6,2},
		{281,22.2,0},
		{263,20.6,2},
		{267,22.3,2},
		{253,20.7,1},
		{271,22.3,2},
		{258,20.7,1},
		{272,22.3,3},
		{261,20.7,1},
		{275,22.3,3},
		{263,22.4,2},
		{260,20.7,1},
		{276,22.4,3},
		{260,20.8,1},
		{273,22.5,3},
		{264,20.7,2},
		{283,22.5,0},
		{263,20.7,2},
		{266,22.5,2},
		{265,20.7,2},
		{265,22.4,2},
		{278,22.5,3},
		{264,22.6,2},
		{264,22.7,2},
		{263,22.7,2},
		{254,20.8,1},
		{267,22.7,2},
		{273,22.7,3},
		{266,22.8,2},
		{266,22.7,2},
		{266,20.9,2},
		{255,22.7,1},
		{253,21,1},
		{268,22.7,2},
		{267,21,2},
		{270,22.7,2},
		{279,22.7,3},
		{267,21,2},
		{271,22.7,2},
		{252,20.9,1},
		{262,22.6,1},
		{261,22.7,1},
		{278,22.6,3},
		{280,22.6,3},
		{267,22.6,2},
		{272,22.6,3},
		{258,20.9,1},
		{277,22.6,3},
		{274,22.5,3},
		{276,22.5,3},
		{267,20.8,2},
		{262,22.4,1},
		{257,20.8,1},
		{277,22.4,3},
		{268,22.3,2},
		{270,22.3,2},
		{267,20.7,2},
		{274,22.2,3},
		{267,20.7,2},
		{279,22.2,3},
		{260,20.6,1},
		{271,22.2,2},
		{260,20.6,1},
		{271,22.2,2},
		{257,20.5,1},
		{262,20.6,1},
		{267,22.1,2},
		{263,20.6,2},
		{261,22.1,1},
		{260,20.6,1},
		{273,22.2,3},
		{271,22.2,2},
		{271,20.6,2},
		{279,22.1,3},
		{258,20.5,1},
		{266,22.1,2},
		{276,22.1,3},
		{261,20.5,1},
		{256,20.5,1},
		{268,22.1,2},
		{252,20.5,1},
		{269,22.1,2},
		{256,20.5,1},
		{273,22.1,3},
		{265,20.5,2},
		{273,22,3},
		{261,20.4,1},
		{274,22.1,3},
		{263,20.5,2},
		{277,22.1,3},
		{256,20.5,1},
		{258,22.1,1},
		{268,20.4,2},
		{265,22.1,2},
		{261,20.4,1},
		{289,22.1,0},
		{259,20.4,1},
		{285,22.1,0},
		{280,22.1,3},
		{259,20.4,1},
		{268,22.1,2},
		{259,20.4,1},
		{281,22.1,0},
		{256,20.4,1},
		{266,22.1,2},
		{257,20.5,1},
		{277,22.2,3},
		{262,20.5,1},
		{275,22.2,3},
		{258,20.5,1},
		{283,22.2,0},
		{269,20.4,2},
		{281,22.2,0},
		{257,20.4,1},
		{269,22.2,2},
		{250,20.4,1},
		{269,22.2,2},
		{257,20.5,1},
		{275,22.2,3},
		{261,20.5,1},
		{274,22.2,3},
		{258,20.5,1},
		{273,22.2,3},
		{270,20.5,2},
		{268,22.2,2},
		{277,22.2,3},
		{264,20.4,2},
		{272,22.2,3},
		{269,20.3,2},
		{266,22.2,2},
		{281,22.2,0},
		{257,20.3,1},
		{275,22.1,3},
		{245,20.3,1},
		{279,22.1,3},
		{275,20.3,3},
		{273,22.1,3},
		{255,20.3,1},
		{268,22.1,2},
		{267,20.3,2},
		{261,22.1,1},
		{261,20.3,1},
		{281,22.1,0},
		{265,20.3,2},
		{284,22.1,0},
		{285,22.1,0},
		{260,20.3,1},
		{256,20.4,1},
		{276,22.1,3},
		{274,22.1,3},
		{257,20.4,1},
		{250,22.2,1},
		{268,22.2,2},
		{263,20.4,2},
		{262,22.2,1},
		{281,22.2,0},
		{271,20.4,2},
		{285,22.3,0},
		{253,20.4,1},
		{279,22.3,3},
		{281,20.4,0},
		{274,22.3,3},
		{257,20.4,1},
		{270,22.3,2},
		{261,22.3,1},
		{272,20.4,3},
		{272,22.3,3},
		{252,20.4,1},
		{263,22.3,2},
		{271,22.3,2},
		{258,20.4,1},
		{275,22.4,3},
		{267,20.4,2},
		{269,22.4,2},
		{277,22.4,3},
		{265,20.4,2},
		{266,20.4,2},
		{263,22.4,2},
		{276,22.4,3},
		{267,22.3,2},
		{269,20.4,2},
		{273,22.3,3},
		{258,20.4,1},
		{271,22.4,2},
		{264,20.3,2},
		{284,22.4,0},
		{277,20.4,3},
		{268,22.4,2},
		{259,20.4,1},
		{264,20.4,2},
		{265,22.3,2},
		{259,20.4,1},
		{274,22.3,3},
		{267,22.3,2},
		{271,20.4,2},
		{277,22.3,3},
		{280,22.3,3},
		{251,22.3,1},
		{285,22.3,0},
		{266,20.4,2},
		{269,22.3,2},
		{278,22.3,3},
		{275,20.4,3},
		{267,22.3,2},
		{259,20.4,1},
		{257,22.2,1},
		{269,20.4,2},
		{282,22.3,0},
		{272,22.2,3},
		{271,22.3,2},
		{279,22.2,3},
		{273,22.3,3},
		{267,20.3,2},
		{265,22.2,2},
		{261,22.2,1},
		{278,20.3,3},
		{258,20.2,1},
		{263,22.2,2},
		{264,20.2,2},
		{265,22.2,2},
		{264,20.3,2},
		{278,22.2,3},
		{273,20.2,3},
		{275,22.2,3},
		{250,20.2,1},
		{272,22.1,3},
		{290,22.1,0},
		{265,20.2,2},
		{261,22.1,1},
		{266,20.2,2},
		{270,22.2,2},
		{265,20.2,2},
		{266,22.1,2},
		{255,20.2,1},
		{261,22.1,1},
		{269,20.2,2},
		{273,22.1,3},
		{268,20.2,2},
		{271,22.1,2},
		{281,20.2,0},
		{267,22.1,2},
		{273,20.1,3},
		{281,22.1,0},
		{269,20.1,2},
		{281,22.2,0},
		{265,20.1,2},
		{257,22.1,1},
		{262,20.1,1},
		{273,22.1,3},
		{264,20.1,2},
		{276,20.1,3},
		{279,22.1,3},
		{264,20.1,2},
		{278,22.2,3},
		{272,20.1,3},
		{271,22.1,2},
		{267,20.1,2},
		{276,22.1,3},
		{257,20.1,1},
		{264,22.1,2},
		{270,20.1,2},
		{247,22.2,1},
		{264,20.1,2},
		{280,22.1,3},
		{276,20.1,3},
		{274,22.2,3},
		{281,22.1,0},
		{271,20.1,2},
		{267,22.2,2},
		{283,20.1,0},
		{271,22.1,2},
		{274,22.1,3},
		{258,20.1,1},
		{265,22.1,2},
		{271,22.2,2},
		{268,20.1,2},
		{269,22.1,2},
		{267,22.2,2},
		{272,20.1,3},
		{271,22.2,2},
		{280,22.1,3},
		{268,20,2},
		{275,22.2,3},
		{272,22.2,3},
		{277,22.1,3},
		{270,20,2},
		{273,22.1,3},
		{285,19.9,0},
		{264,22.2,2},
		{268,20,2},
		{273,22.1,3},
		{261,22.1,1},
		{273,19.9,3},
		{276,22.1,3},
		{281,22.1,0},
		{279,20,3},
		{271,22.2,2},
		{268,20,2},
		{273,22.2,3},
		{271,20.1,2},
		{270,22.2,2},
		{279,20,3},
		{266,22.2,2},
		{278,22.2,3},
		{265,22.1,2},
		{279,20,3},
		{276,22.2,3},
		{274,20.1,3},
		{271,22.2,2},
		{274,20,3},
		{268,22.3,2},
		{272,20.1,3},
		{270,22.2,2},
		{271,22.3,2},
		{278,20,3},
		{270,22.4,2},
		{264,20.1,2},
		{263,20.1,2},
		{258,22.4,1},
		{271,20.1,2},
		{260,20.1,1},
		{259,22.4,1},
		{259,22.5,1},
		{272,20.2,3},
		{267,22.5,2},
		{274,22.5,3},
		{269,22.5,2},
		{275,22.6,3},
		{270,22.7,2},
		{280,20.2,3},
		{262,22.6,1},
		{274,20.3,3},
		{272,22.6,3},
		{263,20.2,2},
		{258,22.6,1},
		{275,20.3,3},
		{262,22.6,1},
		{268,20.2,2},
		{272,22.6,3},
		{262,22.6,1},
		{277,22.6,3},
		{264,20.3,2},
		{281,22.6,0},
		{273,22.6,3},
		{277,22.6,3},
		{263,20.3,2},
		{267,20.3,2},
		{292,22.5,0},
		{254,20.3,1},
		{269,22.5,2},
		{273,20.3,3},
		{265,22.6,2},
		{274,20.3,3},
		{272,22.5,3},
		{268,20.4,2},
		{277,22.5,3},
		{272,20.4,3},
		{276,22.5,3},
		{267,20.4,2},
		{289,22.5,0},
		{267,20.4,2},
		{274,22.5,3},
		{272,20.4,3},
		{270,20.4,2},
		{270,22.5,2},
		{276,20.4,3},
		{278,22.5,3},
		{277,20.4,3},
		{280,22.5,3},
		{275,20.4,3},
		{268,22.5,2},
		{265,20.4,2},
		{278,22.5,3},
		{266,20.4,2},
		{272,22.5,3},
		{264,20.4,2},
		{286,22.5,0},
		{275,20.3,3},
		{278,22.5,3},
		{269,20.3,2},
		{262,22.5,1},
		{274,20.3,3},
		{266,22.5,2},
		{270,20.3,2},
		{289,22.5,0},
		{281,20.4,0},
		{279,22.5,3},
		{258,20.3,1},
		{283,22.5,0},
		{269,20.4,2},
		{265,22.5,2},
		{282,20.4,0},
		{275,22.5,3},
		{270,22.5,2},
		{285,22.5,0},
		{274,22.5,3},
		{263,20.4,2},
		{275,22.5,3},
		{267,20.3,2},
		{298,22.5,0},
		{266,20.3,2},
		{278,22.5,3},
		{277,22.5,3},
		{261,20.3,1},
		{273,22.5,3},
		{270,20.3,2},
		{282,22.5,0},
		{277,20.3,3},
		{274,22.5,3},
		{267,20.3,2},
		{284,22.5,0},
		{267,20.3,2},
		{294,22.5,0},
		{267,20.3,2},
		{291,22.5,0},
		{286,22.5,0},
		{279,20.3,3},
		{275,22.5,3},
		{267,20.3,2},
		{279,22.5,3},
		{282,22.5,0},
		{283,20.3,0},
		{299,22.5,0},
		{286,20.3,0},
		{287,20.2,0},
		{285,22.5,0},
		{277,20.3,3},
		{284,22.5,0},
		{264,20.3,2},
		{266,22.5,2},
		{277,20.3,3},
		{284,22.5,0},
		{285,20.3,0},
		{286,22.5,0},
		{283,20.3,0},
		{286,22.5,0},
		{267,20.3,2},
		{289,22.5,0},
		{265,20.3,2},
		{278,22.5,3},
		{277,20.3,3},
		{270,20.3,2},
		{284,22.5,0},
		{283,22.5,0},
		{261,20.3,1},
		{274,22.5,3},
		{268,20.2,2},
		{286,22.4,0}
		};

    Node* root = buildTree(dataset, 1, 3);

    printTree(root, 0);

    for (auto data : dataset) {
        double pred = predict(root, data);
        cout << "pred: " << pred << ", gt: " << data[data.size()-1] << endl;
    }
}

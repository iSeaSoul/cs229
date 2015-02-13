#ifndef _DESICION_TREE_H_
#define _DESICION_TREE_H_

#include <stdio.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <memory>

#define LOG(level, fmt, arg...) \
    printf("[" #level "] [%s:%d] " fmt"\n",\
    __FILE__, __LINE__, ##arg); \
 
#define LOG_WARNING(fmt, arg...) LOG(WARNING, fmt, ##arg)
#define LOG_NOTICE(fmt, arg...) LOG(NOTICE, fmt, ##arg)
#define LOG_FATAL(fmt, arg...) LOG(FATAL, fmt, ##arg)
 
#define ASSERT(cond, fmt, arg...) do { \
    if (!(cond)) { \
        LOG_FATAL("assert failed, condition[ " #cond " ], " fmt, ##arg); \
        return -1; \
    } \
} while(0)

template <class T = double>
void print_vector(const std::vector<T> v) {
    std::cout << "{ ";
    for_each(v.begin(), v.end(), [](T x) {
        std::cout << x << " ";
    });
    std::cout << "}\n";
}

template <class T = double>
void print_matrix(const std::vector<std::vector<T> > v) {
    std::cout << "{\n";
    for_each(v.begin(), v.end(), [](std::vector<T> x) {
        std::cout << "  ";
        print_vector(x);
    });
    std::cout << "}\n";
}

template <class T = int>
void print_map(const std::unordered_map<T, int>& m) {
    std::cout << "{ ";
    for_each(m.begin(), m.end(), [](const std::pair<T, int>& x) {
        std::cout << x.first << "-" << x.second << " ";
    });
    std::cout << "}\n";
}

template <class T>
struct desicion_node_t {
    int row_id;
    std::unordered_map<T, std::shared_ptr<desicion_node_t> > children;
    std::unordered_map<T, int> result; // only for leaf
};

struct food_t {
    std::vector<int> data;
    int result;
};

double entropy(const std::unordered_map<int, int>& food_results) {
    int tot_count = 0;
    for (const auto& x : food_results) {
        tot_count += x.second;
    }
    double ret = 0.0;
    for (const auto& x : food_results) {
        double prob = 1.0 * x.second / tot_count;
        ret -= log2(prob) * prob;
    }
    return ret;
}

template <class T>
class DesicionTree {
public:
    DesicionTree(std::function<double(std::unordered_map<int, int>)> func) {
        set_score_func(func);
        _root = nullptr;
    }

    void add_training_data(food_t food) {
        _training_data.push_back(food);
    }

    std::shared_ptr<desicion_node_t<T> > build_tree(const std::vector<int>& training_data_IDs,
            std::vector<bool> used_column) {
        double base_score = _scroe_func(unique_count_result(training_data_IDs));
        double best_gain = 0;
        int best_split_column_idx = -1;

        size_t column_cnt = _training_data[0].data.size();
        for (size_t idx = 0; idx < column_cnt; ++idx) {
            if (!used_column[idx]) {
                double temp_gain = base_score - calc_split_score(idx, training_data_IDs);
                if (temp_gain > best_gain) {
                    best_gain = temp_gain;
                    best_split_column_idx = idx;
                }
            }
        }
        auto ret_node = std::make_shared<desicion_node_t<T> >();
        if (best_split_column_idx == -1) {
            ret_node->result = unique_count_result(training_data_IDs);
            return ret_node;
        }
        ret_node->row_id = best_split_column_idx;
        used_column[best_split_column_idx] = true;
        for (const auto& x : unique_count(best_split_column_idx, training_data_IDs)) {
            ret_node->children.insert({x, build_tree(
                    split_data(best_split_column_idx, x, training_data_IDs), 
                    used_column)});
        }
        return ret_node;
    }

    std::unordered_map<T, int> query_rec(const std::vector<int>& data, 
            std::shared_ptr<desicion_node_t<T> > node) {
        if (!node->result.empty()) {
            return node->result;
        }
        if (node->children.find(data[node->row_id]) == node->children.end()) {
            return std::unordered_map<T, int>{{T(0), 0}};
        }
        return query_rec(data, node->children[data[node->row_id]]);
    }

    void train() {
        size_t training_set_size = _training_data.size();
        std::vector<int> training_data_IDs;
        for (size_t idx = 0; idx < training_set_size; ++idx) {
            training_data_IDs.push_back(idx);
        }
        std::vector<bool> used_column(training_set_size, false);
        _root = build_tree(training_data_IDs, used_column);
    }

    std::unordered_map<T, int> query(std::vector<int> data) {
        return query_rec(data, _root);
    }

    int serialization();
    int deserialization();

    void print() {
        print_rec_with_tab(_root, 0);
        printf("==========\n");
    }

    void print_rec_with_tab(std::shared_ptr<desicion_node_t<T> > node, int tab_num) {
        static auto print_indent = [](int tab_num) {
            for (int i = 0; i < tab_num; ++i) printf("  ");
        };
        if (!node->result.empty()) {
            print_map(node->result);
            return;
        }
        printf("%d:\n", node->row_id);
        for (const auto& x : node->children) {
            print_indent(tab_num);
            printf("%d-> ", x.first);
            print_rec_with_tab(x.second, tab_num + 1);
        }
    }

    void set_score_func(std::function<double(std::unordered_map<int, int>)> func) {
        _scroe_func = func;
    }

private:
    double calc_split_score(int col_id, const std::vector<int>& training_data_IDs) {
        double score = 0.0;
        for (const auto& x : unique_count(col_id, training_data_IDs)) {
            score -= _scroe_func(unique_count_result(
                    split_data(col_id, x, training_data_IDs)));
        }
        return score;
    }

    // count value field in map
    std::unordered_map<int, int> unique_count_result(const std::vector<int>& training_data_IDs) {
        std::unordered_map<int, int> counter;
        for (const int& id : training_data_IDs) {
            counter[_training_data[id].result] += 1;
        }
        return std::move(counter);
    }

    // count value field in map
    template <class NT = T>
    std::unordered_set<NT> unique_count(int col_id, const std::vector<int>& training_data_IDs) {
        std::unordered_set<NT> counter;
        for (const int& id : training_data_IDs) {
            counter.insert(_training_data[id].data[col_id]);
        }
        return std::move(counter);
    }

    // split training set by <col_id, value>
    std::vector<int> split_data(int col_id, T value, const std::vector<int>& training_data_IDs) {
        std::vector<int> splited_data_IDs;
        for (const int& id : training_data_IDs) {
            if (_training_data[id].data[col_id] == value) {
                splited_data_IDs.push_back(id);
            }
        }
        return std::move(splited_data_IDs);
    }

    std::shared_ptr<desicion_node_t<T> > _root;
    std::vector<food_t> _training_data;

    std::function<double(std::unordered_map<int, int>)> _scroe_func;
};

#endif // _DESICION_TREE_H_

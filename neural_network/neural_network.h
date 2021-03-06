#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include <stdio.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <random>

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

inline std::default_random_engine& local_random_engine() {
    static std::default_random_engine engine;
    return engine;
}

template <class T>
inline std::uniform_real_distribution<T>& local_uniform_real_distribution() {
    static std::uniform_real_distribution<T> distr;
    return distr;
}

inline double real_random_generator() {
    return local_uniform_real_distribution<double>()(local_random_engine());
}

const double INITIAL_RANGE = 0.001;

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

template <class T = double>
void matrix_mul(const std::vector<T>& a, const std::vector<std::vector<T> >& b, 
        std::vector<T>& ret, std::function<T(T)> _activation_func) {
    size_t row_cnt = b.size();
    size_t col_cnt = b[0].size();
    for (size_t idx = 0; idx < col_cnt; ++idx) {
        T sum(0);
        for (size_t ridx = 0; ridx < row_cnt; ++ridx) {
            sum += (ridx == row_cnt - 1? 1 : a[ridx]) * b[ridx][idx];
        }
        ret[idx] = _activation_func(sum);
    }
}

struct food_t {
    std::vector<int> input;
    std::vector<int> output;
};

template <class T = double>
class NeuralNetwork {
public:
    NeuralNetwork(std::function<T(T)> func_a, std::function<T(T)> func_b) {
        set_activation_func(func_a);
        set_deactivation_func(func_b);
        clear_training_data();
    }

    virtual ~NeuralNetwork() {

    }

    void create(int input_num, std::vector<int> hidden_num, int output_num) {
        std::vector<int> nn_layer_num{input_num};
        for (const int num : hidden_num) {
            nn_layer_num.push_back(num);
        }
        nn_layer_num.push_back(output_num);

        _layer_cnt = nn_layer_num.size();
        _layered_graph.resize(_layer_cnt - 1);
        _delta_graph.resize(_layer_cnt - 1);
        _layer_act.resize(_layer_cnt);
        _delta_act.resize(_layer_cnt);

        for (int layer = 0; layer < _layer_cnt; ++layer) {
            _layer_act[layer].resize(nn_layer_num[layer]);
            _delta_act[layer].resize(nn_layer_num[layer]);
            if (layer == _layer_cnt - 1) { // output layer
                break;
            }
            _layered_graph[layer].resize(nn_layer_num[layer] + 1);  // +1 : bias unit
            _delta_graph[layer].resize(nn_layer_num[layer] + 1);
            for (int layer_id = 0; layer_id <= nn_layer_num[layer]; ++layer_id) {
                _layered_graph[layer][layer_id].resize(nn_layer_num[layer + 1]);
                _delta_graph[layer][layer_id].resize(nn_layer_num[layer + 1]);
                for (int i = 0; i < nn_layer_num[layer + 1]; ++i) {
                    _layered_graph[layer][layer_id][i] =
                        (real_random_generator() * 2 - 1) * INITIAL_RANGE;
                }
            }
        }
    }

    void train_single(std::vector<T> input, std::vector<T> output) {
        _layer_act[0] = std::move(input);
        forward();
        back_propagate(std::move(output));
    }

    void clear_training_data() {
        _training_data.clear();
    }

    void add_training_data(std::vector<int> input, std::vector<int> output) {
        _training_data.push_back(food_t{input, output});
    }

    int train() {
        random_shuffle(_training_data.begin(), _training_data.end());
        for (int layer_id = 0; layer_id < _layer_cnt - 1; ++layer_id) {
            for (int i = 0; i <= _layer_act[layer_id].size(); ++i) {
                fill(_delta_graph[layer_id][i].begin(), _delta_graph[layer_id][i].end(), 0);
            }
        }
        for (const food_t& food : _training_data) {
            auto convert_food = [](std::vector<int> v, size_t sz) -> std::vector<T> {
                std::vector<T> ret(sz, T(0));
                for (const int& i : v) {
                    ret[i] = T(1);
                }
                return ret;
            };
            train_single(convert_food(food.input, _layer_act[0].size()), 
                    convert_food(food.output, _layer_act[_layer_cnt - 1].size()));
        }
        for (int layer_id = 0; layer_id < _layer_cnt - 1; ++layer_id) {
            for (int i = 0; i <= _layer_act[layer_id].size(); ++i) {
                for (int j = 0; j < _layer_act[layer_id + 1].size(); ++j) {
                    _layered_graph[layer_id][i][j] += 
                        _learning_rate * _delta_graph[layer_id][i][j];
                }
            }
        }
        return 0;
    }

    void query(std::vector<int> input, std::vector<T>& output) {
        convert_input(input);
        forward();
        output = _layer_act[_layer_cnt - 1];
    }

    void convert_input(std::vector<int> input) {
        fill(_layer_act[0].begin(), _layer_act[0].end(), 0);
        for (const int& i : input) {
            _layer_act[0][i] = 1.0;
        }
    }

    void forward() {
        for (int i = 0; i < _layer_cnt - 1; ++i) {
            matrix_mul(_layer_act[i], _layered_graph[i], _layer_act[i + 1], _activation_func);
        }
    }

    void back_propagate(std::vector<T> output) {
        for (int j = 0; j < _layer_act[_layer_cnt - 1].size(); ++j) {
            T error = output[j] - _layer_act[_layer_cnt - 1][j];
            _delta_act[_layer_cnt - 1][j] = 
                _deactivation_func(_layer_act[_layer_cnt - 1][j]) * error;
        }
        for (int layer_id = _layer_cnt - 2; layer_id >= 1; --layer_id) {
            for (int i = 0; i < _layer_act[layer_id].size(); ++i) {
                T error(0);
                for (int j = 0; j < _layer_act[layer_id + 1].size(); ++j) {
                    error += _delta_act[layer_id + 1][j] * _layered_graph[layer_id][i][j];
                }
                _delta_act[layer_id][i] = _deactivation_func(_layer_act[layer_id][i]) * error;
            }
        }
        for (int layer_id = 0; layer_id < _layer_cnt - 1; ++layer_id) {
            for (int i = 0; i <= _layer_act[layer_id].size(); ++i) {
                for (int j = 0; j < _layer_act[layer_id + 1].size(); ++j) {
                    _delta_graph[layer_id][i][j] += 
                        (i == _layer_act[layer_id].size()?
                            1 : _layer_act[layer_id][i]) * _delta_act[layer_id + 1][j];
                }
            }
        }
    }

    int serialization();
    int deserialization();

    void print() {
        for (int layer_id = 0; layer_id < _layer_cnt - 1; ++layer_id) {
            print_matrix(_layered_graph[layer_id]);
        }
    }

    void set_activation_func(std::function<T(T)> func) {
        _activation_func = func;
    }
    void set_deactivation_func(std::function<T(T)> func) {
        _deactivation_func = func;
    }

private:
    std::vector<std::vector<std::vector<T> > > _layered_graph;
    std::vector<std::vector<std::vector<T> > > _delta_graph;
    std::vector<std::vector<T> > _layer_act;
    std::vector<std::vector<T> > _delta_act;
    std::vector<food_t> _training_data;

    int _layer_cnt;
    std::function<T(T)> _activation_func;
    std::function<T(T)> _deactivation_func;

    double _learning_rate = 0.5;
};

#endif // _NEURAL_NETWORK_H_

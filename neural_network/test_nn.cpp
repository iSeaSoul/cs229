#include "neural_network.h"

void test_query(NeuralNetwork<double> nn) {
    std::vector<double> ret;
    nn.query(std::vector<int>{0}, ret);
    std::cout << "0 - "; print_vector(ret);
    nn.query(std::vector<int>{1}, ret);
    std::cout << "1 - "; print_vector(ret);
    nn.query(std::vector<int>{2}, ret);
    std::cout << "2 - "; print_vector(ret);
}

int main() {
    NeuralNetwork<double> nn(
        [](double x) { return 1.0 / (1.0 + exp(-x)); }, 
        [](double x) { return x * (1 - x); }
    );

    nn.create(3, std::vector<int>{3}, 2);
    test_query(nn);

    nn.add_training_data(std::vector<int>{0}, std::vector<int>{0});
    nn.add_training_data(std::vector<int>{0, 1}, std::vector<int>{0});
    nn.add_training_data(std::vector<int>{1, 2}, std::vector<int>{1});
    nn.add_training_data(std::vector<int>{2}, std::vector<int>{1});
    nn.add_training_data(std::vector<int>{0, 1, 2}, std::vector<int>{1});

    for (int tms = 0; tms < 10000; ++tms)   {
        nn.train();
        if (tms % 1000 == 0) {
            // nn.print();
            test_query(nn);
        }
    }

    return 0;
}

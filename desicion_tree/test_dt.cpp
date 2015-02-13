#include "desicion_tree.h"

int main() {
    freopen ("dt_food.in", "r", stdin);

    DesicionTree<int> dt(entropy);

    int line_num, row_num;
    std::cin >> line_num >> row_num;
    for (int i = 0; i < line_num; ++i) {
        std::vector<int> vec;
        int bar;
        for (int j = 0; j < row_num; ++j) {
            std::cin >> bar;
            vec.push_back(bar);
        }
        std::cin >> bar;
        dt.add_training_data(food_t{vec, bar});
    }

    dt.train();
    dt.print();

    print_map(dt.query(std::vector<int>{1, 0, 0, 0}));
    print_map(dt.query(std::vector<int>{0, 1, 0, 0}));
    print_map(dt.query(std::vector<int>{0, 2, 0, 0}));
    print_map(dt.query(std::vector<int>{0, 0, 2, 0}));
    print_map(dt.query(std::vector<int>{0, 0, 0, 1}));

    return 0;
}
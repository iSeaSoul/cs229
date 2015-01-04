import random

def get_random(L, R):
    return int(random.random() * (R - L + 1)) + L

def gen_data():
    for i in xrange(1000):
        x = 1 + get_random(0, 2000) / 1000.0
        y = 1 + get_random(0, 2000) / 1000.0
        ret = 0 if x >= y else 1
        if abs(x - y) < 0.2 and get_random(1, 5) == 1:
            ret = 1 - ret
        print x, y, ret

gen_data()
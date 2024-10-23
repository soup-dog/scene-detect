from scene_detect import *

if __name__ == '__main__':
    assert posterise(0, 0, 0) == 0
    assert 0b11111000 & 1 == 0
    assert 0b11111000 & 0b1000 == 0b1000
    assert 0b11111000 & 0b100 == 0
    assert posterise(1, 1, 1) == 0

    assert try_parse_vector("") is None
    assert try_parse_vector(":") is None
    assert try_parse_vector("1") is None
    assert try_parse_vector("1:0") == (1, 0)
    assert try_parse_vector(":1") == (-1, 1)


def parse_int_tuple(input):
    return tuple(int(k.strip()) for k in input[1:-1].split(','))

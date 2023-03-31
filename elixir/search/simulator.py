import math

from .utils import to_divide


class LinkedSet(object):

    def __init__(self) -> None:
        super().__init__()
        self.fifo_dict = dict()

    def __len__(self) -> int:
        return len(self.fifo_dict)

    def __contains__(self, x: int) -> bool:
        return x in self.fifo_dict

    def full(self, n: int):
        return len(self.fifo_dict) >= n

    def push(self, x: int):
        assert x not in self.fifo_dict
        self.fifo_dict[x] = True

    def pop_value(self, x: int):
        assert x in self.fifo_dict
        self.fifo_dict.pop(x)

    def pop_left(self):
        x = next(iter(self.fifo_dict))
        self.fifo_dict.pop(x)


def calc_move_times(param_per_step: list, param_to_chunk: dict, n_blocks: int):
    chunk_per_step = list()

    for param_set in param_per_step:
        id_set = set()
        for name in param_set:
            # continue if the parameter is ignored
            if name not in param_to_chunk:
                continue
            id_set.add(param_to_chunk[name])
        if len(id_set) > 0:
            chunk_per_step.append(id_set)

    offload_time = 0
    upload_time = 0
    chunks_in_rcache = LinkedSet()
    for chunk_set in reversed(chunk_per_step):

        # return inf if there is no enough chunks
        if len(chunk_set) > n_blocks:
            return float('inf')

        for chunk_id in chunk_set:
            if chunk_id in chunks_in_rcache:
                # pop this chunk out
                chunks_in_rcache.pop_value(chunk_id)
                # append this chunk to the tail, since it is used the most recently
                chunks_in_rcache.push(chunk_id)
            else:
                if chunks_in_rcache.full(n_blocks):
                    # pop the least recently used chunk
                    chunks_in_rcache.pop_left()
                    # this evicted chunks should be uploaded before
                    upload_time += 1
                # append this chunk to the tail, since it is used the most recently
                chunks_in_rcache.push(chunk_id)
                # this chunk will be offloaded
                offload_time += 1

    upload_time += len(chunks_in_rcache)
    assert upload_time == offload_time
    return (offload_time << 1)


def find_optimal_chunk_size(
    # pre-commit: do not rearrange
        param_per_step: list,
        param_names: list,
        param_numels: list,
        cuda_elements: int,
        overlap: bool,
        min_range: int,
        max_range: int,
        interval: int):

    max_numel = 0
    for numel in param_numels:
        max_numel = max(max_numel, numel)
    test_size = to_divide(max(max_numel, min_range), interval)
    # floor rounding
    cuda_elements = to_divide(cuda_elements - interval + 1, interval)
    max_range = min(max_range, cuda_elements)

    min_move_elements = float('+inf')
    best_size = test_size
    best_number_blocks = 0
    best_waste = 0

    def dispatch_chunks(param_to_chunk: dict, block_size: int) -> int:
        chunk_id = 0
        acc = 0
        left = 0
        for (name, numel) in zip(param_names, param_numels):
            if numel > left:
                acc += left
                chunk_id += 1
                left = block_size
            left -= numel
            param_to_chunk[name] = chunk_id
        return (chunk_id, left + acc)

    assert test_size <= max_range, 'max_numel or min_range is larger than max_range or cuda capacity'
    while test_size <= max_range:
        # calculate the number of blocks
        number_blocks = int(cuda_elements // test_size)
        # if prefetch is enabled, we pretend that two chunks are reserved
        if overlap:
            number_blocks -= 2
        if number_blocks <= 0:
            continue
        # initialize the chunk id for each parameter
        param_to_chunk = dict()
        number_chunks, current_waste = dispatch_chunks(param_to_chunk, test_size)
        number_blocks = min(number_blocks, number_chunks)
        # calculate the minimum number of movements
        move_times = calc_move_times(param_per_step, param_to_chunk, number_blocks)

        current_move_elements = move_times * test_size
        # print("test", test_size, current_move_elements)
        if current_move_elements < min_move_elements:
            min_move_elements = current_move_elements
            best_size = test_size
            best_number_blocks = number_blocks
            best_waste = current_waste

        test_size += interval

    if min_move_elements == float('inf'):
        raise RuntimeError('optimal search: can not find a valid solution')

    return best_size, best_number_blocks, best_waste


def bandwidth_c2g(n: int):
    return 16.3 * n + 8.7


def bandwidth_g2c(n: int):
    return 15.8 * n + 2.3


def velocity_gpu(n: int):
    return 50 * n


def velocity_cpu(n: int):
    return 1.66 * math.log(n) + 5.15


def rcache_prioirity_check(n: int, r_os: int, e_p: int, e_o: int):
    rcache_save = (r_os) / n * e_p * (1.0 / bandwidth_c2g(n) + 1.0 / bandwidth_g2c(n))
    gpu_optim_save = e_o / bandwidth_c2g(n) + e_p / bandwidth_g2c(n) + 1.0 / velocity_cpu(n) - 1.0 / velocity_gpu(n)
    print(rcache_save, gpu_optim_save)
    return rcache_save > gpu_optim_save

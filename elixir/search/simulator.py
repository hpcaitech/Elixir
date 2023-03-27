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


def calc_replace_time(param_per_step: list, param_to_chunk: dict, n_chunks: int):
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
        if len(chunk_set) > n_chunks:
            return float('inf')

        for chunk_id in chunk_set:
            if chunk_id in chunks_in_rcache:
                # pop this chunk out
                chunks_in_rcache.pop_value(chunk_id)
                # append this chunk to the tail, since it is used the most recently
                chunks_in_rcache.push(chunk_id)
            else:
                if chunks_in_rcache.full(n_chunks):
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

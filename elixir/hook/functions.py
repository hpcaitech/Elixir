import torch

from elixir.chunk import ChunkFetcher


def prefwd_postbwd_function(fetcher: ChunkFetcher):

    class PreFwdPostBwd(torch.autograd.Function):

        @staticmethod
        def forward(ctx, params, *args):
            ctx.params = params
            return args

        @staticmethod
        def backward(ctx, *grads):
            with torch._C.DisableTorchFunction():
                print('grad You know', end=' ')
                for g in grads:
                    print(torch.sum(g), end=' ')
                print('')
                fetcher.trans_to_hold(ctx.params, phase='b')
                return (None, *grads)

    def exec_func(params, *args):
        chunks = fetcher.trans_to_compute(params)
        fetcher.fetch_chunks(chunks)
        return PreFwdPostBwd.apply(params, *args)

    return exec_func


def postfwd_prebwd_function(fetcher: ChunkFetcher):

    class PostFwdPreBwd(torch.autograd.Function):

        @staticmethod
        def forward(ctx, params, *args):
            ctx.params = params
            return args

        @staticmethod
        def backward(ctx, *grads):
            with torch._C.DisableTorchFunction():
                chunks = fetcher.trans_to_compute(ctx.params)
                fetcher.fetch_chunks(chunks)
                print('grad before', end=' ')
                for g in grads:
                    print(torch.sum(g), g.data_ptr(), end=' ')
                print('')
                print('fetching', end=' ')
                for c in chunks:
                    print(c.chunk_id, c.rcb.payload.data_ptr(), end=' ')
                print('')
                print('grad after', end=' ')
                for g in grads:
                    print(torch.sum(g), end=' ')
                print('')
                return (None, *grads)

    def exec_func(params, *args):
        fetcher.trans_to_hold(params, phase='f')
        return PostFwdPreBwd.apply(params, *args)

    return exec_func

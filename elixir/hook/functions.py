import torch

from elixir.chunk import ChunkFetcher

from .storage import BufferStore


def prefwd_postbwd_function(fetcher: ChunkFetcher, store: BufferStore):

    class PreFwdPostBwd(torch.autograd.Function):

        @staticmethod
        def forward(ctx, params, *args):
            with torch._C.DisableTorchFunction():
                ctx.params = params
                chunks = fetcher.trans_to_compute(params)
                fetcher.fetch_chunks(chunks)

                offset = 0
                for p in ctx.params:
                    if not fetcher.is_in_fused(p):
                        # we should add parameters to buffer
                        # because their blocks may be changed
                        offset = store.insert(p, offset)

            return args

        @staticmethod
        def backward(ctx, *grads):
            with torch._C.DisableTorchFunction():
                print('grad You know', end=' ')
                for g in grads:
                    print(torch.sum(g), end=' ')
                print('')
                fetcher.trans_to_hold(ctx.params, phase='b')

                for p in ctx.params:
                    if not fetcher.is_in_fused(p):
                        store.erase(p)

            return (None, *grads)

    return PreFwdPostBwd.apply


def postfwd_prebwd_function(fetcher: ChunkFetcher, store: BufferStore):

    class PostFwdPreBwd(torch.autograd.Function):

        @staticmethod
        def forward(ctx, params, *args):
            with torch._C.DisableTorchFunction():
                ctx.params = params

                for p in ctx.params:
                    if not fetcher.is_in_fused(p):
                        store.erase(p)

            return args

        @staticmethod
        def backward(ctx, *grads):
            with torch._C.DisableTorchFunction():
                chunks = fetcher.trans_to_compute(ctx.params)
                fetcher.fetch_chunks(chunks)

                offset = 0
                for p in ctx.params:
                    if not fetcher.is_in_fused(p):
                        # we should add parameters to buffer
                        # because their blocks may be changed
                        offset = store.insert(p, offset)

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

    return PostFwdPreBwd.apply

import torch

from elixir.chunk import ChunkFetcher, TensorState


def prefwd_postbwd_function(fetcher: ChunkFetcher):

    class PreFwdPostBwd(torch.autograd.Function):

        @staticmethod
        def forward(ctx, params, *args):
            ctx.params = params
            chunks = fetcher.trans_to_compute(params)
            fetcher.fetch_chunks(chunks)
            return args

        @staticmethod
        def backward(ctx, *grads):
            fetcher.trans_to_hold(ctx.params, phase='b')
            return (None, *grads)

    return PreFwdPostBwd


def postfwd_prebwd_function(fetcher: ChunkFetcher):

    class PostFwdPreBwd(torch.autograd.Function):

        @staticmethod
        def forward(ctx, params, *args):
            ctx.params = params
            fetcher.trans_to_hold(params, phase='f')

        @staticmethod
        def backward(ctx, *grads):
            chunks = fetcher.trans_to_compute(ctx.params)
            fetcher.fetch_chunks(chunks)
            return (None, *grads)

    return PostFwdPreBwd

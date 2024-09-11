import torch


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """
    Dataloader which does not reset iterator at the end of each epoch.
    Improves performance when using multiple epochs with the same dataset. (~30% speedup)
    Credits: https://github.com/huggingface/pytorch-image-models/blob/4d0737d5fa9674e64cb9210c688d2dd168dbb448/timm/data/loader.py#L314C1-L346C42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return (
            len(self.sampler)
            if self.batch_sampler is None
            else len(self.batch_sampler.sampler)
        )

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

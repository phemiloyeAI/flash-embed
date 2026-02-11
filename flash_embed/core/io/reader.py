import webdataset as wds
from braceexpand import braceexpand

class WebDataSetReader:
    # a simple wrapper reader to give us IterableDatasets (only local files)
    # here we just want to read. no decoding (we want better control over that)
    def __init__(
            self, input_dataset: str, batch_size: int = 1,
            shuffle: bool = False, num_workers: int = 0,
            image_key: str = "jpg", caption_key: str = "txt"
        ):

        self.input_dataset = list(braceexpand(input_dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __read_dataset(self):
        ds = wds.WebDataset(
            urls=self.input_dataset
        )


# can I merge the decoding and dataloading into same thread/process?


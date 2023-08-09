def print_sizes(batch):
    for key in batch.keys():
        print(f'{key} : {batch[key].size()}')
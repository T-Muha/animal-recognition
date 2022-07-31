from random import sample
from ImageGenerator import ImageGenerator

def create_datasets(train_ratio, valid_ratio, test_ratio):


    all_filenames = []
    for animal in [d for d in os.listdir(datadir) if '.' not in d]:
        for img_name in os.listdir(datadir + animal):
            all_filenames.append(animal + '/' + img_name)

    full_size = len(all_filenames)
    train_size = int(full_size * train_ratio)
    valid_size = int(full_size * valid_ratio)
    test_size = int(full_size * test_ratio)

    train_filenames = sample(all_filenames, train_size)
    all_filenames = [x for x in all_filenames if x not in train_filenames]
    valid_filenames = sample(all_filenames, valid_size)
    all_filenames = [x for x in all_filenames if x not in valid_filenames]
    test_filenames = sample(all_filenames, test_size)

    train_dataset = ImageGenerator(filenames=train_filenames, name='train')
    valid_dataset = ImageGenerator(filenames=valid_filenames, name='valid')
    test_dataset = ImageGenerator(filenames=test_filenames, name='test')

    return train_dataset, valid_dataset, test_dataset
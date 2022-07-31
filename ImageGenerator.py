import os
from random import choice
import numpy as np
from PIL import Image
import time

class_encoding = {'butterfly': 0, 'cat': 1, 'chicken': 2, 'cow': 3, 'dog': 4, 'elephant': 5, 'horse': 6, 'sheep': 7, 'spider': 8, 'squirrel': 9}

class ImageGenerator():
    """
    animal_n is number of images by animal.
    n is the total number of images.
    image_tracker is a dictionary of boolean arrays of the images read for each animal. True is unread, False is read.
    file_names is the name of the image corresponding to the image tracker. It is sorted to ensure determinism between runs.
    i is the current position of the iterator.
    """    

    def __init__(self, filenames=None, image_dir='data/Animals-10/', name=None):
        self.name = name
        self.image_dir = image_dir
        self.animals = [animal for animal in os.listdir(self.image_dir) if '.' not in animal]
        # self.n_by_animal = {animal: len(os.listdir(self.image_dir + animal)) for animal in os.listdir(self.image_dir) if '.' not in animal}
        
        if filenames:
            # self.read_tracker = np.array([True for _ in range(len(filenames))])
            self.classes = np.array([class_encoding[f.split('/')[0]] for f in filenames])
            self.filenames = filenames
        else:
            NotImplementedError("Full Dataset Generator Not Yet Implemented.")
            # self.read_tracker = 
            # self.classes = 
            # self.filenames = 

        # Removing used image nums from a list is 240 times faster than iterating True/False list
        self.remaining_image_nums = [i for i in range(len(self.filenames))] 
        self.n = len(self.filenames)
        self.i = 0

        self.image_load_time = 0
        self.choice_time = 0


    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()

    def next(self):
        if self.i < self.n:

            start = time.time()
            # remaining_image_nums = [i for i, val in enumerate(self.read_tracker) if val]
            image_num = choice(self.remaining_image_nums)
            image_name = self.filenames[image_num]
            self.choice_time += (time.time() - start)
            start = time.time()
            image = Image.open(self.image_dir + image_name)
            # self.read_tracker[image_num] = False
            self.remaining_image_nums.remove(image_num)

            if not image.mode == 'RGB':
                image = image.convert('RGB')
            self.image_load_time += (time.time() - start)

            self.i += 1
            return np.array(image)

        raise StopIteration()
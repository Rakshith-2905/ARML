""" Code for loading data. """
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import pickle
from utils import get_images
import math
from sklearn.utils import shuffle
import ipdb

FLAGS = flags.FLAGS


class DataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)
        self.num_datasets = FLAGS.num_datasets

        if FLAGS.datasource == '2D':
            self.dim_input = 2
            self.dim_output = 1
            self.input_range = config.get('input_range', [-5.0, 5.0])

        elif FLAGS.datasource == 'plainmulti':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            self.plainmulti = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi', 'vgg_flower', 'GTSRB']
            # self.plainmulti = ['DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi',  'CUB_Bird', 'vgg_flower', 'GTSRB']
            
            print("\n\nDatasets trained and tested on: \n\n", [self.plainmulti[i]\
                for i in range(self.num_datasets)])
            metatrain_folders, metaval_folders = [], []
            for eachdataset in self.plainmulti:
                metatrain_folders.append(
                    [os.path.join('{0}/plainmulti/{1}/train'.format(FLAGS.datadir, eachdataset), label) \
                     for label in os.listdir('{0}/plainmulti/{1}/train'.format(FLAGS.datadir, eachdataset)) \
                     if
                     os.path.isdir(os.path.join('{0}/plainmulti/{1}/train'.format(FLAGS.datadir, eachdataset), label)) \
                     ])
                if FLAGS.test_set:
                    metaval_folders.append(
                        [os.path.join('{0}/plainmulti/{1}/test'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/plainmulti/{1}/test'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/plainmulti/{1}/test'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
                else:
                    metaval_folders.append(
                        [os.path.join('{0}/plainmulti/{1}/val'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/plainmulti/{1}/val'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/plainmulti/{1}/val'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])

        elif FLAGS.datasource == 'artmulti':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            self.artmulti = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi', 'CUB_Bird_blur',
                                'DTD_Texture_blur', 'FGVC_Aircraft_blur', 'FGVCx_Fungi_blur', 'CUB_Bird_pencil',
                                'DTD_Texture_pencil', 'FGVC_Aircraft_pencil', 'FGVCx_Fungi_pencil']
            metatrain_folders, metaval_folders = [], []
            for eachdataset in self.artmulti:
                metatrain_folders.append(
                    [os.path.join('{0}/artmulti/{1}/train'.format(FLAGS.datadir, eachdataset), label) \
                     for label in os.listdir('{0}/artmulti/{1}/train'.format(FLAGS.datadir, eachdataset)) \
                     if
                     os.path.isdir(
                         os.path.join('{0}/artmulti/{1}/train'.format(FLAGS.datadir, eachdataset), label)) \
                     ])
                if FLAGS.test_set:
                    metaval_folders.append(
                        [os.path.join('{0}/artmulti/{1}/test'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/artmulti/{1}/test'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/artmulti/{1}/test'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
                else:
                    metaval_folders.append(
                        [os.path.join('{0}/artmulti/{1}/val'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/artmulti/{1}/val'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/artmulti/{1}/val'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])

        elif FLAGS.datasource == 'domainNet':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            self.domainNet = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

            print("\n\nDatasets trained and tested on: \n\n", [self.domainNet[i]\
                for i in range(self.num_datasets)])
            # random.shuffle(self.domainNet)
            metatrain_folders, metaval_folders = [], []
            for eachdataset in self.domainNet:
                metatrain_folders.append(
                    [os.path.join('{0}/domainNet/{1}/train'.format(FLAGS.datadir, eachdataset), label) \
                     for label in os.listdir('{0}/domainNet/{1}/train'.format(FLAGS.datadir, eachdataset)) \
                     if
                     os.path.isdir(os.path.join('{0}/domainNet/{1}/train'.format(FLAGS.datadir, eachdataset), label)) \
                     ])
                if FLAGS.test_set:
                    metaval_folders.append(
                        [os.path.join('{0}/domainNet/{1}/train'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/domainNet/{1}/train'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/domainNet/{1}/train'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
                else:
                    metaval_folders.append(
                        [os.path.join('{0}/domainNet/{1}/val'.format(FLAGS.datadir, eachdataset), label) \
                         for label in os.listdir('{0}/domainNet/{1}/val'.format(FLAGS.datadir, eachdataset)) \
                         if os.path.isdir(
                            os.path.join('{0}/domainNet/{1}/val'.format(FLAGS.datadir, eachdataset), label)) \
                         ])
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])

        elif FLAGS.datasource == 'synthetic':
            num_total_batches = 200000
            self.num_classes = 2
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            self.latent_dim = 128

            if FLAGS.create_synth:
                print('Creating synthetic data')
                gauss_features = np.random.normal(size=[self.latent_dim, self.dim_input])
                                
                with open('synthetic_data/gauss_features_1.pickle', 'wb') as handle:
                    pickle.dump(gauss_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                thetas = {}
                # gauss teacher for all tasks in the training set
                theta1 = np.random.normal(loc=0.0, scale=1.0, size=[num_total_batches, self.latent_dim, 1])
                theta2 = np.random.normal(loc=1.0, scale=1.0, size=[num_total_batches, self.latent_dim, 1])
                theta3 = np.random.normal(loc=-1.0, scale=1.0, size=[num_total_batches, self.latent_dim, 1])
                    
                thetas['theta_tasks_1'] = theta1
                thetas['theta_tasks_2'] = theta2
                thetas['theta_tasks_3'] = theta3

                with open('synthetic_data/thetas.pickle', 'wb') as handle:
                    pickle.dump(thetas, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            with open('synthetic_data/gauss_features_1.pickle', 'rb') as handle:
                self.gauss_features = pickle.load(handle)
            
            with open('synthetic_data/thetas.pickle', 'rb') as handle:
                self.thetas = pickle.load(handle)
            # Repeat the gauss features to generate for the entire batch
            self.gauss_features = np.expand_dims(self.gauss_features, axis=0)
            self.gauss_features = np.repeat(
                self.gauss_features, self.batch_size*self.num_samples_per_class*5, axis=0)
            
            self.batch_count = 0

        else:
            raise ValueError('Unrecognized data source')

    def make_data_tensor_plainmulti(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = FLAGS.num_test_task
        # make list of files
        print('Generating filenames')
        all_filenames = []
        for image_itr in range(num_total_batches):
            sel = np.random.randint(self.num_datasets)
            if FLAGS.train == False and FLAGS.test_dataset != -1:
                sel = FLAGS.test_dataset
            sampled_character_folders = random.sample(folders[sel], self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes),
                                           nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource in ['plainmulti', 'artmulti']:
            image = tf.image.decode_jpeg(image_file, channels=3)
            image.set_shape((self.img_size[0], self.img_size[1], 3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0], self.img_size[1], 1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
            [image],
            batch_size=batch_image_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_image_size,
        )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i * examples_per_batch:(i + 1) * examples_per_batch]
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)
                true_idxs = class_idxs * self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch, true_idxs))
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def make_data_tensor_artmulti(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = FLAGS.num_test_task

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sel = np.random.randint(12)
            if FLAGS.train == False and FLAGS.test_dataset != -1:
                sel = FLAGS.test_dataset
            sampled_character_folders = random.sample(folders[sel], self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes),
                                           nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        # image_file=datadict[filename_queue]
        if FLAGS.datasource in ['plainmulti', 'artmulti']:
            image = tf.image.decode_jpeg(image_file, channels=3)
            # image = tf.convert_to_tensor(image_file)
            image.set_shape((self.img_size[0], self.img_size[1], 3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0], self.img_size[1], 1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
            [image],
            batch_size=batch_image_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_image_size,
        )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i * examples_per_batch:(i + 1) * examples_per_batch]
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)
                true_idxs = class_idxs * self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch, true_idxs))
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def make_data_tensor_domainNet(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = FLAGS.num_test_task
        
        # make list of files
        print('Generating filenames')
        all_filenames = []
        for image_itr in range(num_total_batches):
            sel = np.random.randint(self.num_datasets)
            if FLAGS.train == False and FLAGS.test_dataset != -1:
                sel = FLAGS.test_dataset
            
            sampled_character_folders = random.sample(folders[sel], self.num_classes)

            random.shuffle(sampled_character_folders)

            labels_and_images = get_images(sampled_character_folders, range(self.num_classes),
                                        nb_samples=self.num_samples_per_class, shuffle=False)

            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)
            
            
        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource in ['plainmulti', 'artmulti', 'domainNet']:
            image = tf.image.decode_jpeg(image_file, channels=3)
            image = tf.image.resize(image, (self.img_size[0], self.img_size[1]), preserve_aspect_ratio=False)

            # image.set_shape((self.img_size[0], self.img_size[1], 3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0], self.img_size[1], 1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
            [image],
            batch_size=batch_image_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_image_size,
        )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i * examples_per_batch:(i + 1) * examples_per_batch]
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)
                true_idxs = class_idxs * self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch, true_idxs))
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches


    def generate_2D_batch(self, train=False):
        dim_input = self.dim_input
        dim_output = self.dim_output
        batch_size = self.batch_size
        num_samples_per_class = self.num_samples_per_class

        # sin
        amp = np.random.uniform(0.1, 5.0, size=self.batch_size)
        phase = np.random.uniform(0., 2 * np.pi, size=batch_size)
        freq = np.random.uniform(0.8, 1.2, size=batch_size)

        # linear
        A = np.random.uniform(-3.0, 3.0, size=batch_size)
        b = np.random.uniform(-3.0, 3.0, size=batch_size)

        # quadratic
        A_q = np.random.uniform(-0.2, 0.2, size=batch_size)
        b_q = np.random.uniform(-2.0, 2.0, size=batch_size)
        c_q = np.random.uniform(-3.0, 3.0, size=batch_size)

        # cubic
        A_c = np.random.uniform(-0.1, 0.1, size=batch_size)
        b_c = np.random.uniform(-0.2, 0.2, size=batch_size)
        c_c = np.random.uniform(-2.0, 2.0, size=batch_size)
        d_c = np.random.uniform(-3.0, 3.0, size=batch_size)

        # 3d curve
        A_3cur = np.random.uniform(-1.0, 1.0, size=batch_size)
        B_3cur = np.random.uniform(-1.0, 1.0, size=batch_size)

        # ripple
        A_r = np.random.uniform(-0.2, 0.2, size=batch_size)
        B_r = np.random.uniform(-3.0, 3.0, size=batch_size)

        sel_set = np.zeros(batch_size)

        init_inputs = np.zeros([batch_size, num_samples_per_class, dim_input])
        outputs = np.zeros([batch_size, num_samples_per_class, dim_output])

        for func in range(batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], size=(num_samples_per_class, dim_input))
            sel = np.random.randint(FLAGS.sync_group_num)
            if FLAGS.train == False and FLAGS.test_dataset != -1:
                sel = FLAGS.test_dataset
            if sel == 0:
                init_inputs[func, :, 1:2] = 1
                outputs[func] = amp[func] * np.sin(freq[func] * init_inputs[func, :, 0:1] + phase[func])
            elif sel == 1:
                init_inputs[func, :, 1:2] = 1
                outputs[func] = A[func] * init_inputs[func, :, 0:1] + b[func]
            elif sel == 2:
                # outputs[func] = A_q[func] * np.square(init_inputs[func] - c_q[func]) + b_q[func]
                init_inputs[func, :, 1:2] = 1
                outputs[func] = A_q[func] * np.square(init_inputs[func, :, 0:1]) + b_q[func] * init_inputs[func, :, 0:1] + \
                                c_q[func]
            elif sel == 3:
                init_inputs[func, :, 1:2] = 1
                outputs[func] = A_c[func] * np.power(init_inputs[func, :, 0:1],
                                                     np.tile([3], init_inputs[func, :, 0:1].shape)) + b_c[
                                    func] * np.square(init_inputs[func, :, 0:1]) + c_c[func] * init_inputs[func, :, 0:1] + \
                                d_c[func]
            elif sel == 4:
                outputs[func] = A_3cur[func] * np.square(init_inputs[func, :, 0:1]) + B_3cur[func] * np.square(
                    init_inputs[func, :, 1:2])
            elif sel == 5:
                outputs[func] = np.sin(
                    -A_r[func] * (np.square(init_inputs[func, :, 0:1]) + np.square(init_inputs[func, :, 1:2]))) + B_r[func]
            outputs[func] += np.random.normal(0, 0.3, size=(num_samples_per_class, dim_output))
            sel_set[func] = sel
        funcs_params = {'amp': amp, 'phase': phase, 'freq': freq, 'A': A, 'b': b, 'A_q': A_q, 'c_q': c_q, 'b_q': b_q,
                        'A_c': A_c, 'b_c': b_c, 'c_c': c_c, 'd_c': d_c, 'A_3cur': A_3cur, 'B_3cur': B_3cur, 'A_r':A_r, 'B_r':B_r}
        return init_inputs, outputs, funcs_params, sel_set


    def generate_syn_batch(self, train=False):  

        # Gaussian coeff for an entire batch. (5 times the number of samples per class)
        gauss_coeff = np.random.normal(size=[self.batch_size*self.num_samples_per_class*5, 1, self.latent_dim])          

        if FLAGS.synthetic_case == 2:
            self.gauss_features = np.random.normal(size=[self.latent_dim, self.dim_input])
            self.gauss_features = np.expand_dims(self.gauss_features, axis=0)
            self.gauss_features = np.repeat(
                self.gauss_features, self.batch_size*self.num_samples_per_class*5, axis=0)

        images = np.matmul(gauss_coeff, self.gauss_features)/math.sqrt(self.latent_dim)
        # Relu function
        images[images<0] = 0

        if FLAGS.synthetic_case == 0:
            thetas = self.thetas['theta_tasks_1'][
                self.batch_size*self.batch_count: self.batch_size*(self.batch_count+1)]
        else:
            dist_choice = str(np.random.randint(1,4))
            thetas = self.thetas['theta_tasks_'+dist_choice][
                self.batch_size*self.batch_count: self.batch_size*(self.batch_count+1)]

        self.batch_count += 1
        # Repeat the gauss features to generate for the entire batch
        thetas = np.expand_dims(thetas, axis=0)
        thetas = np.repeat(thetas, self.num_samples_per_class*5, axis=0)
        thetas = thetas.reshape([-1,128,1])
            
        # Create labels with gauss teacher function to be {0,1}
        labels = np.sign(np.matmul(gauss_coeff, thetas)/math.sqrt(self.latent_dim)).astype(int)
        labels = labels[:,0,0]
        labels[labels<0] = 0

        # Extract the indices of the two classes
        cls_a_indices = np.where(labels==0)
        cls_b_indices = np.where(labels==1)

        # Select the desired number of samples for each class
        labels_a = labels[cls_a_indices]
        labels_b = labels[cls_b_indices]

        labels_a = labels_a[0:self.batch_size*self.num_samples_per_class].reshape(
            self.batch_size, -1,1)
        labels_b = labels_b[0:self.batch_size*self.num_samples_per_class].reshape(
            self.batch_size, -1,1)
        
        images_a = images[cls_a_indices]
        images_b = images[cls_b_indices]

        images_a = images_a[0:self.batch_size*self.num_samples_per_class, :, :].reshape(
            self.batch_size, -1,self.dim_input)
        images_b = images_b[0:self.batch_size*self.num_samples_per_class, :, :].reshape(
            self.batch_size, -1,self.dim_input)
        
        images_out, labels_out = [], []
        for i in range(self.batch_size):
            imgs, labels = [], []
            for j in range(self.num_samples_per_class):
                imgs.append(images_a[i][j])
                imgs.append(images_b[i][j])

                labels.append(labels_a[i][j])
                labels.append(labels_b[i][j])

            imgs = np.stack(imgs, axis=0)
            labels = np.stack(labels, axis=0)
            images_out.append(imgs)
            labels_out.append(labels)

        images_out = np.stack(images_out, axis=0)
        labels_out = np.stack(labels_out, axis=0)
        labels_out = tf.keras.utils.to_categorical(labels_out, num_classes=2, dtype='int')

        return images_out, labels_out
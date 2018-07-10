## Copyright (C) 2017-2018, IBM Corp.
## Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
##                and Huan Zhang <ecezhang@ucdavis.edu>
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
## 
## This program is licenced under the Apache 2.0 licence,
## contained in the LICENCE file in this directory.

import numpy as np
import random

def linf_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=np.inf)

def l2_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=2)

def l1_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=1)

def l0_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=0)

def show(img, name = "output.png"):
    """
    Show MNSIT digits in the console.
    """
    np.save('img', img)
    fig = np.around((img + 0.5)*255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    return
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def generate_data(data, samples, targeted=True, random_and_least_likely = False, skip_wrong_label = True, start=0, ids = None, 
        target_classes = None, target_type = 0b1111, predictor = None, imagenet=False, remove_background_class=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    ids: true IDs of images in the dataset, if given, will use these images
    target_classes: a list of list of labels for each ids
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    true_labels = []
    true_ids = []
    information = []
    target_candidate_pool = np.eye(data.test_labels.shape[1])
    target_candidate_pool_remove_background_class = np.eye(data.test_labels.shape[1] - 1)
    print('generating labels...')
    if ids is None:
        ids = range(samples)
    else:
        ids = ids[start:start+samples]
        if target_classes:
            target_classes = target_classes[start:start+samples]
        start = 0
    total = 0
    n_correct = 0
    for i in ids:
        total += 1
        if targeted:
            predicted_label = -1 # unknown
            if random_and_least_likely:
                # if there is no user specified target classes
                if target_classes is None:
                    original_predict = np.squeeze(predictor(np.array([data.test_data[start+i]])))
                    num_classes = len(original_predict)
                    predicted_label = np.argmax(original_predict) + int(imagenet and remove_background_class)
                    least_likely_label = np.argmin(original_predict)
                    top2_label = np.argsort(original_predict)[-2]
                    start_class = 1 if (imagenet and not remove_background_class) else 0
                    random_class = predicted_label
                    new_seq = [least_likely_label, top2_label, predicted_label]
                    while random_class in new_seq:
                        random_class = random.randint(start_class, start_class + num_classes - 1)
                    new_seq[2] = random_class
                    true_label = np.argmax(data.test_labels[start+i])
                    seq = []
                    if true_label != predicted_label and skip_wrong_label:
                        seq = []
                    else:
                        if target_type & 0b10000:
                            for c in range(num_classes):
                                if c != predicted_label:
                                    seq.append(c)
                                    information.append('class'+str(c))
                        else:
                            if target_type & 0b0100:
                                # least
                                seq.append(new_seq[0])
                                information.append('least')
                            if target_type & 0b0001:
                                # top-2
                                seq.append(new_seq[1])
                                information.append('top2')
                            if target_type & 0b0010:
                                # random
                                seq.append(new_seq[2])
                                information.append('random')
                else:
                    # use user specified target classes
                    seq = target_classes[total - 1]
                    information.extend(len(seq) * ['user'])
            else:
                if imagenet:
                    if remove_background_class:
                        seq = random.sample(range(0,1000), 10)
                    else:
                        seq = random.sample(range(1,1001), 10)
                    information.extend(data.test_labels.shape[1] * ['random'])
                else:
                    seq = range(data.test_labels.shape[1])
                    information.extend(data.test_labels.shape[1] * ['seq'])

            is_correct = np.argmax(data.test_labels[start+i]) == predicted_label
            print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, seq = {}, info = {}".format(total, start + i, 
                np.argmax(data.test_labels[start+i]), predicted_label, is_correct, seq, [] if len(seq) == 0 else information[-len(seq):]))
            if is_correct:
                n_correct += 1
            for j in seq:
                # skip the original image label
                # if (j == np.argmax(data.test_labels[start+i])):
                if (j == np.argmax(data.test_labels[start+i]) - int(imagenet and remove_background_class)):
                    continue
                inputs.append(data.test_data[start+i])
                if remove_background_class:
                    targets.append(target_candidate_pool_remove_background_class[j])
                else:
                    targets.append(target_candidate_pool[j])
                true_labels.append(data.test_labels[start+i])
                if remove_background_class:
                    true_labels[-1] = true_labels[-1][1:]
                true_ids.append(start+i)
        else:
            inputs.append(data.test_data[start+i])
            if remove_background_class:
                # shift target class by 1
                print(np.argmax(data.test_labels[start+i]))
                print(np.argmax(data.test_labels[start+i][1:1001]))
                targets.append(data.test_labels[start+i][1:1001])
            else:
                targets.append(data.test_labels[start+i])
            true_labels.append(data.test_labels[start+i])
            if remove_background_class:
                true_labels[-1] = true_labels[-1][1:]
            true_ids.append(start+i)
            information.extend(['original'])

    inputs = np.array(inputs)
    targets = np.array(targets)
    true_labels = np.array(true_labels)
    true_ids = np.array(true_ids)
    print('labels generated')
    print('top-1 accuracy:', n_correct / float(samples))

    return inputs, targets, true_labels, true_ids, information


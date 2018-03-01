from time import time

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import array as np_array, zeros, full, maximum, math

dim = 32

def convolve_map(map, waldo_save, stride):
    with tf.Session() as session:

        saver = tf.train.import_meta_graph(waldo_save+'.meta')
        saver.restore(session, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name("input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        network = graph.get_tensor_by_name("network:0")

        probability_map_input = [[], []]
        for i in range((map.shape[0]-dim)//stride):
            # probability_map_input.append([])
            for j in range((map.shape[1]-dim)//stride):
                probability_map_input[0].append([stride*i, stride*j])
                probability_map_input[1].append(map[stride*i:stride*i+dim, stride*j:stride*j+dim])

        probability_map_input[0] = np_array(probability_map_input[0])
        probability_map_input[1] = np_array(probability_map_input[1])
        print(probability_map_input[1].shape)
        # check_probability_map_input(map, probability_map_input)
        probability_map = []
        size = 500
        for i in range(0, probability_map_input[1].shape[0], size):
            # print(i,":",min(i+size, len(probability_map_input)))
            probability_map.extend(
                network.eval(feed_dict={
                    input: probability_map_input[1][i: min(i+size, probability_map_input[1].shape[0])],
                    keep_prob:1}))

        print(len(probability_map_input[1]))
        print(len(probability_map))
        assert len(probability_map_input[1]) == len(probability_map)
        assert len(probability_map_input[0]) == len(probability_map)
        return probability_map_input[0], np_array(probability_map)



if __name__ == "__main__":
    print("Starting Map Test")

    map = "19.png"
    save = "CNN Waldo Recognizer___2018-02-03_12.01.25"
    numDisplay = 1
    r = 0.82
    r_1 = 1-r

    original_map = np_array(plt.imread("../Maps/Unprocessed/"+map))
    print(original_map.shape)
    # processed_map = read_processed_map("../Maps/Processed/"+map+".pkl")

    print("Analysing...")
    start = time()

    locations, outputs = convolve_map(original_map, save, int(dim/2/2))

    end = time()
    print("Took",(end-start),"Seconds to process")
    start = time()

    probabilities = []
    top = []
    if numDisplay <= 0:
        for l, x in zip(locations, outputs):
            p = 1/(1+math.exp(-(x[0]-x[1])/3))
            probabilities.append(p)
    else:
        for l, x in zip(locations, outputs):
            p = 1/(1+math.exp(-(x[0]-x[1])/3))
            probabilities.append(p)
            if len(top) < numDisplay:
                top.append([l, p])
                if len(top) == numDisplay:
                    top.sort(key=lambda x: x[1], reverse=True)
            else:
                for i in range(numDisplay):
                    if p > top[i][1]:
                        top[i] = [l, p]
                        break

    overlay = zeros(original_map.shape[0:2])
    print(overlay.shape)
    if numDisplay > 0:
        print(top)
        for i in range(numDisplay):
            l = top[i][0]
            p = top[i][1]
            section = full((dim, dim), p)
            overlay[l[0]:l[0]+dim, l[1]:l[1]+dim] = maximum(overlay[l[0]:l[0]+dim, l[1]:l[1]+dim], section)
    else:
        for l, p in zip(locations, probabilities):
            section = full((dim, dim), p*r_1 if p > r else 0)
            overlay[l[0]:l[0]+dim, l[1]:l[1]+dim] = maximum(overlay[l[0]:l[0]+dim, l[1]:l[1]+dim], section)

    end = time()
    print("Took",(end-start),"Seconds to analyse")


    plt.imshow(original_map, cmap='jet')
    plt.imshow(overlay, cmap='gray', alpha=0.75)
    plt.show()


import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import array as np_array, reshape


def get_probability_map(map, waldo_save="CNN Waldo Recognizer___2018-01-11_21.22.59"):
    with tf.Session() as session:

        saver = tf.train.import_meta_graph(waldo_save+'.meta')
        saver.restore(session, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name("input:0")
        network = graph.get_tensor_by_name("network:0")

        probability_map_input = []
        for i in range(map.shape[0]//32):
            # probability_map_input.append([])
            for j in range(map.shape[1]//32):
                probability_map_input.append(map[32*i:32*(i+1), 32*j:32*(j+1)])

        probability_map_input = np_array(probability_map_input)
        print(probability_map_input.shape)
        # check_probability_map_input(map, probability_map_input)
        probability_map = []
        size = 500
        for i in range(0, len(probability_map_input), size):
            # print(i,":",min(i+size, len(probability_map_input)))
            probability_map.extend(
                network.eval(feed_dict={
                    input: probability_map_input[i: min(i+size, len(probability_map_input))]}))

        print(len(probability_map_input))
        print(len(probability_map))
        assert len(probability_map_input) == len(probability_map)
        return reshape(a=np_array(probability_map), newshape=(map.shape[0]//32, map.shape[1]//32, 2))

def check_probability_map_input(map, probability_map_input):
    print("Checking if Map matches Probability Map Input...")
    for i in range(map.shape[0]//32*32):
        i_ = i//32
        imod = i%32
        for j in range(map.shape[1]//32*32):
            map_at = map[i][j]
            probability_map_input_at = probability_map_input[i_, j//32, imod, j%32, :]
            if map_at[0] != probability_map_input_at[0] or \
               map_at[1] != probability_map_input_at[1] or \
               map_at[2] != probability_map_input_at[2]:
                print(i, j)
                print(i_, j//32, imod, j%32)
                print(map_at)
                print(probability_map_input_at)
                exit(1)
    print("Map matches Probability Map Input")

if __name__ == "__main__":

    map = np_array(plt.imread("../Maps/6.png"))
    print(map.shape)
    # 1346, 2048, 3

    analysed = get_probability_map(map)
    probability_map = []
    for row in analysed:
        probability_map.append([])
        for x in row:
            probability_map[-1].append(abs(x[0])/(abs(x[0])+abs(x[1])))

    overlay = []
    for i in range(map.shape[0]//32*32):
        overlay.append([])
        for j in range(map.shape[1]//32*32):
            overlay[-1].append(probability_map[i//32][j//32])

    plt.imshow(map, cmap='jet')
    plt.imshow(overlay, cmap='gray', alpha=0.75)
    plt.show()


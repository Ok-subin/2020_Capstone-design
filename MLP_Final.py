import cv2
import random
import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

lbp_code = []
file = open('C:/Users/Ok Subin/Desktop/train_face_code.txt', 'r')
file2 = open('C:/Users/Ok Subin/Desktop/train_nonface_code.txt', 'r')

while True:
    line = file.readline()
    if not line: break

    split_line = line.split(',')
    lbp_code.append(split_line)

while True:
    line = file2.readline()
    if not line: break

    split_line = line.split(',')
    lbp_code.append(split_line)

for i in range(0, 7698):
    for j in range(0, 5119):
        lbp_code[i][j] = float(lbp_code[i][j]) * 1 / (25 * 20)

lbp_code = np.array(lbp_code, float)

train_label = []

for i in range(0, 699):
    train_label.append(1)

for i in range(699, 7698):
    train_label.append(0)

no_of_different_labels = 2
lr = np.arange(no_of_different_labels)

train_labels_one_hot = []

for i in range(0, 7698):
    if (train_label[i] == 1.):
        train_labels_one_hot.append([0., 1.])
    else:
        train_labels_one_hot.append([1., 0.])

train_labels_one_hot = np.array(train_labels_one_hot)

train_labels_one_hot[train_labels_one_hot == 0] = 0.01
train_labels_one_hot[train_labels_one_hot == 1] = 0.99

#//////////////////////////////////////////////////////////////////////////

# normalization
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)

class MLP_LBP:
    def __init__(self, network_structure, learning_rate, bias=None):
        self.structure = network_structure
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def sigmoid(x):
        return 1 / (1 + np.e ** -x)

    activation_function = sigmoid

    def create_weight_matrices(self):
        X = truncated_normal(mean=2, sd=1, low=-0.5, upp=0.5)

        bias_node = 1 if self.bias else 0
        self.weights_matrices = []
        layer_index = 1
        no_of_layers = len(self.structure)  # structure = [20*256, 50, 2] --> len = 3
        while layer_index < no_of_layers:
            nodes_in = self.structure[layer_index - 1]
            nodes_out = self.structure[layer_index]
            n = (nodes_in + bias_node) * nodes_out
            rad = 1 / np.sqrt(nodes_in)
            X = truncated_normal(mean=2, sd=1, low=-rad, upp=rad)
            wm = X.rvs(n).reshape((nodes_out, nodes_in + bias_node))
            self.weights_matrices.append(wm)
            layer_index += 1

    def train_single(self, input_vector, target_vector):
        no_of_layers = len(self.structure)
        input_vector = np.array(input_vector, ndmin=2).T  # 전치 / ndmin = 2이면 2차원 배열로 ==> [[a], [b], ....]

        layer_index = 0
        # The output/input vectors of the various layers:
        res_vectors = [input_vector]

        while layer_index < no_of_layers - 1:
            in_vector = res_vectors[-1]

            if self.bias:
                # adding bias node to the end of the 'input'_vector
                in_vector = np.concatenate((in_vector, [[self.bias]]))
                res_vectors[-1] = in_vector

            x = np.dot(self.weights_matrices[layer_index], in_vector)
            out_vector = activation_function(x)
            res_vectors.append(out_vector)
            layer_index += 1

        layer_index = no_of_layers - 1
        target_vector = np.array(target_vector, ndmin=2).T
        # The input vectors to the various layers
        output_errors = target_vector - out_vector

        while layer_index > 0:
            out_vector = res_vectors[layer_index]
            in_vector = res_vectors[layer_index - 1]

            if self.bias and not layer_index == (no_of_layers - 1):
                out_vector = out_vector[:-1, :].copy()

            tmp = output_errors * out_vector * (1.0 - out_vector)
            tmp = np.dot(tmp, in_vector.T)

            # if self.bias:
            #    tmp = tmp[:-1,:]

            self.weights_matrices[layer_index - 1] += self.learning_rate * tmp

            output_errors = np.dot(self.weights_matrices[layer_index - 1].T, output_errors)

            if self.bias:
                output_errors = output_errors[:-1, :]

            layer_index -= 1

    # ANN.train(lbp_code, train_labels_one_hot, epochs=epochs)
    def train(self, data_array, labels_one_hot_array, epochs=1, intermediate_results=False):
        intermediate_weights = []

        for epoch in range(epochs):
            for i in range(len(data_array)):
                if i % 2 == 0:
                    train_num = random.randint(0, 698)
                else:
                    train_num = random.randint(699, 7697)

                self.train_single(data_array[train_num], labels_one_hot_array[train_num])

            if intermediate_results:
                intermediate_weights.append((self.wih.copy(), self.who.copy()))
        return intermediate_weights

    # lbp_code - 20, 256
    def run(self, input_vector):
        no_of_layers = len(self.structure) # 3
        if self.bias:
            input_vector = np.concatenate((input_vector, [self.bias]))
        in_vector = np.array(input_vector, ndmin=2).T

        layer_index = 1
        # The input vectors to the various layers
        while layer_index < no_of_layers:
            x = np.dot(self.weights_matrices[layer_index - 1], in_vector)
            out_vector = activation_function(x)

            # input vector for next layer
            in_vector = out_vector
            if self.bias:
                in_vector = np.concatenate((in_vector, [[self.bias]]))

            layer_index += 1
        # print("out : " + str(out_vector[0]))
        # print("out : " + str(out_vector[0][0]))

        return out_vector

    # lbp_code , train_label
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0

        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            # print("0 : " + str(res[0]) + "1 : " + str(res[1]))
            # print("res : " + str(res_max))
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

    def LBP(self, cut_img):
        max_size = 100
        x = 4
        y = 5

        len_x = max_size // x   # 25
        len_y = max_size // y   #20

        img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)  # grayscale로 바꿔줌
        #img = cv2.resize(img, (max_size, max_size))  --> 이미 100 100

        his_list = []
        for i in range(0, x * y):
            tmp = []
            for j in range(0, 256):
                tmp.append(0)
            his_list.append(tmp)

        #dst = np.zeros((max_size - 2, max_size - 2), np.uint8)
        count = 0

        for index_y in range(0, max_size, len_y):  # (0, 100, 20)
            for index_x in range(0, max_size, len_x):  # (0, 100, 25)
                for j in range(1, len_y - 1):  # (1, 19)
                    for i in range(1, len_x - 1):  # (1, 24)

                        center = img[i, j]
                        code = 0

                        code |= ((img[i - 1, j - 1]) > center) << 7
                        code |= ((img[i - 1, j]) > center) << 6
                        code |= ((img[i - 1, j + 1]) > center) << 5
                        code |= ((img[i, j + 1]) > center) << 4
                        code |= ((img[i + 1, j + 1]) > center) << 3
                        code |= ((img[i + 1, j]) > center) << 2
                        code |= ((img[i + 1, j - 1]) > center) << 1
                        code |= ((img[i, j - 1]) > center) << 0

                        int_code = int(code)
                        his_list[count][int_code] += 1
                #for a in range(0, 256):
                    #print(his_list[count][a])
                count += 1

            #print("complete")

        """
        for j in range(0, count):
            for i in range(0, 256):
                if j == count - 1 and i == 255:
                    write_value = str(his_list[j][i])

                else:
                    write_value = str(his_list[j][i]) + "," """

        return his_list

    def cut_img(self):
        idx = 1
        size = 100
        move = 30
        #file = open('C:/Users/Ok Subin/Desktop/test_code.txt', 'w')
        for g in range(1, 11):
            box_info = []
            #img_num = random.randint(1, 10)
            img_num = g
            img_add = "C:/Users/Ok Subin/Desktop/test/"
            img_add += str(img_num)
            img_add += ".jpg"

            img = cv2.imread(img_add)

            h, w, _ = img.shape

            #img = cv2.resize(img, (300, 300))

            for y in range(0, h-size+1, 50):  # 픽셀 이동 간격 5
                for x in range(0, w-size+1, 50):
                    cut_img = img.copy()
                    cut_img = cut_img[y:y + size, x:x + size]
                    #cv2.imshow("", cut_img)
                    #cv2.waitKey(0)

                    his_list = []
                    output_list = []
                    for i in range(0, 20):
                        tmp = []
                        for j in range(0, 256):
                            tmp.append(0)
                        his_list.append(tmp)
                    his_list = self.LBP(cut_img)    #20, 256

                    hist_list = []
                    for a in range(0, 20):
                        for b in range(0, 256):
                            hist_list.append(his_list[a][b])

                    output_list = self.run(hist_list)
                    output = output_list.argmax()

                    if output == 1:
                        tmp = []
                        tmp.append(y)
                        tmp.append(y+size)
                        tmp.append(x)
                        tmp.append(x+size)

                        box_info.append(tmp)

            self.write_bb(box_info, img, g+10)

    def write_bb(self, box_info, img, img_num):
        final_img = img

        for i in range(0, len(box_info)):
            self.top = box_info[i][0]
            self.bottom = box_info[i][1]
            self.left = box_info[i][2]
            self.right = box_info[i][3]

            print("top : " + str(self.top) + ", bot : " + str(self.bottom) + ", left : " + str(self.left) + ", rig : " + str(self.right))

            final_img = cv2.rectangle(final_img, (self.top, self.left), (self.right, self.bottom), (255, 0, 0), 2)

        write_add = "C:/Users/Ok Subin/Desktop/result/"
        write_add += str(img_num)
        write_add += ".jpg"
        cv2.imwrite(write_add, final_img)

#//////////////////////////////////////////////////////////////////////

epochs = 3
image_pixels = 20 * 256

ANN = MLP_LBP(network_structure=[image_pixels, 50, 2],
                    learning_rate=0.01,
                    bias=True)

ANN.train(lbp_code, train_labels_one_hot, epochs=epochs)

ANN.cut_img()

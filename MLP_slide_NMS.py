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

# 25*256 = 6400
for i in range(0, 12460):
    for j in range(0, 6399):
        lbp_code[i][j] = float(lbp_code[i][j]) * 1 / (60 * 60)

lbp_code = np.array(lbp_code, float)

train_label = []

# 1036 + 11424

for i in range(0, 1036):
    train_label.append(1)

for i in range(1036, 12460):
    train_label.append(0)

no_of_different_labels = 2
lr = np.arange(no_of_different_labels)

train_labels_one_hot = []

for i in range(0, 12460):
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
                    train_num = random.randint(0, 1035)
                else:
                    train_num = random.randint(1036, 12459)

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

    def softmax(self, a):
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y

    # lbp_code , train_label
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0

        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            result = res[res_max]

            if res_max == labels[i]:
                if result >= 0.95:
                    corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

    def LBP(self, cut_img):
        max_size = 300
        x = 5
        y = 5

        len_x = max_size // x   # 50
        len_y = max_size // y   # 40

        img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)  # grayscale로 바꿔줌
        img = cv2.resize(img, (max_size, max_size))  #--> 이미 100 100

        his_list = []
        for i in range(0, x * y):
            tmp = []
            for j in range(0, 256):
                tmp.append(0)
            his_list.append(tmp)

        #dst = np.zeros((max_size - 2, max_size - 2), np.uint8)
        count = 0

        for index_y in range(0, max_size, len_y):  # (0, 200, 40)
            for index_x in range(0, max_size, len_x):  # (0, 200, 50)
                for j in range(1, len_y-1):  # (1, 39)
                    for i in range(1, len_x-1):  # (1, 49)

                        ii = index_x + i
                        jj = index_y + j

                        center = img[ii, jj]
                        code = 0

                        code |= ((img[ii - 1, jj - 1]) > center) << 7
                        code |= ((img[ii - 1, jj]) > center) << 6
                        code |= ((img[ii - 1, jj + 1]) > center) << 5
                        code |= ((img[ii, jj + 1]) > center) << 4
                        code |= ((img[ii + 1, jj + 1]) > center) << 3
                        code |= ((img[ii + 1, jj]) > center) << 2
                        code |= ((img[ii + 1, jj - 1]) > center) << 1
                        code |= ((img[ii, jj - 1]) > center) << 0

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
        #size = 100
        #move = 30
        #file = open('C:/Users/Ok Subin/Desktop/test_code.txt', 'w')
        for g in range(1, 4):
            # img_num = random.randint(1, 10)
            img_num = g
            img_add = "C:/Users/Ok Subin/Desktop/test/"
            img_add += str(img_num)
            img_add += ".jpg"
            img = cv2.imread(img_add)

            img = cv2.resize(img, (300, 300))
            h, w, _ = img.shape

            if(h < w):
                cut_size = h
            else:
                cut_size = w

            for f in range(100, cut_size, 50):  #size를 50씩 키우면서 (100, 300, 50)   --> 100, 150, 200, 250
                #print(img_add)
                #print(h)
                size = f
                box_info = []
                for y in range(0, f+1, 10):  # 픽셀 이동 간격 10 (0, size+1, 10) (0, 101, 10)
                    for x in range(0, f+1, 10):
                        cut_img = img.copy()
                        #cv2.imshow("", cut_img)
                        #cv2.waitKey(0)
                        cut_img = cut_img[y:y + size, x:x + size]          # 정사각형
                        #cv2.imshow("", cut_img)
                        #cv2.waitKey(0)

                        his_list = []
                        for i in range(0, 25):
                            tmp = []
                            for j in range(0, 256):
                                tmp.append(0)
                            his_list.append(tmp)
                        his_list = self.LBP(cut_img)  # 20, 256

                        hist_list = []
                        for a in range(0, 25):
                            for b in range(0, 256):
                                hist_list.append(his_list[a][b])

                        output_list = self.run(hist_list)
                        #output = output_list.argmax()

                        soft_input = []
                        soft_input.append(float(output_list[0][0]))
                        soft_input.append(float(output_list[1][0]))

                        soft = self.softmax(soft_input)
                        output = soft.argmax()
                        output_soft = soft[output]

                        #print(output_soft)

                        if output == 1:
                            if output_list[output] >= 0.95:
                            #if output_soft >= 0.8:
                                tmp = []
                                tmp.append(y)
                                tmp.append(y + size)
                                tmp.append(x)
                                tmp.append(x + size)

                                box_info.append(tmp)

                result_box = self.non_max_suppression_fast(box_info, 0.95)
                print("result box : " + str(type(result_box)))
                print("result box : " + str(result_box))
                #self.write_bb(box_info, img, idx)
                self.write_bb(result_box, img, idx)
                idx+=1

    def intersection_over_union(self, box1, box2):

        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        iou = interArea / float(box1Area + box2Area - interArea)

        return iou

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

    def non_max_suppression_fast(self, boxes, overlapThresh):
        if len(boxes) == 0:
            return []

        pick = []
        boxes = np.array(boxes)

        y1 = boxes[:, 0]
        y2 = boxes[:, 1]
        x1 = boxes[:, 2]
        x2 = boxes[:, 3]

        # bounding box의 area 계산
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        # argsort : 작은 숫자부터 큰 숫자까지의 index를 반환 --> 여기서는 y2(bottom)를 기준으로
        # 이게 아래에서부터 검색을 의미.
        # 그럼 이거 대신 bb와 겹침 개수가 큰 것부터 검색하도록 해야 함. 
        #idxs = np.argsort(y2)
        
        # 겹치는 개수 세는 과정
        count_list = []
        delete_list = []
        print("왜지? " + str(len(x1)))
        for target_count in range(0, len(x1)):  # (0, 51)
            total_overlap = 0
            last = len(x1) - 1

            # target index의 박스 좌표들
            target_x1 = x1[target_count]
            target_x2 = x2[target_count]
            target_y1 = y1[target_count]
            target_y2 = y2[target_count]

            target_box = []
            target_box.append(target_x1)
            target_box.append(target_y1)
            target_box.append(target_x2)
            target_box.append(target_y2)

            target_region = (target_x2 - target_x1 + 1) * (target_y2 - target_y1 + 1)

            # target을 제외한 index의 박스 좌표들
            remain_x1 = x1[0 : target_count]
            remain_x1 = np.append(remain_x1, x1[target_count + 1:last + 1])

            remain_x2 = x1[0 : target_count]
            remain_x2 = np.append(remain_x2, x2[target_count + 1:last + 1])

            remain_y1 = x1[0 : target_count]
            remain_y1 = np.append(remain_y1, y1[target_count + 1:last + 1])

            remain_y2 = x1[0 : target_count]
            remain_y2 = np.append(remain_y2, y2[target_count + 1:last + 1])

            # 값 확인해보기
            for i in range(0, 4):
                print("target_box : " + str(target_box[i]))
                print(target_region)

            # target을 제외한 bb와의 겹침 개수 계산
            for i in range(0, len(remain_x1)):      # (0, 50)
                xx1 = remain_x1[i]
                xx2 = remain_x2[i]
                yy1 = remain_y1[i]
                yy2 = remain_y2[i]

                remain_box = []
                remain_box.append(xx1)
                remain_box.append(yy1)
                remain_box.append(xx2)
                remain_box.append(yy2)

                '''
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                print("xx1 : " + str(xx1))
                print("xx2 : " + str(xx2))
                print("yy1 : " + str(yy1))
                print("yy2 : " + str(yy2))

                print("w : " + str(w))
                print("h : " + str(h)) 

                #overlap = (w * h) / target_region '''
                overlap = self.intersection_over_union(target_box, remain_box)
                print("overlap : " + str(overlap))

                # overlap에 대한 threshold 설정 필요하다면 넣기

                if overlap > 0.3:
                    total_overlap += 1

            print("total overlap = " + str(total_overlap))
            count_list.append(total_overlap)
            print("total overlap 추가 후 count list 길이 : " + str(len(count_list)))

            # 총 겹치는 개수가 전체 bb 개수의 10%가 안될 경우 삭제하기 위해 delete list에 추가
            if (total_overlap < len(x1)/10):
                print("delete : " + str(target_count))
                delete_list.append(target_count)
                print("추가 후 delete list 길이 : " + str(len(delete_list)))

        # idxs : 겹침 개수가 작은 것부터 index 나열
        idxs = np.argsort(count_list)

        print("총 겹침 횟수에 대한 list인 count list의 len : " + str(len(count_list)))
        print("이를 큰 순서대로 나열한 len : " + str(len(idxs)))

        '''
        print("original idxs : ")
        for i in range(0, len(x1)):
            print(str(idxs[i]) + " , ")
        print("\n\n") '''

        # 총 bb의 개수에서 일정 비율 이하만큼 겹치면 제거
        # 여기서 delete_list는 위 겹치는 bb count하면서 일정 비율 이하이면 해당 index를 저장하도록 했음.
        idxs = np.delete(idxs, delete_list)
        
        print("after delete idxs : ")
        for i in range(0, len(idxs)):
            print(str(idxs[i]) + " , ")
        print("\n\n")

        while len(idxs) > 0:
            last = len(idxs) - 1
            # i = 가장 큰 원소가 있는 위치
            i = idxs[last]
            pick.append(i)

            # xx1 : y2 값이 가장 클 때의 index의 값인 x1[i]와 나머지 x1의 값들을 비교하여 가장 큰 값 얻기
            # xx2 : 위의 과정을 거쳐 가장 작은 값 얻기
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            file_name = "C:/Users/Ok Subin/Desktop/result/overlap.txt"
            overlap_file = open(file_name, "w")
            overlap_file.write(str(overlap) + " ")
            overlap_file.write("\n----------------------------------------------\n")

            print ("overlap : " + str(overlap))

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

            print("반복문에서 delete idxs : ")
            for i in range(0, len(idxs)):
                print(str(idxs[i]) + " , ")
            print("\n\n")

        return boxes[pick].astype("int")

    def return_weight(self):
        file_name = "C:/Users/Ok Subin/Desktop/result/weight.txt"
        weight_file = open(file_name, "w")
        print(len(self.weights_matrices))
        print(len(self.weights_matrices[0]))
        print(len(self.weights_matrices[0][0]))
        for i in range(0, len(self.weights_matrices)):
            for j in range(0, len(self.weights_matrices[i])):
                for k in range(0, len(self.weights_matrices[i][j])):
                    weight_file.write(str(self.weights_matrices[i][j][k]) + " ")
            weight_file.write("\n----------------------------------------------\n")
            #print("weight " + str(i) + " : " + str(self.weights_matrices[i]))

#//////////////////////////////////////////////////////////////////////

epochs = 10
image_pixels = 5*5

ANN = MLP_LBP(network_structure=[image_pixels, 50, 2],
                    learning_rate=0.01,
                    bias=True)

ANN.train(lbp_code, train_labels_one_hot, epochs=epochs)

corrects, wrongs = ANN.evaluate(lbp_code, train_label)

ANN.cut_img()

print("corrects: ", corrects, "wrongs: ", wrongs)
print("accuracy train: ", corrects / (corrects + wrongs) * 100)

#ANN.return_weight()

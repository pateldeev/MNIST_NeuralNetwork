import numpy as np
import pickle
import math
import cv2


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1-s)  # derivative of sigmoid


class Network:

    def __init__(self, layer_sizes=None):
        if layer_sizes is None:
            layer_sizes = [784, 28, 20, 10]
        # create arrays to hold weights and biases
        self.w = [np.empty([layer_sizes[i-1], layer_sizes[i]]) for i in range(1, len(layer_sizes))]
        self.b = [np.empty(layer_sizes[i]) for i in range(1, len(layer_sizes))]

    def randomize_parameters(self):
        for n, (w, b) in enumerate(zip(self.w, self.b)):
            self.w[n] = np.random.rand(len(w), len(w[0]))
            self.b[n] = np.random.rand(len(b))

    def save_parameters(self, base_dir: str, file_name="params.byte"):
        with open(base_dir + file_name, 'wb') as f:
            for w, b in zip(self.w, self.b):
                temp = w.dumps()
                f.write(len(temp).to_bytes(4, byteorder="big"))
                f.write(w.dumps())
                temp = b.dumps()
                f.write(len(temp).to_bytes(4, byteorder="big"))
                f.write(b.dumps())

    def read_parameters(self, base_dir: str, file_name="params.byte"):
        with open(base_dir + file_name, 'rb') as f:
            for i in range(len(self.w)):
                temp = int.from_bytes(f.read(4), byteorder="big")
                self.w[i] = pickle.loads(f.read(temp))
                temp = int.from_bytes(f.read(4), byteorder="big")
                self.b[i] = pickle.loads(f.read(temp))

    def back_prop(self, img, expected_output: int):
        y = [1 if i == expected_output else 0 for i in range(10)]

        # compute activation (a) of each layer (z --> before sigmoid application to a)
        z = [np.ndarray.astype(np.ndarray.flatten(img), dtype=float)]
        z[-1] /= 255
        a = z.copy()  # no need to apply sigmoid before first layer
        for w, b in zip(self.w, self.b):
            z.append(np.add(np.matmul(a[-1], w), b))  # compute weighted sum from previous layer
            a.append(np.array([sigmoid(z_q) for z_q in z[-1]]))  # activation value after applying sigmoid

        # compute cost (sum of squares)
        c_0 = sum([(a_i - y_i) ** 2 for a_i, y_i in zip(a[-1], y)])
        print("||||||||||||||||||||||||||||")
        print("Output: ", a[-1])
        print("Cost: ", c_0)

        # allocate space to hold gradient - represent optimal change
        g_w, g_b = [np.empty_like(w) for w in self.w], [np.empty_like(b) for b in self.b]

        # allocate space to hold partial derivatives of cost with respect to all activation values
        dc_da = [np.empty_like(a_l) for a_l in a]

        # compute partial derivatives of cost with respect to activation values in final layer
        dc_da[-1] = [2 * (a_i - y_i) for a_i, y_i in zip(a[-1], y)]

        # compute the gradient via back-propagation through layers
        for l in range(len(a) - 1, 0, -1):
            # compute gradient with respect to biases
            for i, (dc_da_i, z_i) in enumerate(zip(dc_da[l], z[l])):
                g_b[l-1][i] = dc_da_i * sigmoid_prime(z_i)

            # compute gradient with respect to weights
            for k, a_k in enumerate(a[l-1]):
                for i, (g_b_i) in enumerate(g_b[l-1]):
                    g_w[l-1][k][i] = g_b_i * a_k

            # compute gradient with respect to activations of previous layers - needed for next iteration
            for k in range(len(dc_da[l-1])):
                # must sum of overall connected activations in current layer
                dc_da[l-1][k] = 0
                for dc_da_i, g_b_i, w_ki in zip(dc_da[l], g_b[l-1], self.w[l-1][k]):
                    dc_da[l-1][k] += g_b_i * w_ki

        # change weights and biases in direction of negative gradient
        for w, b, g_w, g_b in zip(self.w, self.b, g_w, g_b):
            # change biases
            for i, g_b_i in enumerate(g_b):
                b[i] -= g_b_i

            # change weights
            for r, g_w_r in enumerate(g_w):
                for c, g_w_rc in enumerate(g_w_r):
                    w[r][c] -= g_w_rc

        out_new = self.compute_output(img)
        c_0_new = sum([(a_i - y_i) ** 2 for a_i, y_i in zip(out_new, y)])
        print("New Output: ", out_new)
        print("New Cost: ", c_0_new)

    def compute_output(self, img):
        assert len(img) == len(img[0]) == 28

        output = np.ndarray.astype(np.ndarray.flatten(img), dtype=float)
        output /= 255  # normalize pixel values

        for w, b in zip(self.w, self.b):
            output = np.add(np.matmul(output, w), b)  # compute weighted sum
            for i, o in enumerate(output):
                output[i] = sigmoid(o)  # apply sigmoid function

        return output

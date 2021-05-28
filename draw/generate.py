from draw_model import DrawModel
from config import *
from utility import save_image, get_images
from dataUtils import loadConstellations
import torch.nn.utils
import numpy as np

torch.set_default_tensor_type('torch.FloatTensor')

model = DrawModel(T,A,B,z_size,N,dec_size,enc_size)

if USE_CUDA:
    model.cuda()

state_dict = torch.load('save/weights_46000.tar', map_location=torch.device('cpu' if not USE_CUDA else 'gpu'))
model.load_state_dict(state_dict)

images = loadConstellations(selectedDataset="limitedSet", pictureTypes=["final_easy"], pictureSize=(A, B), colorMode="grayscale")
dotted = np.array(list(map(lambda x: np.array(x['final_easy']), list(images.values()))))
dotted = dotted / 255
print(dotted.shape)


def generate(initial_state, count):
    x = model.generate(batch_size, initial_state)
    save_image(x, count, T - 1, 'test')

def get_test_images(A, B):
    images = []
    for j in range(min(dotted[0].shape[0], batch_size)):
        dotted_flatten = np.zeros((batch_size, A * B))
        for i in range(batch_size):
            dotted_flatten[i] = dotted[i][j].flatten()
            initial_state = torch.Tensor(dotted_flatten)
        x = model.generate(batch_size, initial_state)
        images += get_images(x[-1], A, B)
    return images

def generate_test_images():
    for j in range(min(dotted[0].shape[0], batch_size)):
        dotted_flatten = np.zeros((batch_size, A * B))
        for i in range(batch_size):
            dotted_flatten[i] = dotted[i][j].flatten()
            initial_state = torch.Tensor(dotted_flatten)
        generate(initial_state, j)

if __name__ == '__main__':
    generate_test_images()
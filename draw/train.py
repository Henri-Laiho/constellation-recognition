import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from draw_model import DrawModel
from config import *
from utility import Variable,save_image,xrecons_grid
import torch.nn.utils
from dataUtils import loadConstellations
import matplotlib.pyplot as plt
import numpy as np

torch.set_default_tensor_type('torch.FloatTensor')

#train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('data/', train=True, download=True,
#                   transform=transforms.Compose([
#                       transforms.ToTensor()])),
#    batch_size=batch_size, shuffle=False)

images = loadConstellations(selectedDataset="all", pictureTypes=["outline"], pictureSize=(A, B), colorMode="grayscale")
outlines = np.array(list(map(lambda x: x['outline'], list(images.values()))))
#outlines_reshaped = np.zeros((outlines.shape[0], A, B))
outlines_reshaped = list()
i = 0
j = 0
while True:
    if i >= len(outlines):
        break
    if j >= len(outlines[i]):
        i += 1
        j = 0
        continue
    outlines_reshaped.append(outlines[i][j] / 255)
    j += 1
    
    
outlines_reshaped = np.array(outlines_reshaped)
print(outlines_reshaped.shape)
outline_tensor = torch.Tensor(outlines_reshaped)
outline_dataset = torch.utils.data.TensorDataset(outline_tensor)
train_loader = torch.utils.data.DataLoader(outline_dataset, batch_size=batch_size)
model = DrawModel(T,A,B,z_size,N,dec_size,enc_size)
optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(beta1,0.999))

if USE_CUDA:
    model.cuda()

def train():
    avg_loss = 0
    count = 0
    for epoch in range(epoch_num):
        for data in train_loader:
            data = data[0]
            bs = data.size()[0]
            data = Variable(data).view(bs, -1)
            optimizer.zero_grad()
            loss = model.loss(data)
            avg_loss += loss.cpu().data.numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            count += 1
            if count % 100 == 0:
                print('Epoch-{}; Count-{}; loss: {};'.format(epoch, count, avg_loss / 100))
                if count % 500 == 0:
                    torch.save(model.state_dict(),'save/weights_%d.tar'%(count))
                    generate_image(count)
                avg_loss = 0
    torch.save(model.state_dict(), 'save/weights_final.tar')
    generate_image(count)


def generate_image(count):
    x = model.generate(batch_size)
    save_image(x,count)

def save_example_image():
    train_iter = iter(train_loader)
    data = train_iter.next()[0]
    print(data.shape)
    img = data.cpu().numpy().reshape(batch_size, A, B)
    imgs = xrecons_grid(img, B, A)
    plt.matshow(imgs, cmap=plt.cm.gray)
    plt.savefig('image/example.png')

if __name__ == '__main__':
    save_example_image()
    train()
from __future__ import absolute_import
from __future__ import print_function
import PIL
import torch
import glob as gb
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from facenet_pytorch import MTCNN
# import model.resnet50_128 as model
import model.resnet50_2048 as model_resnet
import model.senet50_2048 as model_senet
from facenet_pytorch import MTCNN
# hyper parameters
batch_size = 4
# batch_size = 16
mean = (131.0912, 103.8827, 91.4953)

train_on_gpu = torch.cuda.is_available()

DEVICE = torch.device("cuda")

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    DEVICE = torch.device("cpu")
else:
    print('CUDA is available!  Training on GPU ...')
    DEVICE = torch.device("cuda")

DEVICE = torch.device("cuda")

mtcnn = MTCNN(image_size=224, select_largest=False, post_process=False, device=DEVICE)
trans1 = transforms.Resize((224,224))
trans2 = transforms.ToTensor()

def load_data(path='', shape=None):
    short_size = 224.0
    crop_size = shape
    img = PIL.Image.open(path)
    im_shape = np.array(img.size)    # in the format of (width, height, *)
    
    img = mtcnn(img)
    if img is None:
        # img = torch.zeros((3,224,224))
        img = trans1(PIL.Image.open(path))
        img = trans2(img)

    img = img.permute(1, 2, 0).int().numpy()

    return img - mean


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


def initialize_model_res():
    network = model_resnet.resnet50_ft(weights_path='./model/resnet50_2048.pth').to(DEVICE)
    network.eval()
    return network

def initialize_model_sen():
    network = model_senet.senet50_ft(weights_path='./model/senet50_2048.pth').to(DEVICE)
    network.eval()
    return network


def image_encoding(model_res, model_sen, facepaths):
    num_faces = len(facepaths)

    face_feats = np.empty((num_faces, 2*2048))
    imgpaths = facepaths
    imgchunks = list(chunks(imgpaths, batch_size))

    for c, imgs in tqdm(enumerate(imgchunks), total=len(imgchunks)):
        im_array = np.array([load_data(path=i, shape=(224, 224, 3)) for i in imgs])

        f_sen = model_sen(torch.Tensor(im_array.transpose(0, 3, 1, 2)).to(DEVICE))
        f_res = model_res(torch.Tensor(im_array.transpose(0, 3, 1, 2)).to(DEVICE))
        f = torch.cat((f_sen, f_res), dim=1).detach().cpu().numpy()[:, :, 0, 0]

        start = c * batch_size
        end = min((c + 1) * batch_size, num_faces)
        # This is different from the Keras model where the normalization has been done inside the model.
        face_feats[start:end] = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))
    return face_feats


if __name__ == '__main__':
    # rename samples (test set)/tight_crop -> samples
    facepaths = gb.glob('../samples/*/*.jpg')
    model_eval = initialize_model()
    face_feats = image_encoding(model_eval, facepaths)
    S = np.dot(face_feats, face_feats.T)
    import pylab as plt
    plt.imshow(S)
    plt.show()
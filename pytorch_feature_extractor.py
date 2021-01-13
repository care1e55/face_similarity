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
# import model.resnet50_2048 as model
import model.senet50_2048 as model
from facenet_pytorch import MTCNN
# hyper parameters
batch_size = 16
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
# mtcnn = MTCNN(
#     image_size=224,
#     device=DEVICE,
#     select_largest=False,
#     selection_method = "center_weighted_size",
#     # post_process=False,
#     margin = 14,
#     # min_face_size = 20,
# )

def load_data(path='', shape=None):
    short_size = 224.0
    crop_size = shape
    img = PIL.Image.open(path)
    im_shape = np.array(img.size)    # in the format of (width, height, *)
    
    img = mtcnn(img)
    if img is None:
        img = torch.zeros((3,224,224))
        pass

    img = img.permute(1, 2, 0).int().numpy()

    return img - mean


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


def initialize_model():
    # Download the pytorch model and weights.
    # network = model.resnet50_128(weights_path='./model/resnet50_128.pth').to(DEVICE)
    # network = model.resnet50_ft(weights_path='./model/resnet50_2048.pth').to(DEVICE)
    network = model.senet50_ft(weights_path='./model/senet50_2048.pth').to(DEVICE)
    network.eval()
    return network


def image_encoding(model, facepaths):
    num_faces = len(facepaths)
    # face_feats = np.empty((num_faces, 128))
    face_feats = np.empty((num_faces, 2048))
    imgpaths = facepaths
    imgchunks = list(chunks(imgpaths, batch_size))

    for c, imgs in tqdm(enumerate(imgchunks), total=len(imgchunks)):
        im_array = np.array([load_data(path=i, shape=(224, 224, 3)) for i in imgs])
        # torch.Size([16, 128])
        # torch.Size([16, 128, 1, 1])
        # print()
        # print(model(torch.Tensor(im_array.transpose(0, 3, 1, 2)).to(DEVICE))[0].size())
        # print(model(torch.Tensor(im_array.transpose(0, 3, 1, 2)).to(DEVICE))[1].size())
        # print(model(torch.Tensor(im_array.transpose(0, 3, 1, 2)).to(DEVICE)).size())

        # f = model(torch.Tensor(im_array.transpose(0, 3, 1, 2)).to(DEVICE))[1].detach().cpu().numpy()[:, :, 0, 0]
        # print(f.shape)
        f = model(torch.Tensor(im_array.transpose(0, 3, 1, 2)).to(DEVICE)).detach().cpu().numpy()[:, :, 0, 0]
        # print(f.shape)
        start = c * batch_size
        end = min((c + 1) * batch_size, num_faces)
        # This is different from the Keras model where the normalization has been done inside the model.
        face_feats[start:end] = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))
    # print(face_feats.shape)
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
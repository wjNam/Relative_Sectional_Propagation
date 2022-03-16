import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from modules.vgg import vgg16
from modules.resnet import resnet50
import render
import imageio
from torch.autograd import Variable
from baselines.gradcam import GradCam


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input



def get_args():
    parser = argparse.ArgumentParser(description='Relative Sectional Propagation')
    parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                        help='Model architecture')
    parser.add_argument('--resume', type=str, default='./run/VOC/VGG/vgg_new.pth.tar',
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='model_best',
                        help='set the checkpoint name')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--output-path', type=str, default='./result',
                        help='output result path')
    args = parser.parse_args()
    return args


def visualize(relevances, img_name):
    heatmap = np.sum(relevances, axis=3)
    heatmaps = []
    for h, heat in enumerate(heatmap):
        maps = render.hm_to_rgb(heat, scaling=3, sigma=1, cmap='seismic')
        heatmaps.append(maps)
        imageio.imsave('./result/heatmap/' + img_name + '.jpg', maps, vmax=1, vmin=-1)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def relative_GAM(num_pred, pd_cat, bp):
    cam_rel_tmp = []
    cam_rel_tmp2 = []
    for m in range(num_pred):
        cam_tmp = bp.generate_cam_without_sum(input, int(pd_cat[m].detach().cpu().numpy()))
        cam_tmp = cam_tmp.detach().cpu().numpy()
        cam_tmp = cam_tmp / (cam_tmp.max() + 1e-9)
        if m == g:
            cam_rel_tmp.append(cam_tmp)
        else:
            cam_rel_tmp2.append(cam_tmp)
    rel_tmp = np.sum(cam_rel_tmp, 0) * (len(cam_rel_tmp2)) - np.sum(cam_rel_tmp2, 0)
    pos = np.maximum(rel_tmp, 0)
    neg = np.minimum(rel_tmp, 0)
    pos = pos / (pos.sum() + 1e-9)
    neg = neg / neg.min()
    neg = neg / (neg.sum() + 1e-9) * pos.sum()
    rel_tmp2 = np.zeros_like(rel_tmp)
    rel_tmp2 += pos
    rel_tmp2 -= neg
    rel = Variable(torch.tensor(rel_tmp2)).cuda()
    rel_sum = rel.sum([1], keepdim=True)
    pos_val = rel_sum.gt(0).type(rel.type()) * rel_sum
    neg_val = rel_sum.le(0).type(rel.type()) * rel_sum
    pos_val = pos_val / (pos_val.sum() + 1e-9) * 2
    neg_val = neg_val / (neg_val.sum() + 1e-9)
    rel2 = pos_val - neg_val
    rel = torch.ones_like(rel) * rel2 / rel.shape[1]
    return rel
def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    print('Pred cls : '+str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = Variable(T).cuda()
    return Tt
if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    parser = argparse.ArgumentParser(description='Interpreting the decision of classifier')
    parser.add_argument('--method', type=str, default='RSP', metavar='N',
                        help='Method : LRP or RSP')
    parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                        help='Model architecture: vgg / resnet')
    args = parser.parse_args()
    num_workers = 0
    batch_size = 1

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_name = 'imagenet/'

    # define data loader

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./data/' + data_name,
                             transforms.Compose([
                                 transforms.Scale([224, 224]),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    if args.arc == 'vgg':
        model = vgg16(pretrained=True).cuda()
    elif args.arc == 'resnet':
        model = resnet50(pretrained=True).cuda()
        # model = resnet18(pretrained=True).cuda()
    # method = LRP or RAP
    method = args.method
    model.eval()
    saliency_layer = 29
    bp = GradCam(model, target_layer=saliency_layer)
    for idx, (input, labels) in enumerate(val_loader):

        img_name = val_loader.dataset.imgs[idx][0].split('\\')[1]
        input = Variable(input, volatile=True).cuda()
        input.requires_grad = True

        output = model(input)
        T = compute_pred(output)
        tmp_lb = output.data.max(1, keepdim=True)[1].squeeze(1).item()

        cam_tmp = bp.generate_cam_without_sum(input, tmp_lb)
        cam_tmp = cam_tmp.detach().cpu().numpy()
        cam_tmp = cam_tmp / (cam_tmp.max() + 1e-9)
        rel = Variable(torch.tensor(cam_tmp)).cuda()
        RAP = model.RSP(R=rel)
        Res = (RAP).sum(dim=1, keepdim=True)
        Res_norm = Res * Res.ge(0).type(Res.type())
        Res_norm = Res_norm / Res_norm.max()
        Res_norm = Res_norm[0]
        # save results
        heatmap = Res[0].permute(1, 2, 0).data.cpu().numpy()
        visualize(heatmap.reshape([batch_size, heatmap.shape[0], heatmap.shape[1], 1]), img_name)

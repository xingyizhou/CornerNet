import numpy as np
import cv2
import torch

def visualize(image, tl_heat, br_heat):

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    colors = [((np.random.random((1, 1, 3)) * 0.6 + 0.4)*255).astype(np.uint8)\
               for _ in range(tl_heat.shape[1])]

    tl_hm = _gen_colormap(tl_heat[0].detach().cpu().numpy(), colors)
    br_hm = _gen_colormap(br_heat[0].detach().cpu().numpy(), colors)

    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(3, 1, 1)
    img = (image[0].detach().cpu().numpy() * std + mean) * 255
    img = img.astype(np.uint8).transpose(1, 2, 0)

    tl_blend = _blend_img(img, tl_hm)
    br_blend = _blend_img(img, br_hm)
    cv2.imshow('tl_heatmap', tl_blend)
    cv2.imshow('br_heatmap', br_blend)
    # cv2.waitKey()

def _gen_colormap(heatmap, colors):
    num_classes = heatmap.shape[0]
    h, w = heatmap.shape[1], heatmap.shape[2]
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num_classes):
      color_map = np.maximum(
        color_map, (heatmap[i, :, :, np.newaxis] * colors[i]).astype(np.uint8))
    return color_map


def _blend_img(back, fore, trans=0.7):
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
      fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
      fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    ret = (back * (1. - trans) + fore * trans).astype(np.uint8)
    ret[ret > 255] = 255
    return ret
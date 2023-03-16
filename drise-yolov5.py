import argparse
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

import yolov5
from yolov5.utils.datasets import IMG_FORMATS

from DRise import DRise
from utils.conversions import xywhn2xyxy

YOLOv5_FORMATS = ['pt', 'torchscript', 'onnx', 'xml', 'engine', 'mlmodel', 'saved_model', 'pb', 'tflite', 'tflite']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, default='test/Dani14_chunk_25_frame5.png',
                        help='Image or folder with images to analise')
    parser.add_argument('-w', '--weights', type=str, default='test/s-68v2-s-640-64-iw.pt',
                        help='YOLOv5 weights to evaluate')
    parser.add_argument('-o', '--output', type=str, default='output',
                        help='Save directory for images')
    parser.add_argument('--names', type=str, default='test/ts-dmg-v2.names',
                        help='File containing the names of the classes')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Transparency of saliency map overlap, less is more visible')
    parser.add_argument('-n', '--nmasks', type=int, default=5000,
                        help='Number of masks to use')
    parser.add_argument('-b', '--binsize', type=int, default=16,
                        help='Size of the binary mask')
    parser.add_argument('-p', '--prob', type=float, default=0.5,
                        help='Probability of 1 in the mask (P(0)=1-p)')
    parser.add_argument('--labels', type=str, default='labels',
                        help='Folder name of labels with ground truth')
    parser.add_argument('--other-folder', action='store_true',
                        help='If annotations files are not in the same folder')
    parser.add_argument('--predictions', action='store_true',
                        help='Use YOLOv5 predictions as target bounding boxes')
    opt = parser.parse_args()

    images = []
    impath = ''
    if os.path.isfile(opt.input):
        assert os.path.splitext(opt.input)[1][1:] in IMG_FORMATS, f'Image {opt.input} is not a compatible format.' \
                                                                  f'Compatible formats are:\n {IMG_FORMATS}'
        impath, images = os.path.split(opt.input)
        impath = Path(impath)
        images = [images]
    elif os.path.isdir(opt.input):
        impath = Path(opt.input)
        images = sorted(os.listdir(impath))
        images = [im for im in images if os.path.splitext(im)[1][1:] in IMG_FORMATS]
        assert len(images) > 0, f'No compatible images found in {impath}. Compatible formats are:\n {IMG_FORMATS}'
        print(f'Found {len(images)} images to analyse using D-RISE')
    else:
        assert True, print(f'{opt.input} is not a reacheable file or a dir')

    assert os.path.splitext(opt.weights)[1][1:] in YOLOv5_FORMATS, f'Weights format not compatible. Valid formats are:\n' \
                                                                   f'{YOLOv5_FORMATS}'
    assert os.path.isfile(opt.weights), f'{opt.weights} not found'

    save_dir = Path(opt.output)
    os.makedirs(save_dir, exist_ok=True)

    with open(opt.names, mode='r') as f:
        names = f.readlines()
    names = [n.strip() for n in names]

    if opt.other_folder:
        assert os.path.isdir(opt.labels), f'Folder for labels {opt.labels} not found'
        lbpath = Path(opt.labels)
    else:
        lbpath = Path(impath)
    # labels = [os.path.splitext(im)[0] + '.txt' for im in images]

    detector = yolov5.load(opt.weights)
    predictions = opt.predictions

    for i, imname in enumerate(images):
        print(f'Image {imname} ({i+1}/{len(images)})')
        detections = detector(impath / imname)
        img = detections.imgs[0]

        if predictions:
            target = np.asarray(detections.pred[0])
        else:
            lb = lbpath / (os.path.splitext(imname)[0] + '.txt')
            target = np.loadtxt(lb)
            H, W = img.shape[:2]
            target[:, 1:] = xywhn2xyxy(target[:, 1:], W, H)
        # print(target)
        # detections.display(show=True, save=True, save_dir=Path('output'))
        drise = DRise(img, target, names, nmasks=opt.nmasks, binsize=opt.binsize, prob=opt.prob, pred=predictions)
        detections.display(save=True, save_dir=save_dir)

        for masked in tqdm(drise):
            predictions = detector(masked)
            # print(predictions.pred[0])
            drise.update_saliency_maps(predictions.pred[0])

        images, im_names = drise.images_with_saliency_maps(alpha=0.3)
        j = 0
        for im, cls in zip(images, im_names):
            nn = os.path.splitext(imname)[0]
            name = f'{save_dir}/{nn}-{j}-{cls}.png'
            j += 1
            cv2.imwrite(name, im)
    '''
    groundt = np.loadtxt(im_name.replace('.png', '.txt'))
    # print(groundt)

    image = cv2.imread(im_name)
    H, W = image.shape[:2]

    groundt[:, 1:] = xywhn2xyxy(groundt[:, 1:], W, H)
    # print('GT')
    # print(groundt)

    drise = DRise(image, groundt, names, nmasks=5000)
    # masked = drise.__next__()

    # cv2.imshow('masked', masked)
    # cv2.waitKey(0)


    detections = detector(image)
    # print('detections')
    detections.display(show=True, save=True, save_dir=Path('output'))
    # print(detections.files)

    for masked in tqdm(drise):
        predictions = detector(masked)
        # predictions.display(show=True)
        # print(predictions.pred[0])
        drise.update_saliency_maps(predictions.pred[0])

    images, im_names = drise.images_with_saliency_maps(alpha=0.3)
    i = 0
    for im, cls in zip(images, im_names):
        name = f'output/img-{i}-{cls}.png'
        i += 1
        cv2.imwrite(name, im)

    # plt.imshow(images[10])

    '''
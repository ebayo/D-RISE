import cv2
import numpy as np

from utils.masks import MaskGenerator

# TODO: check detection_target is list of integers


class Rise:
    def __init__(self, image, detection_target, nmasks=5000, binsize=16, prob=0.5, names=None, probabilities=None):
        # detection_target is T x [cls] or T x [cls, conf]
        self.mask_gen = MaskGenerator(binsize, prob)
        self.N = nmasks
        self.lastMask = None
        self.generated = 0
        self.names = names

        # Allow for multiclass classification
        if type(detection_target) in [str, int]:
            detection_target = [detection_target]

        if probabilities is not None:
            if type(probabilities) is float:
                probabilities = [probabilities]

        assert len(detection_target) > 0, 'Define class target(s)'

        self.dt = np.asarray(detection_target)

        self.nc = len(names) if names is not None else max(detection_target)
        T = self.dt.shape[0]
        if probabilities is not None:
            assert len(probabilities) == T, \
                f'List of probabilities must be the same length as detection_target. Got' \
                f'len(probabilities) = {len(probabilities)} and len(detection_target) = {T}'
            self.dt_prob = np.asarray(probabilities)
        else:
            # Take them as labeled ground truth
            self.dt_prob = np.ones((T,))

        self.image = np.copy(image)
        self.H, self.W = image.shape[:2]

        # Is list of classes (as integers), may have only one, still treated as list

        self.maps = np.zeros((T, self.H, self.W))
        self.maskSum = np.zeros((self.H, self.W))
        self.avgWeights = np.zeros((T,))

    def __len__(self):
        return self.N

    def __iter__(self):
        self.generated = 0
        self.lastMask = None
        # TODO: what else do we need to reset --> look at the use of the method
        return self

    def __next__(self):
        if self.generated == self.N:
            raise StopIteration
        mask = self.mask_gen.get_mask(self.H, self.W)
        self.lastMask = mask
        self.maskSum += mask
        self.generated += 1
        return ((self.image.astype(np.float32) / 255 * np.dstack([mask] * 3)) * 255).astype(np.uint8)

    def get_masked_image(self):
        return self.mask_gen.mask_image(self.image)

    def update_saliency_maps(self, detections):
        # detections is len(list) = nc, list of confidence scores
        detections = np.asarray(detections)
        for i, dt in enumerate(self.dt):
            wi = detections[dt]
            self.maps[i] = np.add(self.maps[i], wi * self.lastMask)
            self.avgWeights[i] += wi
        self.lastMask = None

    def images_with_saliency_maps(self, alpha=0.5):
        txt_color = (0, 0, 0)
        lw = 2
        tf = max(lw - 1, 1)
        beta = 1 - alpha
        gamma = 0

        images = []
        im_names = []
        # Normalize all maps at the same time --> looses range of variations in images with high probabilty
        # self.maps = cv2.normalize(self.maps, self.maps, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        for dt, smap, a, p in zip(self.dt, self.maps, self.avgWeights, self.dt_prob):
            mapmax = np.max(smap)
            mapmin = np.min(smap)
            smap = cv2.normalize(smap, smap, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            jetmap = cv2.applyColorMap(smap, cv2.COLORMAP_JET)

            # dis_im = self.image.copy()
            dis_im = cv2.addWeighted(self.image, alpha, jetmap, beta, gamma)

            label = self.names[dt] if self.names is not None else str(dt)
            cv2.putText(dis_im, f'label: {label}', (10, 20), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(dis_im, f'Initial Probability: {p:.3e}', (10, 40), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(dis_im, f'min: {mapmin:.6f}, max: {mapmax:.6f}', (10, 60), 0, lw / 3,
                        txt_color, thickness=tf, lineType=cv2.LINE_AA)
            images.append(dis_im)
            im_names.append(label)

        return images, im_names

    def get_total_mask(self):
        mapmin = np.min(self.maskSum)
        mapmax = np.max(self.maskSum)
        smap = np.zeros(self.maskSum.shape)
        return (mapmin, mapmax), cv2.normalize(self.maskSum, smap, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

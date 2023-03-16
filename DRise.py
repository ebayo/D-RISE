# Main entry point of the algorithm
import cv2
import numpy as np

from utils.masks import MaskGenerator
import utils.metrics as metrics

# TODO: batch processing of images (use same masks for all)
# TODO: check correct dimensions of detection_target if pred


class DRise:
    def __init__(self, image, detection_target, classes, nmasks=5000, binsize=16, prob=0.5, obj=False, simplified=True,
                 pred=False, RGB=False):
        # Ground truth expected as [cls, x1, y1, x2, y2] as would be read from a file
        # Propieties using names from paper
        # If simplified --> use one-hot encoding for target prediction and use simplified metric --> see README.md
        # RGB --> image stored in RGB format, returned as such as well

        self.mask_gen = MaskGenerator(binsize, prob)
        self.N = nmasks
        self.lastMask = None
        self.generated = 0
        self.obj = obj  # If detector used provides with objectness score
        self.metric = metrics.similarity_metric_simplified if simplified else metrics.similarity_metric
        self.simplified = simplified
        self.names = None
        self.RGB = RGB

        if type(classes) is list:
            self.names = classes
            classes = len(classes)

        assert classes > 0, 'Define number of classes'
        self.nc = classes

        self.image = np.copy(image)
        self.H, self.W = image.shape[:2]

        dt = np.asarray(detection_target)
        T = dt.shape[0]

        assert T > 0, 'Initialise with at least one target prediction or ground truth'

        if pred:
            self.dt = dt
        elif type(dt[0]) is not np.ndarray:  # Single object in ground truth
            if simplified:
                one_hot = [1, dt[0]]
            else:
                one_hot = np.zeros(self.nc)
                one_hot[int(detection_target[0])] = 1

            dt = (detection_target[1:], [1], one_hot) if obj else (detection_target[1:], one_hot)
            dt = np.concatenate(dt, dtype=float)  # TODO: check if we need axis=1
            T = 1
            self.dt = np.reshape(dt, (1, dt.shape[0]))
        else:
            if simplified:
                one_hot = np.ones((T, 2))
                one_hot[:, 1] = dt[:, 0]  # [conf = 1, cls]
            else:
                one_hot = np.zeros((T, self.nc))
                for i, j in enumerate(dt[:, 0]):
                    one_hot[i, int(j)] = 1
            dt = (dt[:, 1:], np.ones((T, 1)), one_hot) if obj else (dt[:, 1:], one_hot)
            self.dt = np.concatenate(dt, dtype=float, axis=1)
        # print(self.dt)
        # Initialize saliency maps
        self.maps = np.zeros((T, self.H, self.W))
        self.addedMasks = np.zeros((T,))
        self.maskSum = np.zeros((self.H, self.W))

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

    def __len__(self):
        return self.N

    def get_masked_image(self):
        return self.mask_gen.mask_image(self.image)

    def update_saliency_maps(self, detections):
        detections = np.asarray(detections)
        for i, dt in enumerate(self.dt):
            wi = max([self.metric(dt, dj, self.obj) for dj in detections], default=0)  # account for no detections
            if wi > 0:
                self.maps[i] = np.add(self.maps[i], wi * self.lastMask)
                self.addedMasks[i] += 1
        self.lastMask = None

    def images_with_saliency_maps(self, alpha=0.5):
        color = (0, 255, 0)
        txt_color = (0, 0, 0)
        lw = 2
        tf = max(lw - 1, 1)
        beta = 1 - alpha
        gamma = 0

        images = []
        im_names = []

        for dt, smap, a in zip(self.dt, self.maps, self.addedMasks):
            # print(dt)
            cls = dt[-1] if self.simplified else np.argmax(dt[5:])
            # TODO: normalize all with the same values?? Divide by number of masks, by a max value?
            # smap = smap * a / self.N
            smap = cv2.normalize(smap, smap, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            # smap = smap.astype(np.uint8)
            jetmap = cv2.applyColorMap(smap, cv2.COLORMAP_JET)

            dis_im = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR) if self.RGB else self.image.copy()
            dis_im = cv2.addWeighted(dis_im, alpha, jetmap, beta, gamma)
            # cv2.imshow('heatmap', dis_im)
            # cv2.waitKey(0)
            p1, p2 = (int(dt[0]), int(dt[1])), (int(dt[2]), int(dt[3]))
            cv2.rectangle(dis_im, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

            label = str(int(dt[-1]))
            if self.names is not None:
                label = self.names[int(cls)]
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(dis_im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(dis_im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(dis_im, f'Contributing Masks: {a}/{self.N}', (10, 20), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

            # cv2.imshow(label, dis_im)
            # cv2.waitKey(0)
            images.append(dis_im)
            im_names.append(label)

        return images, im_names

    def get_total_mask(self):
        mapmin = np.min(self.maskSum)
        mapmax = np.max(self.maskSum)
        smap = np.zeros(self.maskSum.shape)
        return (mapmin, mapmax), cv2.normalize(self.maskSum, smap, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

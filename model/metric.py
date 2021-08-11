import numpy as np


class Evaluator:
    def __init__(self, num_class, seen_classes_idx=None, unseen_classes_idx=None):
        self.num_class = num_class
        self.seen_classes_idx = seen_classes_idx
        self.unseen_classes_idx = unseen_classes_idx
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        if self.seen_classes_idx and self.unseen_classes_idx:
            Acc_seen = (
                np.diag(self.confusion_matrix)[self.seen_classes_idx].sum()
                / self.confusion_matrix[self.seen_classes_idx, :].sum()
            )
            Acc_unseen = (
                np.diag(self.confusion_matrix)[self.unseen_classes_idx].sum()
                / self.confusion_matrix[self.unseen_classes_idx, :].sum()
            )
            return {'harmonic':2*Acc_seen*Acc_unseen/(Acc_seen+Acc_unseen), 'seen':Acc_seen, 'unseen':Acc_unseen, 'overall':Acc}
        else:
            return {'overall':Acc}

    def Pixel_Accuracy_Class(self):
        Acc_by_class = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(np.nan_to_num(Acc_by_class))
        if self.seen_classes_idx and self.unseen_classes_idx:
            Acc_seen = np.nanmean(np.nan_to_num(Acc_by_class[self.seen_classes_idx]))
            Acc_unseen = np.nanmean(np.nan_to_num(Acc_by_class[self.unseen_classes_idx]))
            return {'harmonic':2*Acc_seen*Acc_unseen/(Acc_seen+Acc_unseen), 'by_class': Acc_by_class, 'seen':Acc_seen, 'unseen':Acc_unseen, 'overall':Acc}
        else:
            return {'overall':Acc, 'by_class':Acc_by_class}

    def Mean_Intersection_over_Union(self):
        MIoU_by_class = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(np.nan_to_num(MIoU_by_class))
        if self.seen_classes_idx and self.unseen_classes_idx:
            MIoU_seen = np.nanmean(np.nan_to_num(MIoU_by_class[self.seen_classes_idx]))
            MIoU_unseen = np.nanmean(
                np.nan_to_num(MIoU_by_class[self.unseen_classes_idx])
            )
            return {'harmonic':2*MIoU_seen*MIoU_unseen/(MIoU_seen+MIoU_unseen), 'by_class':MIoU_by_class, 'seen':MIoU_seen, 'unseen':MIoU_unseen, 'overall': MIoU}
        else:
            return {'overall': MIoU, 'by_class':MIoU_by_class}

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        if self.seen_classes_idx and self.unseen_classes_idx:
            FWIoU_seen = (
                freq[self.seen_classes_idx][freq[self.seen_classes_idx] > 0]
                * iu[self.seen_classes_idx][freq[self.seen_classes_idx] > 0]
            ).sum()
            FWIoU_unseen = (
                freq[self.unseen_classes_idx][freq[self.unseen_classes_idx] > 0]
                * iu[self.unseen_classes_idx][freq[self.unseen_classes_idx] > 0]
            ).sum()
            return {'overall':FWIoU, 'seen':FWIoU_seen, 'unseen':FWIoU_unseen}
        else:
            return {'overall':FWIoU}

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
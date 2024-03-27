"""
Code: Metrics
Desc: This code heavily borrowed from https://github.com/lartpang with slight modifications.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve, distance_transform_edt as bwdist

import sys
sys.path.append("..")
from utils.my_utils import borders_capture

_EPS = 1e-16
_TYPE = np.float64

def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt

def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)

# def Borders_Capture(gt,pred,dksize=15):
#     gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
#     ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#     contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     img=gt.copy()
#     img[:]=0
#     cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
#     kernel = np.ones((dksize, dksize), np.uint8)
#     img_dilate = cv2.dilate(img, kernel)

#     res = cv2.bitwise_and(img_dilate, gt)
#     b, g, r = cv2.split(res)
#     alpha = np.rollaxis(img_dilate, 2, 0)[0]
#     merge = cv2.merge((b, g, r, alpha))

#     resp = cv2.bitwise_and(img_dilate, pred)
#     b, g, r = cv2.split(resp)
#     alpha = np.rollaxis(img_dilate, 2, 0)[0]
#     mergep = cv2.merge((b, g, r, alpha))

#     merge = cv2.cvtColor(merge, cv2.COLOR_RGB2GRAY)
#     mergep = cv2.cvtColor(mergep, cv2.COLOR_RGB2GRAY)
#     return merge,mergep,np.sum(img_dilate)/255

# MAE
class MAE(object):
    def __init__(self):
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray, area = None):
        pred, gt = _prepare_data(pred, gt)

        mae = self.cal_mae(pred, gt, area)
        self.maes.append(mae)

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray, area) -> float:
        if area is not None:
            mae = np.sum(np.abs(pred - gt))/np.sum(area)
        else:
            mae = np.mean(np.abs(pred - gt))
        return mae

    def get_results(self) -> dict:
        mae = np.mean(np.array(self.maes, _TYPE))
        return dict(mae=mae)

# S measure
class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)
        #return sm

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1])
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info['weight']
        pred1, pred2, pred3, pred4 = part_info['pred']
        gt1, gt2, gt3, gt4 = part_info['gt']
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        h, w = matrix.shape
        if matrix.sum() == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            area_object = np.sum(matrix)
            row_ids = np.arange(h)
            col_ids = np.arange(w)
            x = np.round(np.sum(np.sum(matrix, axis=0) * col_ids) / area_object)
            y = np.round(np.sum(np.sum(matrix, axis=1) * row_ids) / area_object)
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x, y) -> dict:
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(gt=(gt_LT, gt_RT, gt_LB, gt_RB),
                    pred=(pred_LT, pred_RT, pred_LB, pred_RB),
                    weight=(w1, w2, w3, w4))

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(sm=sm)

# E measure
class Emeasure(object):
    def __init__(self):
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel, fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel, pred_bg_numel=bg___numel,
            )

            results_parts = []
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / \
                                     (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs, fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs, pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / \
                                     (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def generate_parts_numel_combinations(self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value)
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=_TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=_TYPE), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))

# weighted F measure
class WeightedFmeasure(object):
    def __init__(self, beta: float = 1):
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])


        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + _EPS)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + _EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m: m + 1, -n: n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=_TYPE))
        return dict(wfm=weighted_fm)

def evaluation(gt_root, pred_root, dataloader, save_png=None, save_txt=None):
    SM     = Smeasure()
    EM     = Emeasure()
    _MAE   = MAE()
    BR_MAE = MAE()
    WFM    = WeightedFmeasure()
    BR_WF  = WeightedFmeasure()

    step_n = len(dataloader)
    with tqdm(total=step_n) as pbar:
        for _, (_, _, _, _, meta_info) in enumerate(dataloader):
            n = len(meta_info)
            for i in range(n):
                name = '{}_{:0>5d}.png'.format(meta_info['category'][i], meta_info['index'][i])
                gt_path = os.path.join(gt_root, name)
                pred_path = os.path.join(pred_root, name)
                
                assert os.path.exists(gt_path)
                assert os.path.exists(pred_path)
                
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                # gt_width, gt_height = gt.shape
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                # pred_width, pred_height = pred.shape
                if gt.shape != pred.shape:
                    # pred = cv2.resize(pred, gt.shape[::-1])
                    cv2.imwrite(pred_path, cv2.resize(pred, gt.shape[::-1]))
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                # S measure
                SM.step(pred=pred, gt=gt)
                # E measure
                EM.step(pred=pred, gt=gt)
                # MAE, Weighted F measure
                WFM.step(pred=pred, gt=gt)
                _MAE.step(pred=pred, gt=gt)
                # BR_gt, BR_pred, area=Borders_Capture(cv2.imread(gt_path), cv2.imread(pred_path), 15)
                BR_gt, BR_pred, area = borders_capture(cv2.imread(gt_path), cv2.imread(pred_path), 15)
                if area > 0:
                    BR_MAE.step(pred=BR_pred, gt=BR_gt,area=area)
                    BR_WF.step(pred=BR_pred, gt=BR_gt)
                    # save predictions
                    if save_png:
                        cv2.imwrite(os.path.join(save_png, name), BR_pred)
            pbar.update()
                    

    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = _MAE.get_results()['mae']
    br_mae= BR_MAE.get_results()['mae']
    wfm = WFM.get_results()['wfm']
    br_wf = BR_WF.get_results()['wfm']

    sm_r = str(sm.round(3))
    em_r = str('-' if em['curve'] is None else em['curve'].mean().round(3))
    wfm_r = str(wfm.round(3))
    mae_r = str(mae.round(3))
    brmae_r = str(br_mae.round(3))
    brwf_r = str(br_wf.round(3))
    
    eval_record = f'Smeasure: {sm_r}\n, MeanEm: {em_r}\n, WeightedFmeasure: {wfm_r}\n, MAE: {mae_r}\n, BoundaryMAE: {brmae_r}\n, BoundaryWeightedFmeasure: {brwf_r}\n'

    print(eval_record)
    print('#'*50)

    if save_txt: 
        f = open(save_txt, 'a')
        f.write(eval_record)
        f.write("\n")
        f.close() 
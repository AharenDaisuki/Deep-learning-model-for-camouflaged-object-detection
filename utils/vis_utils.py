import torch
from matplotlib import pyplot as plt

def vis_feature_map(f, title=None, policy='average'):
    batch_n = f.shape[0]
    feature_maps = []
    for i in range(batch_n):
        if f[i].ndim == 4:
            fi = f[i].squeeze(0) # i-th batch of feature maps
        elif f[i].ndim == 3:
            fi = f[i]
        assert fi.ndim == 3, f'Invalid shape: {fi.shape}'
        if policy=='average':
            fi = torch.sum(fi, 0)
            fi /= fi.shape[0] # average along the channel
        elif policy=='first':
            fi = fi[0] # first channel
        assert fi.ndim > 1, f'Invalid shape: {fi.shape}'
        feature_maps.append(fi)
    fig = plt.figure(figsize=(18, 32))
    for idx, feature in enumerate(feature_maps):
        # print(feature.shape)
        tmp = fig.add_subplot(1, len(feature_maps), idx+1)
        img_plot = plt.imshow(feature.detach().numpy())
        # TODO: save
        if title:
            tmp.set_title('{}: batch {}'.format(title, idx), fontdict = {'fontsize':7})
            # tmp.title.set_text('{}: batch {}'.format(title, idx), fontdict = {'fontsize':10})
        tmp.axis('off')
    plt.show()
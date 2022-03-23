import torch

def keypoints_from_output(heatmaps, o_img_size, k=10):
    heatmaps = aggregate_heatmaps(heatmaps, o_img_size)
    return keypoints_from_heatmaps(heatmaps, k)

def resize_heatmaps(hms, size):
    return [
        torch.nn.functional.interpolate(
            hm,
            size,
            mode='bilinear',
            align_corners=False
        )
        for hm in hms
    ]
def aggregate_heatmaps(hms, size):
    hms = resize_heatmaps(hms, size) # resize heatmaps to same resolution
    hms = (hms[0] + hms[1])/2.0 # avg heatmaps of different original resolutions
    return hms

def keypoints_from_heatmaps(hms, k=10):
    
    h = hms.size(2)
    w = hms.size(3)
    hms = hms.view(*hms.shape[:2], -1)
    values, indices = torch.topk(hms, k, dim=2)
    x = indices % w
    y = (indices / w).long()
    return torch.stack((y,x), dim=3)
    
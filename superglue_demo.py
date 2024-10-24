from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import matplotlib
from superglue_models.matching import Matching
from superglue_models.utils import (AverageTimer,read_image,make_matching_plot, frame2tensor)

torch.set_grad_enabled(False)

if __name__ == '__main__':
    image0_path = 'dataset_tomato01/img0006.png'
    image1_path = 'dataset_tomato01/img0008.png'
    resize = [640, 480]
    superglue_weights = 'indoor'
    

    max_keypoints = 560  
    keypoint_threshold = 0.05  
    nms_radius = 6  
    sinkhorn_iterations = 20
    match_threshold = 0.00000001
    show_keypoints = True
    output_dir = 'output1'


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running inference on device "{device}"')
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue_weights,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']


    image0, inp0, scales0 = read_image(image0_path, device, resize)
    image1, inp1, scales1 = read_image(image1_path, device, resize)

    last_data = matching.superpoint({'image': inp0})
    last_data = {k + '0': last_data[k] for k in keys}
    last_data['image0'] = inp0
    last_frame = image0

    frame_tensor = frame2tensor(image1, device)
    pred = matching({**last_data, 'image1': frame_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = matplotlib.cm.jet(confidence[valid])
    text = [
        
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {:06}:{:06}'.format(0, 1),
    ]
    out = make_matching_plot(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=output_dir, show_keypoints=show_keypoints, small_text=small_text)

    cv2.imshow('Tomato', out)
    cv2.waitKey(0)

    if output_dir is not None:
        out_file = str(Path(output_dir, 'matches.png'))
        print(f'\nWriting image to {out_file}')
        cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()

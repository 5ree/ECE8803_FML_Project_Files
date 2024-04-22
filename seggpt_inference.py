import os
import argparse
import sys

import torch
import numpy as np
from PIL import Image

from seggpt_engine import inference_image, inference_video
import models_seggpt


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--input_image', type=str, help='path to input image to be tested',
                        default=None)
    parser.add_argument('--prompt_image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt_target', type=str, nargs='+', help='path to prompt target',
                        default=None)
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model

if __name__ == '__main__':
    args = get_args_parser()

    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
    print('Model loaded.')

    samples = np.load('/storage/ice1/0/3/sprathipati6/fml/proj/samples.npy', allow_pickle=True)
    labels = np.load('/storage/ice1/0/3/sprathipati6/fml/proj/labels.npy', allow_pickle=True)

    my_ph0_dir = "/home/hice1/sprathipati6/scratch/fml/proj/Prompting results/Bus/st1"
    mask_dir = "/home/hice1/sprathipati6/scratch/fml/proj/Prompting results/Bus/st1/masks"
    all_mask_files = os.listdir(mask_dir)

    input_img = Image.fromarray(samples[0]).save("input_img.png")
    prompt_img = Image.fromarray(samples[95]).save("prompt_img.png")

    args.prompt_image = ["prompt_img.png"]
    args.prompt_target = ["/storage/ice1/0/3/sprathipati6/fml/proj/Prompting results/Bus/st1/masks/95_0_mask.png"]

    input_imgs_dir = "/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/input_images"
    output_imgs_dir = "/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/output_images"
    output_masks_dir = "/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/output_masks"
    ground_truths_dir = "/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/ground_truths"

    args.gen_data = False

    if args.gen_data:
        for i in range(len(samples)):
            if i != 95:
                print(f"Processing sample# {i}")
                input_img = samples[i]

                # Create a dir with sample num
                input_dir = os.path.join(input_imgs_dir, str(i))
                os.makedirs(input_dir, exist_ok=True)

                output_dir = os.path.join(output_imgs_dir, str(i))
                os.makedirs(output_dir, exist_ok=True)

                output_mask_dir = os.path.join(output_masks_dir, str(i))
                os.makedirs(output_mask_dir, exist_ok=True)

                ground_truth_dir = os.path.join(ground_truths_dir, str(i))
                os.makedirs(ground_truth_dir, exist_ok=True)

                # Write input_img to input_dir
                input_img_path = os.path.join(input_dir, "input_img.png")
                input_img = Image.fromarray(input_img).save(input_img_path)

                ground_truth_path = os.path.join(ground_truth_dir, "ground_truth.png")
                ground_truth = Image.fromarray(labels[i]).save(ground_truth_path)

                # Create out_path and mask_path
                out_path = os.path.join(output_dir, "output_img.png")
                mask_path = os.path.join(output_mask_dir, "output_mask.png")

                inference_image(model, device, input_img_path, args.prompt_image, args.prompt_target, out_path, mask_path)

    ph0_ious = {}
    seggpt_ious = {}

    for i in range(len(samples)):
        if i != 95:

            label = labels[i]

            gen_bin_mask = np.array(Image.open(f"/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/output_masks/{i}/output_mask.png"))

            # gen_bin_mask = gen_bin_mask.mean(axis=2)

            filtered_sorted_files = sorted(
                [f for f in all_mask_files if f.startswith(f'{i}_') and f.endswith('_mask.png')],
                key=lambda x: int(x.split('_')[1])
            ) 
            
            if (len(filtered_sorted_files) != 0):
                last_file = filtered_sorted_files[-1] 
                ph0_mask = np.array(Image.open(os.path.join(mask_dir, last_file)))

                # Calculate IoU
                intersection = np.logical_and(gen_bin_mask, label)
                union = np.logical_or(gen_bin_mask, label)
                iou_score = np.sum(intersection) / np.sum(union)
                seggpt_ious[i] = iou_score

                intersection = np.logical_and(ph0_mask, label)
                union = np.logical_or(ph0_mask, label)
                iou_score = np.sum(intersection) / np.sum(union)
                ph0_ious[i] = iou_score
            else:
                print(f"Skipping for {i}")
    
    for i in ph0_ious:
        seg = seggpt_ious[i]
        ph0 = ph0_ious[i]

        delta = abs(seg - ph0)
        ACCEPTABLE = 0.00

        # if seg > 0.75:
        #     continue

        if delta > ACCEPTABLE:
            if seg > ph0:
                print(f"Sample#: {i}, SegGPT IoU: {seg:6.2f}, Ph0 IoU: {ph0:6.2f}, SegGPT better by {delta:.2f}")
            else:
                print(f"Sample#: {i}, SegGPT IoU: {seg:6.2f}, Ph0 IoU: {ph0:6.2f}, SegGPT worse  by {delta:.2f}")

    avg_ph0 = sum(ph0_ious.values()) / len(ph0_ious)
    avg_seggpt  = sum(seggpt_ious.values()) / len(seggpt_ious)


    all_my_all_good = [0, 1, 2, 3, 7, 8, 9, 11, 13, 15, 17, 18, 19, 21, 22, 24, 25, 28, 29, 31, 32, 35, 36, 40, 43, 44, 45, 46, 48, 51, 52, 57, 60, 61, 63, 64, 65, 67, 69, 71, 72, 73, 74, 77, 79, 80, 81, 85, 89, 90, 92, 94, 96, 99, 100, 101, 102, 103, 104, 106, 107, 109, 113, 114, 116, 119, 121, 122, 123, 124, 125, 127, 128, 129, 130, 131, 132, 133, 136, 137, 139, 140, 141, 142, 143, 145, 148, 150, 151, 152, 153, 154, 155, 156, 158, 159, 160, 161, 165, 166, 170, 171, 172, 174, 175, 176, 177, 178, 179, 180, 181, 182, 185, 186, 187, 188, 191, 193, 194, 195, 196, 197, 198, 199, 202, 205, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 220, 222, 226, 228, 229, 230, 231, 233, 234, 237, 240, 241, 242, 244, 245, 246, 247, 248, 251, 252, 253, 255, 256, 257, 258, 259, 260, 262, 263, 264, 266, 267, 268, 269, 270, 271, 272, 273, 274, 276, 280, 281, 283, 285, 286, 287, 290, 291, 292, 293, 295, 296, 297, 298, 299, 300, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 330, 331, 332, 333, 335, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 348, 349, 350, 351, 352, 353, 354, 356, 358, 359, 363, 364, 365, 366, 367, 371, 373, 377, 378, 380, 381, 383, 385, 386, 387, 388, 390, 393, 395, 396, 397, 399]

    all_my_all_bad = [4, 6, 10, 12, 23, 26, 30, 34, 37, 42, 47, 54, 56, 62, 68, 82, 83, 84, 87, 88, 97, 98, 108, 111, 115, 118, 120, 134, 146, 162, 164, 168, 173, 200, 201, 204, 219, 221, 224, 225, 227, 261, 265, 277, 284, 288, 301, 302, 315, 329, 336, 355, 368, 372, 374, 375, 379, 384, 389, 391, 394]

    sum_p = 0
    sum_s = 0
    for a in all_my_all_good:
        sum_p += ph0_ious[a]
        sum_s += seggpt_ious[a]

    all_my = all_my_all_good
    print(f"Average good-only IoU for SegGPT = {sum_s/len(all_my)}, Ph) = {sum_p/len(all_my)}")

    sum_p = 0
    sum_s = 0
    for a in all_my_all_bad:
        sum_p += ph0_ious[a]
        sum_s += seggpt_ious[a]

    all_my = all_my_all_bad
    print(f"Average bad-only IoU for SegGPT = {sum_s/len(all_my)}, Ph) = {sum_p/len(all_my)}")

    print(f"Average IoU for SegGPT = {avg_seggpt}, Ph) = {avg_ph0}")

    print('Finished.')

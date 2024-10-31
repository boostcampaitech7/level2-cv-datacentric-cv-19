import argparse
import cv2
import glob
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from drct.archs.DRCT_arch import *
from drct.models import *
import tqdm.notebook

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default="./experiments/pretrained_models/net_g_latest.pth"
    )
    parser.add_argument('--input', type=str, default='datasets', help='input test image folder')
    parser.add_argument('--output', type=str, default='/data/ephemeral/home/srdata', help='output folder')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRCT(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        gc=32,
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)
    
    window_size = 16
    
    # Find all image files in the specified directory and its subdirectories
    img_paths = glob.glob("/data/ephemeral/home/newcode2/code/data/***/**/*.[jp][pn]*[g]", recursive=True)
    
    for imgname in img_paths:
        print('Processing:', imgname)

        # 이미지 읽기 및 전처리
        img = cv2.imread(imgname, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                _, _, h_old, w_old = img.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
                img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
                
                print("Starting image processing...")
                output = test(img, model, args, window_size)
                output = output[..., :h_old * args.scale, :w_old * args.scale]

        except Exception as error:
            print('Error:', error, imgname)
        else:
            print("Post-processing and saving the image...")
            output = output.squeeze().float().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

            # Create the mirrored output path
            relative_path = os.path.relpath(imgname, "/data/ephemeral/home/newcode2/code/data")
            output_path = os.path.join(args.output, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_file_path = f'{output_path}_DRCT-L_X4.png'
            cv2.imwrite(output_file_path, output)
            print("Image saved successfully:", output_file_path)

    torch.cuda.empty_cache()

def test(img_lq, model, args, window_size):
    b, c, h, w = img_lq.size()
    sf = args.scale
    
    tile = min(args.tile or 128, h, w)
    tile_overlap = args.tile_overlap
    
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride))
    E = torch.zeros(b, c, h*sf, w*sf).cpu()
    W = torch.zeros_like(E)

    total_tiles = len(h_idx_list) * len(w_idx_list)
    pbar = tqdm.tqdm(total=total_tiles, desc="Processing tiles")

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            
            with torch.no_grad():
                out_patch = model(in_patch)
            
            out_patch = out_patch.cpu()
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
            W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            
            torch.cuda.empty_cache()
            pbar.update(1)

    pbar.close()
    output = E.div_(W)
    print("Finished processing all tiles")
    return output

if __name__ == '__main__':
    main()

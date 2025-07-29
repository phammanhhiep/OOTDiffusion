from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location
import argparse

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import os
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


def get_image_name(image_file_name):
    return image_file_name.split(".")[0]


def get_data_names(root_dir, pair_pth=None, cloth_dir=None):
    """Summary
    
    Args:
        pair_pth (None, optional): Description
        cloth_dir (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    person_names = []
    cloth_names = []
    if pair_pth:
        with open(pair_pth, "r") as fd:
            for line in fd.readlines():
                pname, cname = line.strip().split()
                person_names.append(pname)
                cloth_names.append(cname)
    else:
        cloth_names = get_file_paths(
            os.path.join(root_dir, cloth_dir), only_name=True
        )
        person_names = cloth_names
    return person_names, cloth_names


def create_composite_image(images):
    """
    Create a composite image by arranging a list of PIL Images in a row.
    Assumes all images are the same height.
    """
    if not images:
        return None

    # Get dimensions
    width, height = images[0].size
    total_width = width * len(images)
    
    # Create a new blank image
    composite_img = Image.new('RGB', (total_width, height))
    
    # Paste images in a row
    x_offset = 0
    for img in images:
        composite_img.paste(img, (x_offset, 0))
        x_offset += width
        
    return composite_img


def main(args):
    openpose_model = OpenPose(args.gpu_id)
    parsing_model = Parsing(args.gpu_id)    
    category_dict = ['upperbody', 'lowerbody', 'dress']
    category_dict_utils = ['upper_body', 'lower_body', 'dresses']

    model_type = args.model_type # "hd" or "dc"
    category = args.category # 0:upperbody; 1:lowerbody; 2:dress

    image_scale = args.scale
    n_steps = args.step
    n_samples = args.sample
    seed = args.seed

    if model_type == "hd":
        model = OOTDiffusionHD(args.gpu_id)
    elif model_type == "dc":
        model = OOTDiffusionDC(args.gpu_id)
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    model_image_dir = "image"
    cloth_dir = "cloth"
    root_dir = args.data_root_dir
    (
        person_file_names, 
        cloth_file_names
    ) = get_data_names(root_dir, args.data_pair_path, cloth_dir)
   
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for person_file_name, cloth_file_name in zip(person_file_names, cloth_file_names):

                pname = get_image_name(person_file_name)
                cname = get_image_name(cloth_file_name)        
                model_path = os.path.join(
                    root_dir,
                    model_image_dir,
                    "{}".format(person_file_name)
                )
                cloth_path = os.path.join(
                    root_dir,
                    cloth_dir,
                    "{}".format(cloth_file_name)
                )
                cloth_img = Image.open(cloth_path).resize((768, 1024))
                model_img = Image.open(model_path).resize((768, 1024))
                keypoints = openpose_model(model_img.resize((384, 512)))
                model_parse, _ = parsing_model(model_img.resize((384, 512)))

                mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
                mask = mask.resize((768, 1024), Image.NEAREST)
                mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
                masked_vton_img = Image.composite(mask_gray, model_img, mask)

                images = model(
                    model_type=model_type,
                    category=category_dict[category],
                    image_garm=cloth_img,
                    image_vton=masked_vton_img,
                    mask=mask,
                    image_ori=model_img,
                    num_samples=n_samples,
                    num_steps=n_steps,
                    image_scale=image_scale,
                    seed=seed,
                )
                composite_img = create_composite_image([model_img, masked_vton_img, images[0], cloth_img])
                output_name = f'{pname}_{cname}.png'
                save_path = os.path.join(args.output_dir, output_name)
                composite_img.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run ootd')
    parser.add_argument('--output_dir', type=str, default="", required=False)
    parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
    parser.add_argument('--data_root_dir', type=str, default="", required=True)
    parser.add_argument('--data_pair_path', type=str, default="", required=True)
    parser.add_argument('--model_type', type=str, default="hd", required=False)
    parser.add_argument('--category', '-c', type=int, default=0, required=False)
    parser.add_argument('--scale', type=float, default=2.0, required=False)
    parser.add_argument('--step', type=int, default=20, required=False)
    parser.add_argument('--sample', type=int, default=4, required=False)
    parser.add_argument('--seed', type=int, default=-1, required=False)
    args = parser.parse_args() 

    main(args)
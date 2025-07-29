from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location
import argparse

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import Dataset

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


class VitonhdInferenceDataset(Dataset):
    def __init__(self, 
        root_dir, 
        openpose_model, 
        parsing_model, 
        data_pair_path=""
    ):
        super().__init__()
        self.parsing_model = parsing_model
        self.openpose_model = openpose_model
        self.root_dir = root_dir
        self.model_image_dir = "image"
        self.cloth_dir = "cloth"
        (
            self.person_file_names, 
            self.cloth_file_names
        ) = self.get_data_names(data_pair_path, self.cloth_dir)

    def __getitem__(self, index):
        person_file_name = self.person_file_names[index]
        cloth_file_name = self.cloth_file_names[index]
        pname = self.get_image_name(person_file_name)
        cname = self.get_image_name(cloth_file_name)        
        model_path = os.path.join(
            self.root_dir,
            self.model_image_dir,
            "{}.jpg".format(person_file_name)
        )
        cloth_path = os.path.join(
            self.root_dir,
            self.cloth_dir,
            "{}.jpg".format(cloth_file_name)
        )
        cloth_img = Image.open(cloth_path).resize((768, 1024))
        model_img = Image.open(model_path).resize((768, 1024))
        keypoints = self.openpose_model(model_img.resize((384, 512)))
        model_parse, _ = self.parsing_model(model_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        masked_vton_img = Image.composite(mask_gray, model_img, mask)

        return {
            "cloth_img": cloth_img,
            "model_img": model_img,
            "mask": mask,
            "masked_vton_img": masked_vton_img,
            "pname": pname,
            "cname": cname,
        }

    def __len__(self):
        return len(self.person_file_names)        

    def get_image_name(self, image_file_name):
        return image_file_name.split(".")[0]

    def get_data_names(self, pair_pth=None, cloth_dir=None):
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
                os.path.join(self.root_dir, cloth_dir), only_name=True
            )
            person_names = cloth_names
        return person_names, cloth_names


class DressCodeDataset(Dataset): pass


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

    test_dataset = VitonhdInferenceDataset(
        root_dir=args.data_root_dir, 
        openpose_model=openpose_model, 
        parsing_model=parsing_model, 
        data_pair_path=args.data_pair_path,        
    )    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4,
    )

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for sample in test_dataloader:
                cloth_img = sample["cloth_img"]
                model_img = sample["model_img"]
                masked_vton_img = sample["masked_vton_img"]
                mask = sample["mask"]
                images = model(
                    model_type=model_type,
                    category=category_dict[category],
                    image_garm=cloth_img,
                    image_vton=masked_vton_img,
                    mask=mask,
                    image_ori=model_img,
                    num_samples=args.n_samples,
                    num_steps=args.n_steps,
                    image_scale=args.image_scale,
                    seed=args.seed,
                )
                composite_img = create_composite_image([model_img, masked_vton_img, images[0], cloth_img])
                output_name = f'{sample["pname"]}_{sample["cname"]}.png'
                save_path = os.path.join(args.output_dir, output_name)
                image.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run ootd')
    parser.add_argument('--output_dir', type=str, default="", required=False)
    parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
    parser.add_argument('--data_root_dir', type=str, default="", required=True)
    parser.add_argument('--data_pair_path', type=str, default="", required=True)
    parser.add_argument('--test_batch_size', type=int, default=1, required=True)
    parser.add_argument('--model_type', type=str, default="hd", required=False)
    parser.add_argument('--category', '-c', type=int, default=0, required=False)
    parser.add_argument('--scale', type=float, default=2.0, required=False)
    parser.add_argument('--step', type=int, default=20, required=False)
    parser.add_argument('--sample', type=int, default=4, required=False)
    parser.add_argument('--seed', type=int, default=-1, required=False)
    args = parser.parse_args() 

    main(args)
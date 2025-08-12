from typing import Dict, List, Optional, Tuple, Iterable, Union

import numpy as np
from PIL import Image
import torch


IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def resize(
        image: Image,
        size: Tuple[int, int],
        resample: Image.Resampling = None,
        reducing_gap: Optional[int] = None
) -> np.ndarray: 
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )

    return resized_image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None, 
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(images=images, size=(height, width), resample=resample) for image in images 
    ]

    # Convert each image to a numpy array
    images = [np.array(images) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(images, scale=rescale_factor) for image in images]
    # Normalize the images
    images = [normalize(images, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. the model expects images in the format [channel, height, width]
    images = [images.transpose(2, 0, 1) for image in images]
    return images 

class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokeniser described here: 
        tokens_to_add = {"additional special tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] # These tokens are used for the object detection (bounding boxes)

        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ] # these tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # we will add the BOS EOS tokens ourselves 
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
        **kwds
    ) -> Dict:
        assert len(images) == 1 and len(text) == 1, f"Recived {len(images)} images for {len(text)} prompts"

        pixel_values = process_images(
            images, 
            size=(self.image_size, self.image_size),
            resample= Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD,
        )

        # convert the list of numpy arrays to a single numpy array with shape [batch_size, channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        # convert the numpy array to a Pytorch tensor
        pixel_values = torch.tensor(pixel_values)

        # pretend a 'self.image_seg_length' number of images tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len= self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ] 

        # Returns the input_ids and attention_mask as pyTorch tensors 
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

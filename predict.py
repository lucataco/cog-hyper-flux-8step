# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from dataclasses import dataclass
from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from typing import List
from transformers import CLIPImageProcessor
from PIL import Image
from loras_cache import LorasCache
from diffusers.pipelines.flux import (
    FluxPipeline,
    FluxInpaintPipeline,
    FluxImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker
)

MODEL_CACHE = "Hyper-FLUX.1-dev"
MODEL_URL = "https://weights.replicate.delivery/default/ByteDance/Hyper-FLUX.1-dev-8steps/model.tar"
SAFETY_CACHE = "safety-cache"
FEATURE_EXTRACTOR = "/src/feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def make_multiple_of_16(x):
    return (x + 15) // 16 * 16

@dataclass
class LoadedLoRAs:
    main: str | None
    extra: str | None

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        print("Loading Flux txt2img Pipeline")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.txt2img_pipe = FluxPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16
        ).to("cuda")

        pipe_kwargs = {
            "transformer": self.txt2img_pipe.transformer,
            "scheduler": self.txt2img_pipe.scheduler,
            "vae": self.txt2img_pipe.vae,
            "text_encoder": self.txt2img_pipe.text_encoder,
            "text_encoder_2": self.txt2img_pipe.text_encoder_2,
            "tokenizer": self.txt2img_pipe.tokenizer,
            "tokenizer_2": self.txt2img_pipe.tokenizer_2,
        }

        # Load img2img pipelines
        print("Loading Flux dev img2img pipeline")
        self.img2img_pipe = FluxImg2ImgPipeline(**pipe_kwargs).to("cuda")

        # Load inpainting pipelines
        print("Loading Flux dev inpaint pipeline")
        self.inpaint_pipe = FluxInpaintPipeline(**pipe_kwargs).to("cuda")
        
        self.loras_cache = LorasCache()
        self.loaded_lora_urls = LoadedLoRAs(main=None, extra=None)

        print("setup took: ", time.time() - start)

    @torch.amp.autocast('cuda')
    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to("cuda")
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def aspect_ratio_to_width_height(self, aspect_ratio: str) -> tuple[int, int]:
        return ASPECT_RATIOS[aspect_ratio]

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image. The size will always be 1 megapixel, i.e. 1024x1024 if aspect ratio is 1:1. To use arbitrary width and height, set aspect ratio to 'custom'.",
            choices=list(ASPECT_RATIOS.keys()) + ["custom"],
            default="1:1",
        ),
        width: int = Input(
            description="Width of the generated image. Optional, only used when aspect_ratio=custom. Must be a multiple of 16 (if it's not, it will be rounded to nearest multiple of 16)",
            ge=256,
            le=1440,
            default=None,
        ),
        height: int = Input(
            description="Height of the generated image. Optional, only used when aspect_ratio=custom. Must be a multiple of 16 (if it's not, it will be rounded to nearest multiple of 16)",
            ge=256,
            le=1440,
            default=None,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1, le=30, default=8,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process",
            ge=0, le=10, default=3.5,
        ),
        seed: int = Input(description="Random seed. Set for reproducible generation", default=None),
        image: Path = Input(
            description="Input image for img2img or inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
            default=None,
        ),
        mask: Path = Input(
            description="Input mask for inpainting mode. Black areas will be preserved, white areas will be inpainted. Must be provided along with 'image' for inpainting mode.",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        lora: str = Input(
            description="Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
            default=None,
        ),
        lora_scale: float = Input(
            description="Determines how strongly the main LoRA should be applied.",
            default=1.0,
            le=2.0,
            ge=-1.0,
        ),
        extra_lora: str = Input(
            description="A second(extra) LoRA, ignored if lora is not provided. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
            default=None,
        ),
        extra_lora_scale: float = Input(
            description="Determines how strongly the extra LoRA should be applied.",
            default=1.0,
            le=2.0,
            ge=-1.0,
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if aspect_ratio == "custom":
            if width is None or height is None:
                raise ValueError(
                    "width and height must be defined if aspect ratio is 'custom'"
                )
            width = make_multiple_of_16(width)
            height = make_multiple_of_16(height)
        else:
            width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        max_sequence_length = 512

        is_img2img_mode = image is not None and mask is None
        is_inpaint_mode = image is not None and mask is not None

        flux_kwargs = {}
        print(f"Prompt: {prompt}")

        if is_img2img_mode or is_inpaint_mode:
            input_image = Image.open(image).convert("RGB")
            original_width, original_height = input_image.size

            # Calculate dimensions that are multiples of 16
            target_width = make_multiple_of_16(original_width)
            target_height = make_multiple_of_16(original_height)
            target_size = (target_width, target_height)

            print(
                f"[!] Resizing input image from {original_width}x{original_height} to {target_width}x{target_height}"
            )

            # Determine if we should use highest quality settings
            use_highest_quality = output_quality == 100 or output_format == "png"

            # Resize the input image
            resampling_method = Image.LANCZOS if use_highest_quality else Image.BICUBIC
            input_image = input_image.resize(target_size, resampling_method)
            flux_kwargs["image"] = input_image

            # Set width and height to match the resized input image
            flux_kwargs["width"], flux_kwargs["height"] = target_size

            if is_img2img_mode:
                print("[!] img2img mode")
                pipe = self.img2img_pipe
            else:  # is_inpaint_mode
                print("[!] inpaint mode")
                mask_image = Image.open(mask).convert("RGB")
                mask_image = mask_image.resize(target_size, Image.NEAREST)
                flux_kwargs["mask_image"] = mask_image
                pipe = self.inpaint_pipe

            flux_kwargs["strength"] = prompt_strength
        else:  # is_txt2img_mode
            print("[!] txt2img mode")
            pipe = self.txt2img_pipe
            flux_kwargs["width"] = width
            flux_kwargs["height"] = height

        if lora:
            start_time = time.time()
            if extra_lora:
                flux_kwargs["joint_attention_kwargs"] = {"scale": 1.0}
                print(f"Loading LoRA ({lora}) and extra LoRA ({extra_lora})")
                self.load_multiple_loras(lora, extra_lora, pipe)
                pipe.set_adapters(
                    ["main", "extra"], adapter_weights=[lora_scale, extra_lora_scale]
                )
            else:
                flux_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
                print(f"Loading LoRA ({lora})")
                self.load_single_lora(lora, pipe)
                pipe.set_adapters(["main"], adapter_weights=[lora_scale])
            print(f"Loaded LoRAs in {time.time() - start_time:.2f}s")
        else:
            pipe.unload_lora_weights()
            self.loaded_lora_urls = LoadedLoRAs(main=None, extra=None)

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
            "output_type": "pil"
        }

        output = pipe(**common_args, **flux_kwargs)

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker and has_nsfw_content[i]:
                print(f"NSFW content detected in image {i}")
                continue
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != 'png':
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception("NSFW content detected. Try running it again, or try a different prompt.")

        return output_paths


    def load_single_lora(self, lora_url: str, pipe: FluxPipeline | FluxImg2ImgPipeline | FluxInpaintPipeline):
        # If no change, skip
        if lora_url == self.loaded_lora_urls.main:
            print("Weights already loaded")
            return

        pipe.unload_lora_weights()
        lora_path = self.loras_cache.ensure(lora_url)
        pipe.load_lora_weights(lora_path, adapter_name="main", low_cpu_mem_usage=True)
        self.loaded_lora_urls = LoadedLoRAs(main=lora_url, extra=None)
        pipe.to("cuda")

    def load_multiple_loras(self, main_lora_url: str, extra_lora_url: str, pipe: FluxPipeline | FluxImg2ImgPipeline | FluxInpaintPipeline):

        # If no change, skip
        if (
            main_lora_url == self.loaded_lora_urls.main
            and extra_lora_url == self.loaded_lora_urls.extra
        ):
            print("Weights already loaded")
            return

        # We always need to load both?
        pipe.unload_lora_weights()

        main_lora_path = self.loras_cache.ensure(main_lora_url)
        pipe.load_lora_weights(main_lora_path, adapter_name="main", low_cpu_mem_usage=True)

        extra_lora_path = self.loras_cache.ensure(extra_lora_url)
        pipe.load_lora_weights(extra_lora_path, adapter_name="extra", low_cpu_mem_usage=True)

        self.loaded_lora_urls = LoadedLoRAs(
            main=main_lora_url, extra=extra_lora_url
        )
        pipe.to("cuda")
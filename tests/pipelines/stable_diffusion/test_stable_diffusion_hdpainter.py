# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import random
import traceback
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionHDPainterPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    require_python39_or_higher,
    require_torch_2,
    require_torch_gpu,
    run_test_in_subprocess,
    slow,
    torch_device,
)

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin

enable_full_determinism()
torch.use_deterministic_algorithms(True, warn_only=True)

class StableDiffusionHDPainterPipelineFastTests(
    PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionHDPainterPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"mask", "masked_image_latents"})

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            time_cond_proj_dim=time_cond_proj_dim,
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = DDIMScheduler()
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0, img_res=64, output_pil=True):
        # TODO: use tensor inputs instead of PIL, this is here just to leave the old expected_slices untouched
        if output_pil:
            # Get random floats in [0, 1] as image
            image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
            image = image.cpu().permute(0, 2, 3, 1)[0]
            mask_image = torch.ones_like(image)
            # Convert image and mask_image to [0, 255]
            image = 255 * image
            mask_image = 255 * mask_image
            # Convert to PIL image
            init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((img_res, img_res))
            mask_image = Image.fromarray(np.uint8(mask_image)).convert("RGB").resize((img_res, img_res))
        else:
            # Get random floats in [0, 1] as image with spatial size (img_res, img_res)
            image = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed)).to(device)
            # Convert image to [-1, 1]
            init_image = 2.0 * image - 1.0
            mask_image = torch.ones((1, 1, img_res, img_res), device=device)

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "rasg_scale_factor": 2,
            "painta_scale_factors": [2]
        }
        return inputs

    def test_stable_diffusion_hdpainter(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionHDPainterPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        with open("/home/hayk.manukyan/repos/diffusers/debug.txt", 'a') as f: f.write(f"SLICE test_stable_diffusion_hdpainter_fp16 {image_slice}")
        expected_slice = np.array([0.5066, 0.4322, 0.4942, 0.3106, 0.3096, 0.4898, 0.5117, 0.5197,  0.4896])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_rasg(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionHDPainterPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs, use_rasg = True, use_painta = False).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        with open("/home/hayk.manukyan/repos/diffusers/debug.txt", 'a') as f: f.write(f"SLICE test_stable_diffusion_hdpainter_fp16 {image_slice}")
        expected_slice = np.array( [0.5066, 0.4323, 0.4942, 0.3106, 0.3096, 0.4898, 0.5117, 0.5197, 0.4896])
        
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_painta(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionHDPainterPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs, use_rasg = False, use_painta = True).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        with open("/home/hayk.manukyan/repos/diffusers/debug.txt", 'a') as f: f.write(f"SLICE test_stable_diffusion_hdpainter_fp16 {image_slice}")
        expected_slice = np.array( [0.5066, 0.4322, 0.4942, 0.3106, 0.3096, 0.4898, 0.5117, 0.5197, 0.4896])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        
@slow
@require_torch_gpu
class StableDiffusionHDPainterPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_hdpainter(self):
        pipe = StableDiffusionHDPainterPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        with open("/home/hayk.manukyan/repos/diffusers/debug.txt", 'a') as f: f.write(f"SLICE test_stable_diffusion_hdpainter_fp16 {image_slice}")
        expected_slice = np.array([0.1677, 0.1971, 0.1661,  0.1862, 0.1606, 0.1803, 0.1643, 0.2175, 0.2130])
        
        assert np.abs(expected_slice - image_slice).max() < 6e-4

    def test_stable_diffusion_hdpainter_fp16(self):
        pipe = StableDiffusionHDPainterPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        with open("/home/hayk.manukyan/repos/diffusers/debug.txt", 'a') as f: f.write(f"SLICE test_stable_diffusion_hdpainter_fp16 {image_slice}")
        expected_slice = np.array([0.1540, 0.1279, 0.1474, 0.2062, 0.1918, 0.1647, 0.1611, 0.2387, 0.1713])

        assert np.abs(expected_slice - image_slice).max() < 1e-1
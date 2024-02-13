from PIL import Image
from cog import BasePredictor, Input, Path
from controlnet_aux import (
    HEDdetector,
    OpenposeDetector,
    MLSDdetector,
    CannyDetector,
    LineartDetector,
    MidasDetector
)
from compel import Compel
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image
import insightface
import onnxruntime
from insightface.app import FaceAnalysis
import cv2
import gfpgan
from diffusers.utils import load_image
import numpy as np
import tempfile
import time
from typing import List

class Predictor(BasePredictor):
    def setup(self):
        model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
        adapter_id = "latent-consistency/lcm-lora-sdv1-5"

        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")
        # load and fuse lcm lora
        self.pipe.load_lora_weights(adapter_id)
        self.pipe.fuse_lora()

        self.face_swapper = insightface.model_zoo.get_model('cache/inswapper_128.onnx', providers=onnxruntime.get_available_providers())
        self.face_analyser = FaceAnalysis(name='buffalo_l')
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        self.face_enhancer = gfpgan.GFPGANer(model_path='cache/GFPGANv1.4.pth', upscale=1)


    def get_face(self, img_data):
        analysed = self.face_analyser.get(img_data)
        try:
            largest = max(analysed, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return largest
        except:
            print("No face found") 
            return None

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image", default= None),
        prompt: str = Input(description="Prompt - using compel, use +++ to increase words weight:: doc: https://github.com/damian0815/compel/tree/main/doc || https://invoke-ai.github.io/InvokeAI/features/PROMPTS/#attention-weighting",),
        negative_prompt: str = Input(
            description="Negative prompt - using compel, use +++ to increase words weight//// negative-embeddings available ///// FastNegativeV2 , boring_e621_v4 , verybadimagenegative_v1 || to use them, write their keyword in negative prompt",
            default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ),
        num_inference_steps: int = Input(description="Steps to run denoising", default=20),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.0,
            ge=0.1,
            le=30.0,
        ),
        # seed: int = Input(description="Seed", default=None),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
        # num_outputs: int = Input(
        #     description="Number of images to generate",
        #     ge=1,
        #     le=10,
        #     default=1,
        # ),
        max_width: int = Input(
            description="Max width/Resolution of image",
            default=512,
        ),
        max_height: int = Input(
            description="Max height/Resolution of image",
            default=512,
        ),
        ip_adapter_scale: float = Input(
            description="Scale for IP adapter",
            default=0.8,
            ge=0,
            le=5,
        ),

    ) -> List[Path]:
        
        if disable_safety_check:
            self.pipe.safety_checker = None
        
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        image = load_image(image)
        image_ = self.pipe(
            prompt=prompt,
            negative_prompt= negative_prompt,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            ip_adapter_image=image,
            width=max_width,
            height=max_height
        ).images[0]
        frame = cv2.cvtColor(np.array(image_), cv2.COLOR_RGB2BGR)
        face = self.get_face(frame)
        source_face = self.get_face(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        result = self.face_swapper.get(frame, face, source_face, paste_back=True)
        _, _, result = self.face_enhancer.enhance(
            result,
            paste_back=True
        )

        out_path = Path(tempfile.mkdtemp()) / f"{str(int(time.time()))}.jpg"
        cv2.imwrite(str(out_path), result)
        return out_path
    

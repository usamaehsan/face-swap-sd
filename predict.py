from PIL import Image
from cog import BasePredictor, Input, Path
from controlnet_aux import ( OpenposeDetector,)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from compel import Compel
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image,ControlNetModel, StableDiffusionControlNetPipeline
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

MODEL_CACHE = "weights"

class Predictor(BasePredictor):
    def setup(self):
        model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
        adapter_id = "latent-consistency/lcm-lora-sdv1-5"

        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
        # load and fuse lcm lora
        self.pipe.load_lora_weights(adapter_id)
        self.pipe.fuse_lora()

        self.face_swapper = insightface.model_zoo.get_model(f'{MODEL_CACHE}/inswapper_128.onnx', providers=onnxruntime.get_available_providers())
        self.face_analyser = FaceAnalysis(name='buffalo_l')
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        self.face_enhancer = gfpgan.GFPGANer(model_path=f'{MODEL_CACHE}/GFPGANv1.4.pth', upscale=1)

        self.compel_proc = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)

        self.openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        self.pose_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
        )

        p= self.pipe
        self.controlnet_pipe = StableDiffusionControlNetPipeline(
            vae=p.vae,
            text_encoder=p.text_encoder,
            tokenizer=p.tokenizer,
            unet=p.unet,
            scheduler=p.scheduler,
            safety_checker=p.safety_checker,
            feature_extractor=p.feature_extractor,
            controlnet=[self.pose_controlnet],
        )
        self.controlnet_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
        # load and fuse lcm lora
        # self.pipe.load_lora_weights(adapter_id)
        # self.pipe.fuse_lora()

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
        pose_image: Path = Input(description="pose image", default= None),
        prompt: str = Input(description="Prompt - using compel, use +++ to increase words weight:: doc: https://github.com/damian0815/compel/tree/main/doc || https://invoke-ai.github.io/InvokeAI/features/PROMPTS/#attention-weighting",),
        negative_prompt: str = Input(
            description="Negative prompt - using compel, use +++ to increase words weight//// negative-embeddings available ///// FastNegativeV2 , boring_e621_v4 , verybadimagenegative_v1 || to use them, write their keyword in negative prompt",
            default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ),
        num_inference_steps: int = Input(description="Steps to run denoising", default=20),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=2.0,
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
        width: int = Input(
            description="Max width/Resolution of image",
            default=512,
        ),
        height: int = Input(
            description="Max height/Resolution of image",
            default=512,
        ),
        ip_adapter_scale: float = Input(
            description="Scale for IP adapter",
            default=0.8,
            ge=0,
            le=5,
        ),
        pose_scale: float = Input(
            description="Scale for open pose controlnet",
            default=0.8,
            ge=0,
            le=5,
        ),
        use_pose_image_resolution: bool = Input(
            description="image will be generated in pose' width and height", default=True
        ),
        use_gfpgan: bool = Input(
            description="gfpgan to enhance face", default=True
        ),
    ) -> List[Path]:
        t1 = time.time()
        if disable_safety_check:
            self.pipe.safety_checker = None
        
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        try:
            image = load_image(str(image))
        except Exception as e:
            print(e)
            image = Image.open(image)
        
        w,h = width, height
        print(f"time1 : {time.time() - t1:.2f} seconds")

        if pose_image:
            pose_t = time.time()
            try:
                pose_image_ = load_image(str(pose_image))
            except Exception as e:
                print(e)
                pose_image_ = Image.open(pose_image)

            # self.pipe.controlnet = MultiControlNetModel([self.pose_controlnet])
            control_scales = [pose_scale]
            if use_pose_image_resolution:
                w,h = pose_image_.size
            control_images = [self.openpose(pose_image_).resize((w,h))]
            print(f"pose time : {time.time() - pose_t:.2f} seconds")
        else:
            control_scales = []
            self.pipe.controlnet = MultiControlNetModel()
            control_images = []

        predict_t = time.time()
        if pose_image:
            image_ = self.controlnet_pipe(
                prompt_embeds=self.compel_proc(prompt),
                negative_prompt_embeds=self.compel_proc(negative_prompt),
                image= control_images,
                controlnet_conditioning_scale=control_scales,
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale,
                ip_adapter_image=image,
                width=w,
                height=h
            ).images[0]
        else:
            image_ = self.pipe(
                prompt_embeds=self.compel_proc(prompt),
                negative_prompt_embeds=self.compel_proc(negative_prompt),
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale,
                ip_adapter_image=image,
                width=w,
                height=h
            ).images[0]
        print(f"prediction time : {time.time() - predict_t:.2f} seconds")

        swap_time = time.time()
        frame = cv2.cvtColor(np.array(image_), cv2.COLOR_RGB2BGR)
        face = self.get_face(frame)
        source_face = self.get_face(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        result = self.face_swapper.get(frame, face, source_face, paste_back=True)
        print(f"swap time : {time.time() - swap_time:.2f} seconds")
        if use_gfpgan:
            gfp_t= time.time()
            _, _, result = self.face_enhancer.enhance(
                result,
                paste_back=True
            )
            print(f"gfpgan time : {time.time() - gfp_t:.2f} seconds")

        out_path = f"/tmp/output_0.png"
        cv2.imwrite(str(out_path), result)

        print(f"total time : {time.time() - t1:.2f} seconds")
        return Path(out_path)
    

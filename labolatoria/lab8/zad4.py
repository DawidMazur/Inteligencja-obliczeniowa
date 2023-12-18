from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionUpscalePipeline
import torch
from PIL import Image
from slugify import slugify

print("Torch version:", torch.__version__)

print("Is CUDA enabled?", torch.cuda.is_available())

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16")
pipe.to("cuda")

prompt = "sunset in the desert"

image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
print(image)

# # Save image to PNG
image.save(slugify(prompt) + ".png")

image.show()


# bigger resolution

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipeline = pipeline.to("cuda")

upscaled_image = pipeline(prompt=prompt, image=image).images[0]
upscaled_image.save("upsampled_" + slugify(prompt) + ".png")





# # show image
# from IPython.display import Image
#
# Image("image.png")

# # paczka ma 5GB


# za dużo miejsca na dysku ale działa
# from diffusers import AutoPipelineForText2Image
# import torch
#
# pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16",
#                                                  torch_dtype=torch.float16)
# pipe.enable_model_cpu_offload()
#
# prompt = "A cinematic shot of black hole in space"
#
# generator = torch.Generator(device="cpu").manual_seed(0)
# image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]


from diffusers import DiffusionPipeline, LCMScheduler
import torch

# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
# pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
#
# pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
# pipe.load_lora_weights("./pixel-art-xl.safetensors", adapter_name="pixel")
#
# pipe.set_adapters(["lora", "pixel"], adapter_weights=[1.0, 1.2])
# pipe.to(device="cuda", dtype=torch.float16)
#
# prompt = "A cinematic shot of black hole in space"
# negative_prompt = "3d render, realistic"
#
# num_images = 9
#
# for i in range(num_images):
#     img = pipe(
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         num_inference_steps=8,
#         guidance_scale=1.5,
#     ).images[0]
#
#     img.save(f"lcm_lora_{i}.png")

from transformers import LlamaForCausalLM

old_forward = LlamaForCausalLM.forward

def forward(self, input_ids, attention_mask, **kwargs):
    """Condition the Flamingo layers on the media locations before forward()"""
    if not self.initialized_flamingo:
        raise ValueError(
            "Flamingo layers are not initialized. Please call `init_flamingo` first."
        )

    media_locations = input_ids == self.media_token_id

    # if there are media already cached and we're generating and there are no media tokens in the input,
    # we'll assume that ALL input tokens should attend to the last previous media that is cached.
    # this is especially important for HF generate() compatibility, since generate() calls forward()
    # repeatedly one token at a time (with no media tokens).
    # without this check, the model would not attend to any images when generating (after the first token)
    use_cached_media_locations = (
        self._use_cached_vision_x
        and self.is_conditioned()
        and not media_locations.any()
    )

    for layer in self._get_decoder_layers():
        if not use_cached_media_locations:
            layer.condition_media_locations(media_locations)
        layer.condition_use_cached_media(use_cached_media_locations)

    # package arguments for the other parent's forward. since we don't know the order of the arguments,
    # make them all kwargs
    kwargs["input_ids"] = input_ids
    kwargs["attention_mask"] = attention_mask
    return old_forward(self, **kwargs)  # Call the other parent's forward method



LlamaForCausalLM.forward = forward

from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip

from open_flamingo.src.flamingo import Flamingo
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance

def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        load_in_4bit = True
    )

    # convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        # TODO: investigate also training the output embeddings when untied

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
}

from huggingface_hub import hf_hub_download
checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')

import torch

a = torch.load(checkpoint_path, map_location="cuda")

from huggingface_hub import hf_hub_download
import torch
import os
# from open_flamingo import create_model_and_transforms
from accelerate import Accelerator
from einops import repeat
from PIL import Image
import sys
sys.path.append('./')
from src.utils import FlamingoProcessor

def clean_generation(response):
    """
    for some reason, the open-flamingo based model slightly changes the input prompt (e.g. prepends <unk>, an adds some spaces)
    """
    return response.replace('<unk> ', '').strip()

accelerator = Accelerator() #when using cpu: cpu=True

device = accelerator.device

print('Loading model..')


model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="huggyllama/llama-7b",
    tokenizer_path= "huggyllama/llama-7b",
    cross_attn_every_n_layers=4,
)

model.perceiver.cuda()
torch.cuda.empty_cache()

model.vision_encoder.cuda()
torch.cuda.empty_cache()

model.lang_encoder.gated_cross_attn_layers.to(torch.float16).cuda()
torch.cuda.empty_cache()
torch.cuda.empty_cache()

import gc  # Import the garbage collection module

# Collect and free up unused memory
gc.collect()

# load med-flamingo checkpoint:
model.load_state_dict(a, strict=False)
processor = FlamingoProcessor(tokenizer, image_processor)


# Select only the first 2 image paths
image_paths = [
    "img/pneumonia.jpg",
    "img/brokenarm.jpg",
    "img/braintumor.jpg",
    "img/brokentoe.jpg",
    "img/brokenwrist.jpg",
]

# Prepare the model using the Accelerator
model = accelerator.prepare(model)
is_main_process = accelerator.is_main_process
model.eval()

"""
Step 1: Load images
"""
# Load demo images using PIL and store in a list
demo_images = [Image.open(path) for path in image_paths]

"""
Step 2: Define multimodal few-shot prompt
"""

# Define a few-shot prompt containing text and <image> placeholders
prompt = "You are a helpful medical assistant. You are being provided with images," \
    "a question about the image and an answer. Follow the examples and answer the last question." \
    "<image>Question: What is the most likely diagnosis? Answer: pneumonia.<|endofchunk|>" \
    "<image>Question: What is the most likely diagnosis? Answer: broken arm.<|endofchunk|>" \
    "<image>Question: What is the most likely diagnosis? Answer: brain tumor.<|endofchunk|>" \
    "<image>Question: What is the most likely diagnosis? Answer: broken toe.<|endofchunk|>" \
    "<image>Question: What is the most likely diagnosis? Answer:"

"""
Step 3: Preprocess data
"""
print('Preprocess data')

# Preprocess demo images using the FlamingoProcessor
pixels = processor.preprocess_images(demo_images)

pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)

# Encode the text prompt using the FlamingoProcessor
tokenized_data = processor.encode_text(prompt)

"""
Step 4: Generate response
"""

# Generate a response using the multimodal few-shot prompt
print('Generate from multimodal few-shot prompt')

# Use mixed-precision training context for improved performance
with torch.autocast('cuda', torch.float16):

    # Generate text using the model
    generated_text = model.generate(
        vision_x=pixels.to(device),  # Convert images to the device
        lang_x=tokenized_data["input_ids"].to(device),  # Convert text input to the device
        attention_mask=tokenized_data["attention_mask"].to(device),  # Convert attention mask to the device
        max_new_tokens=10,  # Limit the maximum number of new tokens in the generated response
    )

# Decode the generated text using the processor's tokenizer
response = processor.tokenizer.decode(generated_text[0])

# Clean up the generated response
response = clean_generation(response)

# Print the cleaned response
print(f'{response=}')
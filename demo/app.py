"""
ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts
=======================================================================

A Gradio web interface for the ShapeWords paper, allowing users to generate
images guided by 3D shape information.

Author: Melinos Averkiou
Date: 11 March 2025
Version: 1.0

Paper: "ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts"
arXiv: https://arxiv.org/abs/2412.02912
Project Page: https://lodurality.github.io/shapewords/

Citation:
@misc{petrov2024shapewords,
    title={ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts},
    author={Dmitry Petrov and Pradyumn Goyal and Divyansh Shivashok and Yuanming Tao and Melinos Averkiou and Evangelos Kalogerakis},
    year={2024},
    eprint={2412.02912},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.02912},
}

License: MIT License

Usage:
    python app.py [--share]

This demo allows users to:
1. Select a 3D object category
2. Choose a specific 3D shape using a slider
3. Enter a text prompt
4. Generate images guided by the selected 3D shape

The code is structured as a class and is compatible with Hugging Face ZeroGPU deployment.
"""

import os
import sys
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageFont, ImageDraw
from diffusers.utils import load_image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import open_clip
import gdown
import argparse
import random
import spaces

class ShapeWordsDemo:
    # Constants
    NAME2CAT = {
        "chair": "03001627", "table": "04379243", "jar": "03593526", "skateboard": "04225987", 
        "car": "02958343", "bottle": "02876657", "tower": "04460130", "bookshelf": "02871439", 
        "camera": "02942699", "airplane": "02691156", "laptop": "03642806", "basket": "02801938", 
        "sofa": "04256520", "knife": "03624134", "can": "02946921", "rifle": "04090263", 
        "train": "04468005", "pillow": "03938244", "lamp": "03636649", "trash bin": "02747177", 
        "mailbox": "03710193", "watercraft": "04530566", "motorbike": "03790512", 
        "dishwasher": "03207941", "bench": "02828884", "pistol": "03948459", "rocket": "04099429", 
        "loudspeaker": "03691459", "file cabinet": "03337140", "bag": "02773838", 
        "cabinet": "02933112", "bed": "02818832", "birdhouse": "02843684", "display": "03211117", 
        "piano": "03928116", "earphone": "03261776", "telephone": "04401088", "stove": "04330267", 
        "microphone": "03759954", "bus": "02924116", "mug": "03797390", "remote": "04074963", 
        "bathtub": "02808440", "bowl": "02880940", "keyboard": "03085013", "guitar": "03467517", 
        "washer": "04554684", "bicycle": "02834778", "faucet": "03325088", "printer": "04004475", 
        "cap": "02954340", "phone": "02992529", "clock": "03046257", "helmet": "03513137", 
        "microwave": "03761084", "plant": "03991062"
    }

    def __init__(self):
        # Initialize class attributes
        self.pipeline = None
        self.shape2clip_model = None
        self.text_encoder = None
        self.tokenizer = None
        self.category_embeddings = {}
        self.category_counts = {}
        self.available_categories = []
        self.shape_thumbnail_cache = {}  # Cache for shape thumbnails
        self.CAT2NAME = {v: k for k, v in self.NAME2CAT.items()}
        
        # Initialize all models and data
        self.initialize_models()

    def draw_text(self, img, text, color=(10, 10, 10), size=80, location=(200, 30)):
        img = img.copy()
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("Arial", size=size)
        except IOError:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox(location, text, font=font)
        draw.rectangle(bbox, fill="white")
        draw.text(location, text, color, font=font)
        
        return img

    def get_ulip_image(self, guidance_shape_id, angle='036'):
        shape_id_ulip = guidance_shape_id.replace('_', '-')
        ulip_template = 'https://storage.googleapis.com/sfr-ulip-code-release-research/shapenet-55/only_rgb_depth_images/{}_r_{}_depth0001.png'
        ulip_path = ulip_template.format(shape_id_ulip, angle)
        
        try:
            ulip_image = load_image(ulip_path).resize((512, 512))
            return ulip_image
        except Exception as e:
            print(f"Error loading image: {e}")
            return Image.new('RGB', (512, 512), color='gray')

    def get_ulip_thumbnail(self, guidance_shape_id, angle='036', size=(150, 150)):
        """Get a thumbnail version of the ULIP image for use in the gallery"""
        image = self.get_ulip_image(guidance_shape_id, angle)
        return image.resize(size)

    def initialize_models(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Download Shape2CLIP code if it doesn't exist
        if not os.path.exists("shapewords_paper_code"):
            os.system("git clone https://github.com/lodurality/shapewords_paper_code.git")
        
        # Import Shape2CLIP model
        sys.path.append("./shapewords_paper_code")
        from shapewords_paper_code.geometry_guidance_models import Shape2CLIP
        
        # Initialize the pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", 
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        )
        
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config, 
            algorithm_type="sde-dpmsolver++"
        )
        
        # Load CLIP model
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14', 
            pretrained='laion2b_s32b_b79k'
        )
        
        # Move models to device if not using ZeroGPU
        if device.type == "cuda":
            self.pipeline = self.pipeline.to(device)
            self.pipeline.enable_model_cpu_offload()
        
        clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        
        # Look for Shape2CLIP checkpoint in multiple locations
        checkpoint_paths = [
            "projection_model-0920192.pth",
            "embeddings/projection_model-0920192.pth"
        ]
        
        checkpoint_found = False
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"Found Shape2CLIP checkpoint at: {checkpoint_path}")
                checkpoint_found = True
                break
        
        # Download Shape2CLIP checkpoint if not found
        if not checkpoint_found:
            checkpoint_path = "projection_model-0920192.pth"
            print("Downloading Shape2CLIP model checkpoint...")
            gdown.download("1nvEXnwMpNkRts6rxVqMZt8i9FZ40KjP7", checkpoint_path, quiet=False)
            print("Download complete")
        
        # Initialize Shape2CLIP model
        self.shape2clip_model = Shape2CLIP(depth=6, drop_path_rate=0.1, pb_dim=384)
        self.shape2clip_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if device.type == "cuda":
            self.shape2clip_model = self.shape2clip_model.to(device)
        self.shape2clip_model.eval()
        
        # Scan for available embeddings
        self.scan_available_embeddings()

    def scan_available_embeddings(self):
        self.available_categories = []
        self.category_counts = {}
        
        for category, cat_id in self.NAME2CAT.items():
            possible_filenames = [
                f"pointbert_shapenet_{cat_id}.npz",
                f"{cat_id}_pb_embs.npz",
                f"embeddings/pointbert_shapenet_{cat_id}.npz",
                f"embeddings/{cat_id}_pb_embs.npz"
            ]
            
            found_file = None
            for filename in possible_filenames:
                if os.path.exists(filename):
                    found_file = filename
                    break
                    
            if found_file:
                try:
                    pb_data = np.load(found_file)
                    if 'ids' in pb_data:
                        count = len(pb_data['ids'])
                    else:
                        # Try to infer the correct keys
                        keys = list(pb_data.keys())
                        if len(keys) >= 1:
                            count = len(pb_data[keys[0]])
                        else:
                            count = 0
                    
                    if count > 0:
                        self.available_categories.append(category)
                        self.category_counts[category] = count
                        print(f"Found {count} embeddings for category '{category}'")
                except Exception as e:
                    print(f"Error loading embeddings for {category}: {e}")
        
        if not self.available_categories:
            self.available_categories = ["chair"]  # Fallback
            self.category_counts["chair"] = 50     # Default value
        
        # Sort categories alphabetically
        self.available_categories.sort()
        
        print(f"Found {len(self.available_categories)} categories with embeddings")
        print(f"Available categories: {', '.join(self.available_categories)}")

    def load_category_embeddings(self, category):
        if category in self.category_embeddings:
            return self.category_embeddings[category]
        
        if category not in self.NAME2CAT:
            return None, []
        
        cat_id = self.NAME2CAT[category]
        
        # Check for different possible embedding filenames and locations
        possible_filenames = [
            f"pointbert_shapenet_{cat_id}.npz",           
            f"{cat_id}_pb_embs.npz",                      
            f"embeddings/pointbert_shapenet_{cat_id}.npz", 
            f"embeddings/{cat_id}_pb_embs.npz",           
        ]
        
        # Find the first existing file
        pb_emb_filename = None
        for filename in possible_filenames:
            if os.path.exists(filename):
                pb_emb_filename = filename
                print(f"Found embeddings file: {pb_emb_filename}")
                break
        
        if pb_emb_filename is None:
            print(f"No embeddings found for {category}")
            return None, []
        
        # Load embeddings
        try:
            print(f"Loading embeddings from {pb_emb_filename}...")
            pb_data = np.load(pb_emb_filename)
            
            # Check for different key names in the NPZ file
            if 'ids' in pb_data and 'embs' in pb_data:
                pb_dict = dict(zip(pb_data['ids'], pb_data['embs']))
            else:
                # Try to infer the correct keys
                keys = list(pb_data.keys())
                if len(keys) >= 2:
                    # Assume first key is for IDs and second is for embeddings
                    pb_dict = dict(zip(pb_data[keys[0]], pb_data[keys[1]]))
                else:
                    print("Unexpected embedding file format")
                    return None, []
            
            all_ids = sorted(list(pb_dict.keys()))
            print(f"Loaded {len(all_ids)} shape embeddings for {category}")
            
            # Cache the results
            self.category_embeddings[category] = (pb_dict, all_ids)
            return pb_dict, all_ids
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            print(f"Exception details: {str(e)}")
            return None, []

    def get_shape_preview(self, category, shape_idx, size=(300, 300)):
        """Get a preview image for a specific shape"""
        if shape_idx is None or shape_idx < 0:
            return None
            
        pb_dict, all_ids = self.load_category_embeddings(category)
        if pb_dict is None or not all_ids or shape_idx >= len(all_ids):
            return None
        
        shape_id = all_ids[shape_idx]
        
        try:
            # Get the shape image at the requested size
            preview_image = self.get_ulip_image(shape_id)
            preview_image = preview_image.resize(size)
            preview_with_text = self.draw_text(preview_image, f"Shape #{shape_idx}", size=30, location=(10, 10))
            return preview_with_text
        except Exception as e:
            print(f"Error loading preview for {shape_id}: {e}")
            # Create an empty error image
            empty_img = Image.new('RGB', size, color='gray')
            error_text = f"Error loading Shape #{shape_idx}"
            return self.draw_text(empty_img, error_text, size=30, location=(10, 10))

    def on_slider_change(self, shape_idx, category):
        """Update the preview when the slider changes"""
        max_idx = self.category_counts.get(category, 0) - 1
        
        # Get preview image
        preview_image = self.get_shape_preview(category, shape_idx)
        
        # Update counter text
        counter_text = f"Shape {shape_idx} of {max_idx}"
        
        return preview_image, counter_text, shape_idx

    def prev_shape(self, current_idx, category):
        """Go to previous shape"""
        max_idx = self.category_counts.get(category, 0) - 1
        new_idx = max(0, current_idx - 1)
        
        # Get preview image
        preview_image = self.get_shape_preview(category, new_idx)
        
        # Update counter text
        counter_text = f"Shape {new_idx} of {max_idx}"
        
        return new_idx, preview_image, counter_text

    def next_shape(self, current_idx, category):
        """Go to next shape"""
        max_idx = self.category_counts.get(category, 0) - 1
        new_idx = min(max_idx, current_idx + 1)
        
        # Get preview image
        preview_image = self.get_shape_preview(category, new_idx)
        
        # Update counter text
        counter_text = f"Shape {new_idx} of {max_idx}"
        
        return new_idx, preview_image, counter_text

    def jump_to_start(self, category):
        """Jump to the first shape"""
        max_idx = self.category_counts.get(category, 0) - 1
        new_idx = 0
        
        # Get preview image
        preview_image = self.get_shape_preview(category, new_idx)
        
        # Update counter text
        counter_text = f"Shape {new_idx} of {max_idx}"
        
        return new_idx, preview_image, counter_text

    def jump_to_end(self, category):
        """Jump to the last shape"""
        max_idx = self.category_counts.get(category, 0) - 1
        new_idx = max_idx
        
        # Get preview image
        preview_image = self.get_shape_preview(category, new_idx)
        
        # Update counter text
        counter_text = f"Shape {new_idx} of {max_idx}"
        
        return new_idx, preview_image, counter_text

    def random_shape(self, category):
        """Select a random shape from the category"""
        max_idx = self.category_counts.get(category, 0) - 1
        if max_idx <= 0:
            return 0, self.get_shape_preview(category, 0), f"Shape 0 of 0"
            
        # Generate random index
        random_idx = random.randint(0, max_idx)
        
        # Get preview image
        preview_image = self.get_shape_preview(category, random_idx)
        
        # Update counter text
        counter_text = f"Shape {random_idx} of {max_idx}"
        
        return random_idx, preview_image, counter_text

    def on_category_change(self, category):
        """Update the slider and preview when the category changes"""
        # Reset to the first shape
        current_idx = 0
        max_idx = self.category_counts.get(category, 0) - 1
        
        # Get preview image
        preview_image = self.get_shape_preview(category, current_idx)
        
        # Update counter text
        counter_text = f"Shape {current_idx} of {max_idx}"
        
        # Need to update the slider range
        new_slider = gr.Slider(
            minimum=0,
            maximum=max_idx,
            step=1,
            value=current_idx,
            label="Shape Index"
        )
        
        return new_slider, current_idx, preview_image, counter_text

    def get_guidance(self, test_prompt, category_name, guidance_emb):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prompt_tokens = torch.LongTensor(self.tokenizer.encode(test_prompt, padding='max_length')).to(device)
        
        with torch.no_grad():
            out = self.text_encoder(prompt_tokens.unsqueeze(0), output_attentions=True)
            prompt_emb = out.last_hidden_state.detach().clone()

        if len(guidance_emb.shape) == 1:
            guidance_emb = torch.FloatTensor(guidance_emb).unsqueeze(0).unsqueeze(0)
        else:
            guidance_emb = torch.FloatTensor(guidance_emb).unsqueeze(0)
        guidance_emb = guidance_emb.to(device)

        eos_inds = torch.where(prompt_tokens.unsqueeze(0) == 49407)[1]
        obj_word = category_name
        obj_word_token = self.tokenizer.encode(obj_word)[-2]
        chair_inds = torch.where(prompt_tokens.unsqueeze(0) == obj_word_token)[1]

        eos_strength = 0.8
        obj_strength = 1.0

        self.shape2clip_model.eval()
        with torch.no_grad():
            guided_prompt_emb_cond = self.shape2clip_model(prompt_emb.float(), guidance_emb[:,:,:].float()).half()
            guided_prompt_emb = guided_prompt_emb_cond.clone()
            
        guided_prompt_emb[:,:1] = 0
        guided_prompt_emb[:,:chair_inds] = 0
        guided_prompt_emb[:,chair_inds] *= obj_strength
        guided_prompt_emb[:,eos_inds+1:] = 0
        guided_prompt_emb[:,eos_inds] *= eos_strength
        guided_prompt_emb[:,chair_inds+1:eos_inds:] = 0
        fin_guidance = guided_prompt_emb

        return fin_guidance, prompt_emb

    # For ZeroGPU compatibility, uncomment this decorator when using ZeroGPU
    @spaces.GPU(duration=120)
    def generate_images(self, prompt, category, selected_shape_idx, guidance_strength, seed):
        # Clear status text immediately
        status = ""
        
        # Check if the category is in the prompt
        if category not in prompt:
            # Add the category to the prompt
            prompt = f"{prompt} {category}"
            status = f"<div style='padding: 10px; background-color: #f0f7ff; border-left: 5px solid #3498db; margin-bottom: 10px;'>Note: Added '{category}' to your prompt since it was missing.</div>"
        
        # Verify that the prompt doesn't contain other conflicting categories
        for other_category in self.available_categories:
            if other_category != category:
                # Check with word boundaries to avoid partial matches
                # e.g., "dishwasher" shouldn't match "washer"
                if f" {other_category} " in f" {prompt} " or prompt == other_category:
                    return [], f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Your prompt contains '{other_category}' but you selected '{category}'. Please use matching category in prompt and selection.</div>"
        
        # Load category embeddings if not already loaded
        pb_dict, all_ids = self.load_category_embeddings(category)
        if pb_dict is None or not all_ids:
            return [], f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Failed to load embeddings for {category}</div>"
        
        # Ensure shape index is valid
        if selected_shape_idx is None or selected_shape_idx < 0:
            selected_shape_idx = 0
        
        max_idx = len(all_ids) - 1
        selected_shape_idx = max(0, min(selected_shape_idx, max_idx))
        guidance_shape_id = all_ids[selected_shape_idx]
        
        # Set device and generator
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(seed)
        
        results = []
        
        # Add status message for generation
        updating_status = f"<div style='padding: 10px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin-bottom: 10px;'>Generating images using Shape #{selected_shape_idx}...</div>"
        
        try:
            # For ZeroGPU, move models to GPU if not already there
            if hasattr(spaces, 'GPU'):
             self.pipeline = self.pipeline.to(device)
             self.shape2clip_model = self.shape2clip_model.to(device)
            
            # Generate base image (without guidance)
            with torch.no_grad():
                base_images = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=50,
                    num_images_per_prompt=1,
                    generator=generator,
                    guidance_scale=7.5
                ).images
            
            base_image = base_images[0]
            base_image = self.draw_text(base_image, "Unguided result")
            results.append(base_image)
        except Exception as e:
            print(f"Error generating base image: {e}")
            status = f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Error generating base image: {str(e)}</div>"
            return results, status
        
        try:
            # Get shape guidance image
            ulip_image = self.get_ulip_image(guidance_shape_id)
            ulip_image = self.draw_text(ulip_image, "Guidance shape")
            results.append(ulip_image)
        except Exception as e:
            print(f"Error getting guidance shape: {e}")
            status = f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Error getting guidance shape: {str(e)}</div>"
            return results, status
        
        try:
            # Get shape guidance embedding
            pb_emb = pb_dict[guidance_shape_id]
            out_guidance, prompt_emb = self.get_guidance(prompt, category, pb_emb)
        except Exception as e:
            print(f"Error generating guidance: {e}")
            status = f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Error generating guidance: {str(e)}</div>"
            return results, status
        
        try:
            # Generate guided image
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                guided_images = self.pipeline(
                    prompt_embeds=prompt_emb + guidance_strength * out_guidance,
                    num_inference_steps=50,
                    num_images_per_prompt=1,
                    generator=generator,
                    guidance_scale=7.5
                ).images
            
            guided_image = guided_images[0]
            guided_image = self.draw_text(guided_image, f"Guided result (λ={guidance_strength:.1f})")
            results.append(guided_image)
            
            # Success status
            status = f"<div style='padding: 10px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin-bottom: 10px;'>✓ Successfully generated images using Shape #{selected_shape_idx} from category '{category}'.</div>"
            
            # For ZeroGPU, optionally move models back to CPU to free resources
            if hasattr(spaces, 'GPU'):
                self.pipeline = self.pipeline.to('cpu')
                self.shape2clip_model = self.shape2clip_model.to('cpu')
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error generating guided image: {e}")
            status = f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Error generating guided image: {str(e)}</div>"
        
        return results, status

    def update_prompt_for_category(self, old_prompt, new_category):
        # Remove all existing categories from the prompt
        cleaned_prompt = old_prompt
        for cat in self.available_categories:
            # Skip the current category
            if cat == new_category:
                continue
                
            # Replace the category with a space, being careful about word boundaries
            cleaned_prompt = cleaned_prompt.replace(f" {cat} ", " ")
            cleaned_prompt = cleaned_prompt.replace(f" {cat}", "")
            cleaned_prompt = cleaned_prompt.replace(f"{cat} ", "")
            # Only do exact match for the whole prompt
            if cleaned_prompt == cat:
                cleaned_prompt = ""
        
        # Add the new category if it's not already in the cleaned prompt
        cleaned_prompt = cleaned_prompt.strip()
        if new_category not in cleaned_prompt:
            if cleaned_prompt:
                return f"{cleaned_prompt} {new_category}"
            else:
                return new_category
        else:
            return cleaned_prompt

    def on_demo_load(self):
        """Function to ensure initial image is loaded when demo starts"""
        default_category = "chair" if "chair" in self.available_categories else self.available_categories[0]
        initial_img = self.get_shape_preview(default_category, 0)
        return initial_img

    def create_ui(self):
        # Ensure chair is in available categories, otherwise use the first available
        default_category = "chair" if "chair" in self.available_categories else self.available_categories[0]
        
        with gr.Blocks(title="ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts") as demo:
            gr.Markdown("""
            # ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts
            
            ShapeWords incorporates target 3D shape information with text prompts to guide image synthesis.
            
            - **Website**: [ShapeWords Project Page](https://lodurality.github.io/shapewords/)
            - **Paper**: [ArXiv](https://arxiv.org/abs/2412.02912)
            - **Publication**: Accepted to CVPR 2025
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="Prompt", 
                        placeholder="an aquarelle drawing of a chair",
                        value=f"an aquarelle drawing of a {default_category}"
                    )
                    
                    category = gr.Dropdown(
                        label="Object Category", 
                        choices=self.available_categories,
                        value=default_category
                    )
                    
                    # Hidden field to store selected shape index
                    selected_shape_idx = gr.Number(
                        value=0,
                        visible=False
                    )
                    
                    # Create a slider for shape selection with preview
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Slider for shape selection
                            shape_slider = gr.Slider(
                                minimum=0,
                                maximum=self.category_counts.get(default_category, 0) - 1,
                                step=1,
                                value=0,
                                label="Shape Index",
                                interactive=True
                            )
                            
                            # Display shape index counter
                            shape_counter = gr.Markdown(f"Shape 0 of {self.category_counts.get(default_category, 0) - 1}")
                            
                            # Quick navigation buttons
                            with gr.Row():
                                jump_start_btn = gr.Button("⏮️ First", size="sm")
                                random_btn = gr.Button("🎲 Random", size="sm", variant="secondary")
                                jump_end_btn = gr.Button("Last ⏭️", size="sm")
                            
                            with gr.Row():
                                prev_shape_btn = gr.Button("◀️ Previous", size="sm")
                                next_shape_btn = gr.Button("Next ▶️", size="sm")
                        
                        with gr.Column(scale=1):
                            # Preview image for the current shape
                            current_shape_image = gr.Image(
                                label="Selected Shape",
                                height=300,
                                width=300
                            )
                    
                    guidance_strength = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.1, value=0.9,
                        label="Guidance Strength (λ)"
                    )
                    
                    seed = gr.Slider(
                        minimum=0, maximum=10000, step=1, value=42,
                        label="Random Seed"
                    )
                    
                    run_button = gr.Button("Generate Images", variant="primary")
                    
                    info = gr.Markdown("""
                    **Note**: Higher guidance strength (λ) means stronger adherence to the 3D shape.
                    Start with λ=0.9 for a good balance between shape and prompt adherence.
                    """)
                    
                    status_text = gr.HTML("")
                    
                with gr.Column(scale=2):
                    gallery = gr.Gallery(
                        label="Results",
                        show_label=True,
                        elem_id="results_gallery",
                        columns=3,
                        height="auto"
                    )
            
            # Make sure the initial image is loaded when the demo starts
            demo.load(
                fn=self.on_demo_load,
                inputs=None,
                outputs=[current_shape_image]
            )
            
            # Connect slider to update preview
            shape_slider.change(
                fn=self.on_slider_change,
                inputs=[shape_slider, category],
                outputs=[current_shape_image, shape_counter, selected_shape_idx]
            )
            
            # Previous shape button
            prev_shape_btn.click(
                fn=self.prev_shape,
                inputs=[selected_shape_idx, category],
                outputs=[shape_slider, current_shape_image, shape_counter]
            )
            
            # Next shape button
            next_shape_btn.click(
                fn=self.next_shape,
                inputs=[selected_shape_idx, category],
                outputs=[shape_slider, current_shape_image, shape_counter]
            )
            
            # Jump to start button
            jump_start_btn.click(
                fn=self.jump_to_start,
                inputs=[category],
                outputs=[shape_slider, current_shape_image, shape_counter]
            )
            
            # Jump to end button
            jump_end_btn.click(
                fn=self.jump_to_end,
                inputs=[category],
                outputs=[shape_slider, current_shape_image, shape_counter]
            )
            
            # Random shape button
            random_btn.click(
                fn=self.random_shape,
                inputs=[category],
                outputs=[shape_slider, current_shape_image, shape_counter]
            )
            
            # Update the UI when category changes
            category.change(
                fn=self.on_category_change,
                inputs=[category],
                outputs=[shape_slider, selected_shape_idx, current_shape_image, shape_counter]
            )
            
            # Automatically update prompt when category changes
            category.change(
                fn=self.update_prompt_for_category,
                inputs=[prompt, category],
                outputs=[prompt]
            )
            
            # Clear status text before generating new images
            run_button.click(
                fn=lambda: None,  # Empty function to clear the status
                inputs=None,
                outputs=[status_text]
            )
            
            # Generate images when button is clicked
            run_button.click(
                fn=self.generate_images,
                inputs=[prompt, category, selected_shape_idx, guidance_strength, seed],
                outputs=[gallery, status_text]
            )
            
            gr.Markdown("""
            ## Credits
            
            This demo is based on the ShapeWords paper by Petrov et al. (2024) accepted to CVPR 2025.
            
            If you use this in your work, please cite:
            ```
            @misc{petrov2024shapewords,
                  title={ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts}, 
                  author={Dmitry Petrov and Pradyumn Goyal and Divyansh Shivashok and Yuanming Tao and Melinos Averkiou and Evangelos Kalogerakis},
                  year={2024},
                  eprint={2412.02912},
                  archivePrefix={arXiv},
                  primaryClass={cs.CV},
                  url={https://arxiv.org/abs/2412.02912}, 
            }
            ```
            """)
        
        return demo


# Main function and entry point
def main():
    parser = argparse.ArgumentParser(description="ShapeWords Gradio Demo")
    parser.add_argument('--share', action='store_true', help='Create a public link')
    args = parser.parse_args()
    
    # Create the demo app and UI
    app = ShapeWordsDemo()
    demo = app.create_ui()
    demo.launch(share=args.share)

if __name__ == "__main__":
    main()

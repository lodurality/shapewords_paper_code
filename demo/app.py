"""
ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts
=======================================================================

A Gradio web interface for the ShapeWords paper, allowing users to generate
images guided by 3D shape information.

Author: Melinos Averkiou
Date: 24 March 2025
Version: 1.5

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
1. Select a 3D object category from ShapeNetCore
2. Choose a specific 3D shape using a slider or the navigation buttons (including a random shape button)
3. Enter a text prompt or pick a random one
4. Generate images guided by the selected 3D shape and the text prompt

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
import gdown
import argparse
import random
import spaces # for Hugging Face ZeroGPU deployment
import re
import plotly.graph_objects as go
from numpy.lib.user_array import container

# Only for Hugging Face hosting - Add the Hugging Face cache to persistent storage to avoid downloading safetensors every time the demo sleeps and wakes up
os.environ['HF_HOME'] = '/data/.huggingface'

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
        self.category_point_clouds = {}
        self.from_navigation = False
        # Initialize all models and data
        self.initialize_models()



    def initialize_models(self):
        # device = DEVICE
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device} in initialize_models")

        # Download Shape2CLIP code if it doesn't exist
        if not os.path.exists("shapewords_paper_code"):
            print("Loading models file")
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

        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer

        # Look for Shape2CLIP checkpoint in multiple locations
        checkpoint_paths = [
            "./projection_model-0920192.pth",
            "/data/projection_model-0920192.pth" # if using Hugging Face persistent storage look in a /data/ directory
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
            gdown.download("https://drive.google.com/uc?id=1nvEXnwMpNkRts6rxVqMZt8i9FZ40KjP7", checkpoint_path, quiet=False) # download in same directory as app.py
            print("Download complete")

        # Initialize Shape2CLIP model
        self.shape2clip_model = Shape2CLIP(depth=6, drop_path_rate=0.1, pb_dim=384)
        self.shape2clip_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.shape2clip_model.eval()

        # Scan for available embeddings
        self.scan_available_embeddings()

    def scan_available_embeddings(self):
        self.available_categories = []
        self.category_counts = {}

        # Try to find PointBert embeddings for all 55 ShapeNetCore shape categories
        for category, cat_id in self.NAME2CAT.items():
            possible_filenames = [
                f"{cat_id}_pb_embs.npz",
                f"embeddings/{cat_id}_pb_embs.npz", 
                f"/data/shapenet_pointbert_tokens/{cat_id}_pb_embs.npz" # if using Hugging Face persistent storage look in a /data/shapenet_pointbert_tokens directory
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

        # Sort categories alphabetically
        self.available_categories.sort()

        print(f"Found {len(self.available_categories)} categories with embeddings")
        print(f"Available categories: {', '.join(self.available_categories)}")
        
        # No embeddings found for any category - DEMO CANNOT RUN - but still load the interface with a default placeholder category, an error will be displayed when trying to generate images
        if not self.available_categories:
            self.available_categories = ["chair"]  # Fallback
            self.category_counts["chair"] = 50     # Default value

    def load_category_embeddings(self, category):
        if category in self.category_embeddings:
            return self.category_embeddings[category]

        if category not in self.NAME2CAT:
            return None, []

        cat_id = self.NAME2CAT[category]

        # Check for different possible embedding filenames and locations
        possible_filenames = [
            f"{cat_id}_pb_embs.npz",
            f"embeddings/{cat_id}_pb_embs.npz",
            f"/data/shapenet_pointbert_tokens/{cat_id}_pb_embs.npz" # if using Hugging Face persistent storage look in a /data/shapenet_pointbert_tokens directory
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

    def load_category_point_clouds(self, category):
        """Load all point clouds for a category from a single NPZ file"""
        if category not in self.NAME2CAT:
            return None

        cat_id = self.NAME2CAT[category]

        # Cache to avoid reloading
        if category in self.category_point_clouds:
            return self.category_point_clouds[category]

        # Check for different possible point cloud filenames
        possible_filenames = [
            f"{cat_id}.npz",
            f"point_clouds/{cat_id}_clouds.npz",
            f"/point_clouds/{cat_id}_clouds.npz",
            f"/data/point_clouds/{cat_id}_clouds.npz"  # For Hugging Face persistent storage
        ]

        # Find the first existing file
        pc_filename = None
        for filename in possible_filenames:
            if os.path.exists(filename):
                pc_filename = filename
                print(f"Found point cloud file: {pc_filename}")
                break

        if pc_filename is None:
            print(f"No point cloud file found for category {category}")
            return None

        # Load point clouds
        try:
            print(f"Loading point clouds from {pc_filename}...")

            pc_data_map = np.load(pc_filename, allow_pickle=False)
            pc_data = {'ids':pc_data_map['ids'], 'clouds': pc_data_map['clouds']}

            # Cache the loaded data
            self.category_point_clouds[category] = pc_data

            return pc_data
        except Exception as e:
            print(f"Error loading point clouds: {e}")
            return None

    def get_shape_preview(self, category, shape_idx):
        """Get a 3D point cloud visualization for a specific shape"""
        if shape_idx is None or shape_idx < 0:
            return None

        # Get shape ID
        pb_dict, all_ids = self.load_category_embeddings(category)
        if pb_dict is None or not all_ids or shape_idx >= len(all_ids):
            return None

        shape_id = all_ids[shape_idx]

        # Load all point clouds for this category
        pc_data = self.load_category_point_clouds(category)
        if pc_data is None:
            # Fallback to image if point clouds not available
            return self.get_shape_image_preview(category, shape_idx, shape_id)

        # Extract point cloud for this specific shape
        try:
            # Get the arrays from the npz file
            ids = pc_data['ids']
            clouds = pc_data['clouds']

            matching_indices = np.where(ids == shape_id)[0]

            # Check number of matches
            if len(matching_indices) == 0:
                # No matches found - log error and fall back to image
                print(f"Error: Shape ID {shape_id} not found in point cloud data")
                return self.get_shape_image_preview(category, shape_idx, shape_id)
            elif len(matching_indices) > 1:
                # Multiple matches found - unexpected data issue - we will get the first one
                print(f"Warning: Multiple matches ({len(matching_indices)}) found for Shape ID {shape_id}. Using first match.")

            # Get the corresponding point cloud
            matching_idx = matching_indices[0]
            points = clouds[matching_idx]

            # Create 3D visualization
            fig = self.get_shape_pointcloud_preview(points, title=f"Shape #{shape_idx}")
            return fig

        except Exception as e:
            print(f"Error extracting point cloud for {shape_id}: {e}")
            return self.get_shape_image_preview(category, shape_idx, shape_id)

    def get_shape_image_preview(self, category, shape_idx, shape_id):
        """Fallback to image preview if point cloud not available"""
        try:
            preview_image = self.get_ulip_image(shape_id)
            preview_image = preview_image.resize((300, 300))

            # Convert PIL image to plotly figure
            fig = go.Figure()

            # Need to convert PIL image to a format plotly can use
            import io
            import base64

            # Convert PIL image to base64
            buf = io.BytesIO()
            preview_image.save(buf, format='PNG')
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Add image to figure
            fig.add_layout_image(
                dict(
                    source=f"data:image/png;base64,{img_str}",
                    xref="paper", yref="paper",
                    x=0, y=1,
                    sizex=1, sizey=1,
                    sizing="contain",
                    layer="below"
                )
            )

            fig.update_layout(
                title=f"Shape 2D Preview - 3D not available",
                xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
                yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1], scaleanchor="x", scaleratio=1),
                margin=dict(l=0, r=0, b=0, t=0),
                plot_bgcolor='rgba(0,0,0,0)'  # Transparent background
            )

            return fig
        except Exception as e:
            print(f"Error loading preview for {shape_id}: {e}")
            # Create empty figure with error message
            fig = go.Figure()
            fig.update_layout(
                title=f"Error loading Shape #{shape_idx}",
                annotations=[dict(
                    text="Preview not available",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    ont=dict(size=16, color="#E53935"),  # Red error text
                    align="center"
                )],
                margin=dict(l=0, r=0, b=0, t=0, pad=0),
                plot_bgcolor='rgba(0,0,0,0)'  # Transparent background
            )
            return fig

    def get_shape_pointcloud_preview(self, points, title=None):
        """Create a clean 3D point cloud visualization with Y as up axis"""
        # Sample points for better performance (fewer points = smoother interaction)
        sampled_points = points[::1]  # Take every Nth point

        # Create 3D scatter plot with fixed color
        fig = go.Figure(data=[go.Scatter3d(
            x=sampled_points[:, 0],
            y=sampled_points[:, 1],  # Use Z as Y (up axis)
            z=sampled_points[:, 2],  # Use Y as Z
            mode='markers',
            marker=dict(
                size=2.5,
                color='#4285F4',  # Fixed blue color
                opacity=1
            )
        )])

        fig.update_layout(
            title=None,
            scene=dict(
                # Remove all axes elements
                xaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False, showline=False,
                           showbackground=False),
                yaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False, showline=False,
                           showbackground=False),
                zaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False, showline=False,
                           showbackground=False),
                aspectmode='data'  # Maintain data aspect ratio
            ),
            # Eliminate margins
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            autosize=True,
            # Control modebar appearance through layout
            modebar=dict(
                bgcolor='white',
                color='#333',
                orientation='v',  # Vertical orientation
                activecolor='#009688'
            ),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        )

        # Better camera angle
        fig.update_layout(
            scene_camera=dict(
                eye=dict(x=-1.5, y=0.5, z=-1.5),
                up=dict(x=0, y=1, z=0),  # Y is up
                center=dict(x=0, y=0, z=0)
            )
        )

        return fig

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
        
    def on_slider_change(self, shape_idx, category):
        """Update the preview when the slider changes"""
        max_idx = self.category_counts.get(category, 0) - 1

        # Get shape preview
        shape_preview = self.get_shape_preview(category, shape_idx)

        # Update counter text
        counter_text = f"Shape {shape_idx} of {max_idx}"

        return shape_preview, counter_text, shape_idx

    def on_slider_change_no_update(self, shape_idx, category):
        """Handle slider change without updating the plot (used when navigation buttons are clicked)"""
        if self.from_navigation:
            self.from_navigation = False
            # Return the same values without recalculating
            max_idx = self.category_counts.get(category, 0) - 1
            return None, f"Shape {shape_idx} of {max_idx}", shape_idx
        else:
            # Normal processing when slider is moved directly
            return self.on_slider_change(shape_idx, category)

    def prev_shape(self, current_idx, category):
        """Go to previous shape"""
        max_idx = self.category_counts.get(category, 0) - 1
        new_idx = max(0, current_idx - 1)

        # Get preview image
        preview_image = self.get_shape_preview(category, new_idx)

        # Update counter text
        counter_text = f"Shape {new_idx} of {max_idx}"

        # Set a flag to indicate this update came from navigation
        self.from_navigation = True

        return new_idx, preview_image, counter_text

    def next_shape(self, current_idx, category):
        """Go to next shape"""
        max_idx = self.category_counts.get(category, 0) - 1
        new_idx = min(max_idx, current_idx + 1)

        # Get preview image
        preview_image = self.get_shape_preview(category, new_idx)

        # Update counter text
        counter_text = f"Shape {new_idx} of {max_idx}"

        # Set a flag to indicate this update came from navigation
        self.from_navigation = True

        return new_idx, preview_image, counter_text

    def jump_to_start(self, category):
        """Jump to the first shape"""
        max_idx = self.category_counts.get(category, 0) - 1
        new_idx = 0

        # Get preview image
        preview_image = self.get_shape_preview(category, new_idx)

        # Update counter text
        counter_text = f"Shape {new_idx} of {max_idx}"

        # Set a flag to indicate this update came from navigation
        self.from_navigation = True

        return new_idx, preview_image, counter_text

    def jump_to_end(self, category):
        """Jump to the last shape"""
        max_idx = self.category_counts.get(category, 0) - 1
        new_idx = max_idx

        # Get preview image
        preview_image = self.get_shape_preview(category, new_idx)

        # Update counter text
        counter_text = f"Shape {new_idx} of {max_idx}"

        # Set a flag to indicate this update came from navigation
        self.from_navigation = True

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

        # Set a flag to indicate this update came from navigation
        self.from_navigation = True

        return random_idx, preview_image, counter_text

    def random_prompt(self):
        """Select a random prompt from the predefined list"""
        prompts = [
            'a low poly 3d rendering of a [CATEGORY]',
            'an aquarelle drawing of a [CATEGORY]',
            'a photo of a [CATEGORY] on a beach',
            'a charcoal drawing of a [CATEGORY]',
            'a Hieronymus Bosch painting of a [CATEGORY]',
            'a [CATEGORY] under a tree',
            'A Kazimir Malevich painting of a [CATEGORY]',
            'a vector graphic of a [CATEGORY]',
            'a Claude Monet painting of a [CATEGORY]',
            'a Salvador Dali painting of a [CATEGORY]',
            'an Art Deco poster of a [CATEGORY]'
        ]

        # Get a random prompt
        return random.choice(prompts)

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
        print(test_prompt, category_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device} in get_guidance")

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

    @spaces.GPU(duration=120)
    def generate_images(self, prompt, category, selected_shape_idx, guidance_strength, seed):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device} in generate_images")

        # Move models to gpu
        if device.type == "cuda":
            self.pipeline = self.pipeline.to(device)
            self.shape2clip_model = self.shape2clip_model.to(device)

        # Clear status text immediately
        status = ""

        # Replace [CATEGORY] with the selected category (case-insensitive)
        category_pattern = re.compile(r'\[CATEGORY\]', re.IGNORECASE)
        if re.search(category_pattern, prompt):
            # Use re.sub for replacement to maintain the same casing pattern that was used
            final_prompt = re.sub(category_pattern, category, prompt)
        else:
            # Fallback if user didn't use placeholder
            final_prompt = f"{prompt} {category}"
            status = status + f"<div style='padding: 10px; background-color: #f0f7ff; border-left: 5px solid #3498db; margin-bottom: 10px;'>Note: For better results, use [CATEGORY] in your prompt where you want '{category}' to appear, otherwise it is appended at the end of the prompt.</div>"

        error = False
        # Check if prompt contains any other categories
        for other_category in self.available_categories:
            if re.search(r'\b' + re.escape(other_category) + r'\b', prompt):
                status = status + f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Your prompt contains '{other_category}'. Please remove it and use [CATEGORY] instead.</div>"
                error = True
        if error:
            return [], status

        # Load category embeddings if not already loaded
        pb_dict, all_ids = self.load_category_embeddings(category)
        if pb_dict is None or not all_ids:
            status = status + f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Failed to load embeddings for {category}</div>"
            return [], status

        # Ensure shape index is valid
        if selected_shape_idx is None or selected_shape_idx < 0:
            selected_shape_idx = 0

        max_idx = len(all_ids) - 1
        selected_shape_idx = max(0, min(selected_shape_idx, max_idx))
        guidance_shape_id = all_ids[selected_shape_idx]

        # Set generator
        generator = torch.Generator(device=device).manual_seed(seed)

        results = []

        try:
            # Generate base image (without guidance)
            with torch.no_grad():
                base_images = self.pipeline(
                    prompt=final_prompt,
                    num_inference_steps=50,
                    num_images_per_prompt=1,
                    generator=generator,
                    guidance_scale=7.5
                ).images

            results.append(base_images[0])
        except Exception as e:
            print(f"Error generating base image: {e}")
            status = status + f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Error generating base image: {str(e)}</div>"
            return results, status

        try:
            # Get shape guidance embedding
            pb_emb = pb_dict[guidance_shape_id]
            out_guidance, prompt_emb = self.get_guidance(final_prompt, category, pb_emb)
        except Exception as e:
            print(f"Error generating guidance: {e}")
            status = status + f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Error generating guidance: {str(e)}</div>"
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

            results.append(guided_images[0])

            # Success status
            status = status + f"<div style='padding: 10px; background-color: #e8f5e9; border-left: 5px solid #4caf50; margin-bottom: 10px;'>✓ Successfully generated images using Shape #{selected_shape_idx} from category '{category}'.</div>"

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error generating guided image: {e}")
            status = status + f"<div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #e74c3c; font-weight: bold; margin-bottom: 10px;'>⚠️ ERROR: Error generating guided image: {str(e)}</div>"

        return results, status

    def on_demo_load(self):
        """Function to ensure initial image is loaded when demo starts"""
        default_category = "chair" if "chair" in self.available_categories else self.available_categories[0]
        initial_img = self.get_shape_preview(default_category, 0)
        return initial_img

    def create_ui(self):
        # Ensure chair is in available categories, otherwise use the first available
        default_category = "chair" if "chair" in self.available_categories else self.available_categories[0]

        with gr.Blocks(title="ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts",
                       theme=gr.themes.Soft(
                           primary_hue="orange",
                           secondary_hue="blue",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
                           font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"],
                       ),
                       css="""
                      /* Base styles */
                      .container { max-width: 1400px; margin: 0 auto; }

                      /* Typography */
                      .title { text-align: center; font-size: 26px; font-weight: 600; margin-bottom: 3px; }
                      .subtitle { text-align: center; font-size: 16px; margin-bottom: 3px; }
                      .authors { text-align: center; font-size: 15px; margin-bottom: 3px; }
                      .affiliations { text-align: center; font-size: 13px; margin-bottom: 5px; }

                      /* Buttons */
                      .buttons-container { margin: 0 auto 10px; }
                      .buttons-row { display: flex; justify-content: center; gap: 10px; }
                      .nav-button {
                          display: inline-block;
                          padding: 6px 12px;
                          background-color: #363636;
                          color: white !important;
                          text-decoration: none;
                          border-radius: 20px;
                          font-weight: 500;
                          font-size: 14px;
                          transition: background-color 0.2s;
                          text-align: center;
                      }
                      .nav-button:hover { background-color: #505050; }
                      .nav-button.disabled { 
                          opacity: 0.6; 
                          cursor: not-allowed;
                      }

                      /* Form elements */
                      .prompt-text { font-size: 16px; }
                      .instruction-text { font-size: 15px; padding: 10px; border-radius: 8px; background-color: rgba(255, 165, 0, 0.1); }
                      .shape-navigation { 
                          display: flex; 
                          justify-content: center; 
                          align-items: center; 
                          margin: 10px auto;
                          gap: 15px;
                          max-width: 320px;
                      }
                      .shape-navigation button { 
                          min-width: 40px; 
                          max-width: 60px; 
                          width: auto; 
                          padding: 6px 10px; 
                      }
                      .nav-icon-btn { font-size: 18px; }
                      .category-dropdown .wrap { font-size: 16px; }
                      .generate-button { font-size: 18px !important; padding: 12px !important; margin: 15px 0 !important; }
                      .slider-label { font-size: 16px; }
                      .slider-text { font-size: 14px; margin-top: 5px; }
                      .about-section { font-size: 16px; margin-top: 40px; padding: 20px; border-top: 1px solid rgba(128, 128, 128, 0.2); }
                      .status-message { background-color: rgba(0, 128, 0, 0.1); color: #006400; padding: 10px; border-radius: 4px; margin-top: 10px; }
                      .prompt-container { display: flex; align-items: center; }
                      .prompt-input { flex-grow: 1; }
                      .prompt-button { margin-left: 10px; align-self: center; }
                      .results-gallery { min-height: 100px; max-height: 500px; }

                      /* Responsive adjustments */
                      @media (max-width: 768px) {
                          .shape-navigation { 
                              max-width: 100%;
                              gap: 5px;
                          }
                          .shape-navigation button { 
                              min-width: 36px;
                              padding: 6px 0;
                              font-size: 16px;
                          }
                          .buttons-row {
                              flex-wrap: wrap;
                          }
                          .nav-button {
                              margin-bottom: 5px;
                          }
                          .results-gallery {
                              max-height: 320px;
                          }
                      }

                      /* Dark mode overrides */
                      @media (prefers-color-scheme: dark) {
                          .nav-button {
                              background-color: #505050;
                          }
                          .nav-button:hover {
                              background-color: #666666;
                          }
                      }
                      """) as demo:
            # Header with title and links
            gr.Markdown("# ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts",
                        elem_classes="title")
            gr.Markdown("### CVPR 2025", elem_classes="subtitle")
            gr.Markdown(
                "Dmitry Petrov<sup>1</sup>, Pradyumn Goyal<sup>1</sup>, Divyansh Shivashok<sup>1</sup>, Yuanming Tao<sup>1</sup>, Melinos Averkiou<sup>2,3</sup>, Evangelos Kalogerakis<sup>1,2,4</sup>",
                elem_classes="authors")
            gr.Markdown(
                "<sup>1</sup>UMass Amherst    <sup>2</sup>CYENS CoE    <sup>3</sup>University of Cyprus    <sup>4</sup>TU Crete",
                elem_classes="affiliations")

            # Navigation buttons
            with gr.Row():
                with gr.Column(scale=3):
                    pass  # Empty space for alignment
                with gr.Column(scale=2, elem_classes="buttons-container"):
                    gr.HTML("""
                    <div class="buttons-row">
                        <a href="https://arxiv.org/abs/2412.02912" target="_blank" class="nav-button">
                            arXiv
                        </a>
                        <a href="https://lodurality.github.io/shapewords/" target="_blank" class="nav-button">
                            Project Page
                        </a>
                        <a href="#" target="_blank" class="nav-button disabled">
                            Code
                        </a>
                        <a href="#" target="_blank" class="nav-button disabled">
                            Data
                        </a>
                    </div>
                    """)
                with gr.Column(scale=3):
                    pass  # Empty space for alignment

            # Hidden field to store selected shape index
            selected_shape_idx = gr.Number(value=0, visible=False)

            # Prompt Design (full width)
            with gr.Group():
                gr.Markdown("### 📝 Prompt Design")

                with gr.Row():
                    category = gr.Dropdown(
                        label="Object Category",
                        choices=self.available_categories,
                        value=default_category,
                        container=True,
                        elem_classes="category-dropdown",
                        scale=2
                    )

                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="an aquarelle drawing of a [CATEGORY]",
                        value="an aquarelle drawing of a [CATEGORY]",
                        lines=1,
                        scale=5,
                        elem_classes="prompt-input"
                    )

                    random_prompt_btn = gr.Button("🎲 Random\nPrompt",
                                                  size="lg",
                                                  scale=1,
                                                  elem_classes="prompt-button")

                gr.Markdown("""
                Use [CATEGORY] in your prompt where you want the selected object type to appear.
                For example: "a watercolor painting of a [CATEGORY] in the forest"
                """, elem_classes="instruction-text")

            # Middle section - Shape Selection and Results side by side
            with gr.Row(equal_height=False):
                # Left column - Shape Selection
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("### 🔍 Shape Selection")

                        shape_slider = gr.Slider(
                            minimum=0,
                            maximum=self.category_counts.get(default_category, 0) - 1,
                            step=1,
                            value=0,
                            label="Shape Index",
                            interactive=True
                        )

                        shape_counter = gr.Markdown(f"Shape 0 of {self.category_counts.get(default_category, 0) - 1}")

                        gr.Markdown("### Selected Shape (3D Point Cloud)")

                        current_shape_plot = gr.Plot(show_label=False)

                        # Navigation buttons - Icons only for better mobile compatibility
                        with gr.Row(elem_classes="shape-navigation"):
                            jump_start_btn = gr.Button("⏮️", size="sm", elem_classes="nav-icon-btn")
                            prev_shape_btn = gr.Button("◀️", size="sm", elem_classes="nav-icon-btn")
                            random_btn = gr.Button("🎲", size="sm", variant="secondary", elem_classes="nav-icon-btn")
                            next_shape_btn = gr.Button("▶️", size="sm", elem_classes="nav-icon-btn")
                            jump_end_btn = gr.Button("⏭️", size="sm", elem_classes="nav-icon-btn")

                # Right column - Results
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("### 🖼️ Generated Results")
                        gallery = gr.Gallery(
                            label="Results",
                            show_label=False,
                            elem_id="results_gallery",
                            columns=2,
                            height="auto",
                            object_fit="contain",
                            elem_classes="results-gallery"
                        )

            # Generate button (full width)
            with gr.Row():
                run_button = gr.Button("✨ Generate Images", variant="primary", size="lg",
                                       elem_classes="generate-button")

            # Generation Settings (full width)
            with gr.Group():
                gr.Markdown("### ⚙️ Generation Settings")

                with gr.Row():
                    with gr.Column():
                        guidance_strength = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.1, value=0.9,
                            label="Guidance Strength (λ) - Higher λ = stronger shape adherence"
                        )
                    with gr.Column():
                        seed = gr.Slider(
                            minimum=0, maximum=10000, step=1, value=42,
                            label="Random Seed"
                        )

                status_text = gr.HTML("", elem_classes="status-message")

            # About section at the bottom of the page
            with gr.Group(elem_classes="about-section"):
                gr.Markdown("""
                ## About ShapeWords

                ShapeWords incorporates target 3D shape information with text prompts to guide image synthesis.

                ### How It Works
                1. Select an object category from the dropdown menu
                2. Browse through available 3D shapes using the slider or navigation buttons
                3. Create a text prompt using [CATEGORY] as a placeholder
                4. Adjust guidance strength to control shape influence
                5. Click Generate to create images that follow both your text prompt and the selected 3D shape

                ### Citation
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

            # Connect components

            # Make sure the initial image is loaded when the demo starts
            demo.load(
                fn=self.on_demo_load,
                inputs=None,
                outputs=[current_shape_plot]
            )

            # Connect slider to update preview
            shape_slider.change(
                fn=self.on_slider_change,
                inputs=[shape_slider, category],
                outputs=[current_shape_plot, shape_counter, selected_shape_idx]
            )

            # Previous shape button
            prev_shape_btn.click(
                fn=self.prev_shape,
                inputs=[selected_shape_idx, category],
                outputs=[shape_slider, current_shape_plot, shape_counter]
            )

            # Next shape button
            next_shape_btn.click(
                fn=self.next_shape,
                inputs=[selected_shape_idx, category],
                outputs=[shape_slider, current_shape_plot, shape_counter]
            )

            # Jump to start button
            jump_start_btn.click(
                fn=self.jump_to_start,
                inputs=[category],
                outputs=[shape_slider, current_shape_plot, shape_counter]
            )

            # Jump to end button
            jump_end_btn.click(
                fn=self.jump_to_end,
                inputs=[category],
                outputs=[shape_slider, current_shape_plot, shape_counter]
            )

            # Random shape button
            random_btn.click(
                fn=self.random_shape,
                inputs=[category],
                outputs=[shape_slider, current_shape_plot, shape_counter]
            )

            # Connect the random prompt button
            random_prompt_btn.click(
                fn=self.random_prompt,
                inputs=[],
                outputs=[prompt]
            )

            # Update the UI when category changes
            category.change(
                fn=self.on_category_change,
                inputs=[category],
                outputs=[shape_slider, selected_shape_idx, current_shape_plot, shape_counter]
            )

            # Update status text when generating
            run_button.click(
                fn=lambda: """<div style='color: #00cc00; background-color: #1a1a1a; 
                            border: 1px solid #2a2a2a; padding: 10px; border-radius: 4px; 
                            margin-top: 10px; font-weight: bold;'>
                            Generating images...</div>""",
                inputs=None,
                outputs=[status_text]
            )

            # Generate images when button is clicked
            run_button.click(
                fn=lambda p, c, s_idx, g, seed: [
                    [
                        (img, caption) for img, caption in zip(
                        self.generate_images(p, c, s_idx, g, seed)[0],
                        [f"Unguided Result", f"Guided Result (λ = {g})"]
                    )
                    ],  # Gallery images with captions
                    f"""<div style="color: #00cc00; background-color: #1a1a1a; 
                       border: 1px solid #2a2a2a; padding: 10px; border-radius: 4px; 
                       margin-top: 10px; font-weight: bold;">
                       ✓ Successfully generated images using Shape #{s_idx} from category '{c}'.</div>"""
                ],
                inputs=[prompt, category, selected_shape_idx, guidance_strength, seed],
                outputs=[gallery, status_text]
            )

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

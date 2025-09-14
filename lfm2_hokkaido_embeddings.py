#!/usr/bin/env python3
"""
LFM2-VL Multimodal Embedding Generation Script for Hokkaido Listings

Creates multimodal embeddings using LiquidAI's LFM2-VL-1.6B model and fuses them
with existing 9D property vectors using an MLP fusion layer. Stores results in vecs.

Usage:
    python lfm2_hokkaido_embeddings.py --max-listings 100 --batch-size 10
    python lfm2_hokkaido_embeddings.py --listing-id 77810038 --debug
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from dotenv import load_dotenv
import vecs

# Add the scraper directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent / "scraper"))

from supabase import create_client, Client
from suumo.helpers.database import SuumoDatabaseHandler


class MultimodalFusionMLP(nn.Module):
    """MLP layer for fusing LFM2-VL embeddings with 9D property vectors."""
    
    def __init__(self, lfm2_dim: int, property_dim: int = 9, hidden_dims: List[int] = [512, 256], output_dim: int = 128):
        super().__init__()
        self.lfm2_dim = lfm2_dim
        self.property_dim = property_dim
        
        # Create MLP layers
        layers = []
        input_dim = lfm2_dim + property_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, lfm2_embedding: torch.Tensor, property_vector: torch.Tensor) -> torch.Tensor:
        """Forward pass through fusion MLP."""
        # Concatenate inputs
        fused_input = torch.cat([property_vector, lfm2_embedding], dim=-1)
        
        # Pass through MLP
        output = self.mlp(fused_input)
        
        # L2 normalize output
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class LFM2VLEmbeddingGenerator:
    """Handles LFM2-VL model loading and multimodal embedding generation."""
    
    def __init__(self, model_name: str = "LiquidAI/LFM2-VL-1.6B"):
        self.model_name = model_name
        self.fusion_method = "mlp_fusion"
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LFM2-VL will use device: {self.device}")
        if self.device == "cuda":
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        
        # MLP fusion parameters
        self.fusion_mlp = None
        self.mlp_output_dim = 128

        # Setup requests session with retries and connection pooling
        self.requests_session = requests.Session()
        retries = Retry(
            total=3, 
            backoff_factor=2, 
            status_forcelist=[500, 502, 503, 504, 408, 429],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=20,
            pool_maxsize=20,
            pool_block=False
        )
        self.requests_session.mount('http://', adapter)
        self.requests_session.mount('https://', adapter)
        
        # Set headers for better CloudFront compatibility
        self.requests_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ImageDownloader/1.0)',
            'Accept': 'image/*,*/*;q=0.8',
            'Connection': 'keep-alive'
        })
        
    def load_model(self):
        """Load the LFM2-VL model and processor."""
        self.logger.info(f"Loading LFM2-VL model: {self.model_name} on device: {self.device}")
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            
            # For LFM2-VL, use AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.logger.info("LFM2-VL model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading LFM2-VL model: {e}")
            raise
    
    def generate_image_embeddings(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for a batch of images using LFM2-VL."""
        if not self.model:
            self.load_model()
            
        try:
            # Load and prepare all images
            images = []
            valid_indices = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    image = Image.open(image_path).convert("RGB")
                    images.append(image)
                    valid_indices.append(i)
                except Exception as e:
                    self.logger.error(f"Error loading image {image_path}: {e}")
            
            if not images:
                self.logger.error("No valid images found for embedding generation")
                return None
            
            self.logger.info(f"Processing {len(images)} images for embedding generation")
            
            # Generate embeddings for all images
            all_embeddings = []
            
            # Process images one by one to avoid memory issues and use proper LFM2-VL interface
            for image in images:
                try:
                    # Create a simple conversation for each image to get internal representations
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": "Describe this image."}
                            ],
                        },
                    ]
                    
                    # Apply chat template to get proper inputs
                    inputs = self.processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        tokenize=True,
                    ).to(self.device)
                    
                    with torch.no_grad():
                        # Get model outputs with hidden states
                        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
                        
                        # Extract vision embeddings from LFM2-VL
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                            # Use the last hidden state
                            embeddings = outputs.hidden_states[-1]
                        else:
                            # Fallback to last_hidden_state
                            embeddings = outputs.last_hidden_state
                        
                        # Average pool over sequence dimension to get image representation
                        # Shape: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
                        pooled_embedding = torch.mean(embeddings, dim=1).squeeze(0)
                        
                        # Move to CPU and convert to numpy
                        embedding_np = pooled_embedding.cpu().numpy()
                        all_embeddings.append(embedding_np)
                        
                except Exception as e:
                    self.logger.error(f"Error processing individual image: {e}")
                    continue
            
            if not all_embeddings:
                self.logger.error("No embeddings were successfully generated")
                return None
            
            # Convert list to numpy array and perform final average pooling across all images
            all_embeddings = np.array(all_embeddings)
            final_embedding = np.mean(all_embeddings, axis=0)  # Average across all images
            
            self.logger.info(f"Generated final embedding with shape: {final_embedding.shape}")
            return final_embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return None

    def download_image(self, url: str, temp_dir: str) -> Optional[str]:
        """Download an image from URL to temporary directory."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Downloading image (attempt {attempt + 1}/{max_retries}): {url}")
                
                response = self.requests_session.get(url, timeout=(5, 30), stream=True)
                response.raise_for_status()
                
                filename = url.split('/')[-1]
                if not filename or '.' not in filename:
                    filename = f"image_{hash(url) % 100000}.jpg"
                
                filepath = os.path.join(temp_dir, filename)
                
                # Download with streaming to handle large files
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                
                # Verify it's a valid image
                with Image.open(filepath) as img:
                    img.verify()
                
                self.logger.debug(f"Successfully downloaded: {filepath}")
                return filepath
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout downloading {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
            except Exception as e:
                self.logger.error(f"Error downloading image from {url} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                
        self.logger.error(f"Failed to download after {max_retries} attempts: {url}")
        return None
    
    def init_fusion_mlp(self, lfm2_dim: int):
        """Initialize the fusion MLP if not already created."""
        if self.fusion_mlp is None:
            self.fusion_mlp = MultimodalFusionMLP(
                lfm2_dim=lfm2_dim,
                property_dim=9,
                hidden_dims=[512, 256],
                output_dim=self.mlp_output_dim
            ).to(self.device)
            
            # Set to evaluation mode to handle single samples
            self.fusion_mlp.eval()
            
            # Convert MLP to the correct dtype to match LFM2-VL model
            if self.device == "cuda":
                self.fusion_mlp = self.fusion_mlp.half()  # Convert to float16
            
            self.logger.info(f"Initialized fusion MLP with output dim: {self.mlp_output_dim}")
    
    def fuse_embeddings(self, lfm2_embedding: np.ndarray, nine_d_vector: np.ndarray) -> np.ndarray:
        """Fuse LFM2-VL embedding with existing 9D vector using MLP fusion."""
        
        # Initialize MLP if needed
        self.init_fusion_mlp(len(lfm2_embedding))
        
        # Determine the correct dtype based on model precision
        model_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Convert to tensors with matching dtype
        lfm2_tensor = torch.tensor(lfm2_embedding, dtype=model_dtype).unsqueeze(0).to(self.device)
        nine_d_tensor = torch.tensor(nine_d_vector, dtype=model_dtype).unsqueeze(0).to(self.device)
        
        # Pass through MLP
        with torch.no_grad():
            fused_tensor = self.fusion_mlp(lfm2_tensor, nine_d_tensor)
            fused_vector = fused_tensor.squeeze(0).cpu().numpy()
        
        self.logger.info(f"MLP fusion: {len(lfm2_embedding)} + {len(nine_d_vector)} -> {len(fused_vector)}")
        return fused_vector


class LFM2VLListingProcessor:
    """Processes Hokkaido listings with LFM2-VL multimodal embeddings."""
    
    def __init__(self, 
                 db_handler: SuumoDatabaseHandler, 
                 embedding_generator: LFM2VLEmbeddingGenerator):
        
        self.db_handler = db_handler
        self.embedding_generator = embedding_generator
        self.logger = logging.getLogger(__name__)
        
        # Initialize vector store with better error handling
        DB_CONNECTION = os.getenv('POSTGRES_VECS_URL')
        print(DB_CONNECTION)
        if not DB_CONNECTION:
            raise ValueError("POSTGRES_VECS_URL environment variable is required")
        
        try:
            self.logger.info("Connecting to vector database...")
            self.vx = vecs.create_client(DB_CONNECTION)
            self.logger.info("Vector database connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to vector database: {e}")
            self.logger.error("This is likely a network connectivity issue. Check your internet connection.")
            raise
        
        # Use MLP output dimension
        final_dim = self.embedding_generator.mlp_output_dim
        
        # Create collection for LFM2-VL embeddings
        collection_name = f"hokkaido_lfm2_fused_mlp"
        self.embeddings_collection = self.vx.get_or_create_collection(
            name=collection_name,
            dimension=final_dim
        )
        
        self.logger.info(f"Created collection '{collection_name}' with dimension {final_dim}")
        
    def _get_existing_embedding_ids(self) -> set:
        """Fetch all existing listing IDs from the embeddings collection using the vecs client."""
        self.logger.info("Fetching existing embedding IDs from vector store using vecs client...")
        try:
            # The vecs `query` method with a large limit is the way to fetch multiple records.
            # We provide a dummy vector. The main goal is to get the IDs.
            # This is an optimization that might not be supported by all vecs versions.
            all_records = self.embeddings_collection.query(
                data=[0] * self.embeddings_collection.dimension,
                limit=1_000_000, # A large number to effectively get all records
                include_vector=False
            )
            
            if not all_records:
                self.logger.info("No existing embeddings found in the collection.")
                return set()

            # The result format is [(id, vector, metadata)], we just need the id.
            existing_ids = {record[0] for record in all_records}
            self.logger.info(f"Found {len(existing_ids)} existing embeddings.")
            return existing_ids
        except TypeError:
            # This is a fallback for when `include_vector=False` is not supported
            self.logger.warning("`include_vector=False` might not be supported. Retrying without it.")
            try:
                all_records = self.embeddings_collection.query(
                    data=[0] * self.embeddings_collection.dimension,
                    limit=1_000_000
                )
                if not all_records:
                    self.logger.info("No existing embeddings found in the collection (retry).")
                    return set()
                existing_ids = {record[0] for record in all_records}
                self.logger.info(f"Found {len(existing_ids)} existing embeddings (retry).")
                return existing_ids
            except Exception as e:
                self.logger.error(f"Could not fetch existing embedding IDs using vecs (retry): {e}", exc_info=True)
                self.logger.warning("Proceeding without checking for existing embeddings.")
                return set()
        except Exception as e:
            self.logger.error(f"Could not fetch existing embedding IDs using vecs: {e}", exc_info=True)
            self.logger.warning("Proceeding without checking for existing embeddings.")
            return set()

    def get_hokkaido_listings(self, limit: Optional[int] = None, skip_existing: bool = True) -> List[Dict]:
        """Get Hokkaido listings for processing with pagination support."""
        try:
            # Filter for Hokkaido
            hokkaido_filters = [
                'location.like.%北海道%',
                'location.like.%札幌%', 
                'location.like.%函館%',
                'location.like.%旭川%',
                'location.like.%釧路%',
                'location.like.%帯広%',
                'location.like.%北見%',
                'location.like.%苫小牧%'
            ]
            
            all_listings = []
            page_size = 1000  # Supabase default limit
            offset = 0
            
            self.logger.info("Fetching Hokkaido listings with pagination...")
            
            while True:
                # Base query for this page
                query = self.db_handler.supabase.table('listings').select(
                    'listing_id, title, location, price, lat, lng, rooms, size_sqm, listing_type'
                ).eq('is_active', True).not_.is_('lat', 'null').not_.is_('lng', 'null')
                
                # Apply Hokkaido filters
                hokkaido_query = query.or_(', '.join(hokkaido_filters))
                
                # Add pagination
                hokkaido_query = hokkaido_query.range(offset, offset + page_size - 1)
                
                result = hokkaido_query.execute()
                page_data = result.data
                
                if not page_data:
                    break  # No more data
                
                all_listings.extend(page_data)
                self.logger.info(f"Fetched page {offset//page_size + 1}: {len(page_data)} listings "
                               f"(total so far: {len(all_listings)})")
                
                # If we got less than page_size, we've reached the end
                if len(page_data) < page_size:
                    break
                
                offset += page_size
                
                # Optional: break early if we have enough for the requested limit
                # (but continue to get more since we'll filter for existing embeddings)
                if limit and len(all_listings) >= limit * 3:  # Get extra for filtering
                    break
            
            self.logger.info(f"Pagination complete: fetched {len(all_listings)} total Hokkaido listings")
            
            if skip_existing:
                existing_ids = self._get_existing_embedding_ids()
                if existing_ids:
                    original_count = len(all_listings)
                    all_listings = [
                        listing for listing in all_listings 
                        if listing['listing_id'] not in existing_ids
                    ]
                    filtered_count = len(all_listings)
                    self.logger.info(f"Filtered out {original_count - filtered_count} listings that already have embeddings.")
                    self.logger.info(f"{filtered_count} new listings remaining for processing.")

            return all_listings[:limit] if limit else all_listings
            
        except Exception as e:
            self.logger.error(f"Error fetching Hokkaido listings: {e}")
            return []
    

    
    def get_listing_images(self, listing_id: str, max_images: int = 5) -> List[Dict]:
        """Get image URLs for a listing."""
        try:
            result = self.db_handler.supabase.table('images')\
                .select('cloudfront_url, source_url, is_main')\
                .eq('listing_id', listing_id)\
                .order('is_main', desc=True)\
                .limit(max_images)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            self.logger.error(f"Error fetching images for listing {listing_id}: {e}")
            return []
    
    def get_existing_9d_vector(self, listing_id: str) -> Optional[np.ndarray]:
        """Get existing 9D vector from vecs.listing_vecs collection."""
        try:
            # Try to get the listing_vecs collection
            try:
                listing_vecs_collection = self.vx.get_or_create_collection("listing_vecs", dimension=9)
            except:
                self.logger.warning(f"listing_vecs collection not accessible, using zero vector for {listing_id}")
                return None
            
            # Try to query using the listing_id as the record ID directly
            try:
                # First try with metadata filter
                results = listing_vecs_collection.query(
                    data=[0] * 9,  # Dummy query vector
                    limit=1,
                    filters={"listing_id": {"$eq": listing_id}}
                )
                
                if results and len(results) > 0:
                    vector = results[0][1]  # results format: [(id, vector, metadata)]
                    self.logger.info(f"Retrieved 9D vector for {listing_id}: shape {len(vector) if vector is not None else 'None'}")
                    return np.array(vector) if vector is not None else None
                    
            except Exception as metadata_error:
                self.logger.debug(f"Metadata query failed: {metadata_error}")
                
                # Fallback: try using listing_id as direct record ID
                try:
                    # Use the Supabase function instead
                    result = self.db_handler.supabase.rpc(
                        'get_listing_vector', 
                        {'p_listing_id': listing_id}
                    ).execute()
                    
                    if result.data:
                        vector = result.data
                        self.logger.info(f"Retrieved 9D vector via RPC for {listing_id}: shape {len(vector) if vector is not None else 'None'}")
                        return np.array(vector) if vector is not None else None
                        
                except Exception as rpc_error:
                    self.logger.debug(f"RPC query also failed: {rpc_error}")
            
            self.logger.warning(f"No existing 9D vector found for listing {listing_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching 9D vector for listing {listing_id}: {e}")
            return None
    
    def process_single_listing(self, listing: Dict, temp_dir: str) -> Optional[Tuple[str, np.ndarray]]:
        """Process a single listing with LFM2-VL multimodal embedding."""
        listing_id = listing['listing_id']
        
        try:
            # Get listing images
            image_info_list = self.get_listing_images(listing_id, max_images=10)
            if not image_info_list:
                self.logger.warning(f"No images found for listing {listing_id}")
                return None
            
            # Download images
            image_paths = []
            for image_info in image_info_list:
                url = image_info['cloudfront_url'] or image_info['source_url']
                if not url:
                    continue
                    
                image_path = self.embedding_generator.download_image(url, temp_dir)
                if image_path:
                    image_paths.append(image_path)
            
            if not image_paths:
                self.logger.error(f"No images could be downloaded for listing {listing_id}")
                return None
            
            self.logger.debug(f"Downloaded {len(image_paths)} images for listing {listing_id}")
            
            # Generate LFM2-VL embeddings
            lfm2_embedding = self.embedding_generator.generate_image_embeddings(image_paths)
            if lfm2_embedding is None:
                self.logger.error(f"Failed to generate LFM2-VL embedding for listing {listing_id}")
                return None
            
            # Clean up images
            for image_path in image_paths:
                try:
                    os.remove(image_path)
                except:
                    pass
            
            # Get existing 9D vector
            nine_d_vector = self.get_existing_9d_vector(listing_id)
            if nine_d_vector is None:
                self.logger.warning(f"No existing 9D vector found for listing {listing_id}, using zero vector")
                nine_d_vector = np.zeros(9)
            
            # Fuse vectors
            fused_vector = self.embedding_generator.fuse_embeddings(lfm2_embedding, nine_d_vector)
            
            self.logger.debug(f"Generated fused embedding for {listing_id}: "
                            f"lfm2_dim={len(lfm2_embedding)}, "
                            f"9d_dim={len(nine_d_vector)}, "
                            f"fused_dim={len(fused_vector)}")
            
            return listing_id, fused_vector
            
        except Exception as e:
            self.logger.error(f"Error processing listing {listing_id}: {e}")
            return None
    
    def store_embeddings_batch(self, embeddings_batch: List[Tuple[str, np.ndarray, Dict]]):
        """Store batch of LFM2-VL fused embeddings."""
        try:
            self.embeddings_collection.upsert(embeddings_batch)
            self.logger.info(f"Stored batch of {len(embeddings_batch)} LFM2-VL embeddings")
            
        except Exception as e:
            self.logger.error(f"Error storing embeddings batch: {e}")
            # Fallback to individual inserts
            for item in embeddings_batch:
                try:
                    self.embeddings_collection.upsert([item])
                except Exception as individual_error:
                    self.logger.error(f"Failed individual insert for {item[0]}: {individual_error}")
    
    def process_listings_batch(self, listings: List[Dict], batch_size: int = 10):
        """Process listings in batches with LFM2-VL embeddings."""
        total_processed = 0
        total_successful = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.logger.info(f"Using temporary directory: {temp_dir}")
            
            for i in range(0, len(listings), batch_size):
                batch = listings[i:i + batch_size]
                embeddings_batch = []
                
                self.logger.info(f"Processing batch {i//batch_size + 1}: "
                               f"listings {i+1}-{min(i+batch_size, len(listings))}")
                
                for listing in batch:
                    result = self.process_single_listing(listing, temp_dir)
                    if result:
                        listing_id, embedding = result
                        
                        # Enhanced metadata
                        metadata = {
                            "title": listing.get('title'),
                            "location": listing.get('location'),
                            "price": listing.get('price'),
                            "rooms": listing.get('rooms'),
                            "size_sqm": listing.get('size_sqm'),
                            "listing_type": listing.get('listing_type'),
                            "fusion_method": "mlp_fusion",
                            "embedding_model": self.embedding_generator.model_name,
                            "created_at": time.time()
                        }
                        
                        embeddings_batch.append((listing_id, embedding.tolist(), metadata))
                        total_successful += 1
                    
                    total_processed += 1
                
                # Store batch
                if embeddings_batch:
                    self.store_embeddings_batch(embeddings_batch)
                
                self.logger.info(f"Batch complete: {len(embeddings_batch)} embeddings generated")
        
        self.logger.info(f"Processing complete: {total_successful}/{total_processed} listings processed")
        
        # Create index
        try:
            self.embeddings_collection.create_index()
            self.logger.info("Created vector index for LFM2-VL embeddings")
        except Exception as e:
            self.logger.warning(f"Could not create index: {e}")


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def setup_supabase_client() -> Client:
    """Setup Supabase client."""
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY environment variables are required")
    
    return create_client(supabase_url, supabase_key)


def main():
    """Main function."""
    # Load environment
    root_dir = Path(__file__).parent.parent
    env_path = root_dir / '.env'
    load_dotenv(env_path)
    
    parser = argparse.ArgumentParser(description="Generate LFM2-VL multimodal embeddings for Hokkaido listings")
    parser.add_argument("--max-listings", type=int, default=100,
                       help="Maximum number of listings to process (use 0 for no limit)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for processing")
    parser.add_argument("--listing-id", help="Process specific listing ID only")
    parser.add_argument("--model-name", default="LiquidAI/LFM2-VL-1.6B",
                       help="LFM2-VL model to use")
    parser.add_argument("--mlp-output-dim", type=int, default=128,
                       help="Output dimension for MLP fusion")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                       help="Skip listings that already have embeddings (default: True)")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Reprocess all listings, even those with existing embeddings")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        logger.info("Setting up database connection...")
        supabase_client = setup_supabase_client()
        db_handler = SuumoDatabaseHandler(supabase_client, logger=logger)
        
        logger.info("Initializing LFM2-VL embedding generator...")
        embedding_generator = LFM2VLEmbeddingGenerator(
            model_name=args.model_name
        )
        embedding_generator.mlp_output_dim = args.mlp_output_dim
        
        processor = LFM2VLListingProcessor(db_handler, embedding_generator)
        
        # Get listings
        if args.listing_id:
            logger.info(f"Processing single listing: {args.listing_id}")
            result = db_handler.supabase.table('listings').select(
                'listing_id, title, location, price, lat, lng, rooms, size_sqm, listing_type'
            ).eq('listing_id', args.listing_id).execute()
            
            if not result.data:
                logger.error(f"Listing {args.listing_id} not found")
                return
            listings = result.data
        else:
            skip_existing = not args.force_reprocess
            limit = None if args.max_listings == 0 else args.max_listings
            limit_text = "all" if limit is None else str(args.max_listings)
            logger.info(f"Fetching up to {limit_text} Hokkaido listings...")
            listings = processor.get_hokkaido_listings(limit=limit, skip_existing=skip_existing)
        
        if not listings:
            logger.warning("No listings found to process")
            return
        
        # Process listings
        logger.info(f"Starting processing of {len(listings)} listings with MLP fusion...")
        start_time = time.time()
        
        processor.process_listings_batch(listings, batch_size=args.batch_size)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Processing complete! Total time: {processing_time:.2f}s")
        logger.info(f"Average time per listing: {processing_time/len(listings):.2f}s")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
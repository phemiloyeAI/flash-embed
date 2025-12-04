import torch 
import numpy as np
from typing import List, Union
from  all_clip import load_clip

class TorchBackend:
    def __init__(self, model_name: str = "ViT-B/32", device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        self.model, self.preprocess, self.tokenizer = load_clip(model_name, device=self.device)

    def preprocess_texts(self, texts: List[str]) -> torch.Tensor:
        tokenized_texts = self.tokenizer(texts).to(self.device)
        return tokenized_texts

    def preprocess_images(self, images: List[np.ndarray]) -> torch.Tensor:
        preprocessed_images = [self.preprocess(image).unsqueeze(0) for image in images]
        batch_images = torch.cat(preprocessed_images, dim=0).to(self.device)
        return batch_images

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        tokenized_texts = self.preprocess_texts(texts)
        with torch.no_grad():
            text_embeddings = await self.model.encode_text(tokenized_texts)
        return text_embeddings.cpu().numpy()
    
    async def embed_images(self, images: List[np.ndarray]) -> np.ndarray:
        batch_images = self.preprocess_images(images)
        with torch.no_grad():
            image_embeddings = await self.model.encode_image(batch_images)
        return image_embeddings.cpu().numpy()
    


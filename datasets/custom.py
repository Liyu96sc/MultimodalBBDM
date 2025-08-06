import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os
import json
import open_clip
from torch import nn
from transformers import AutoModel, AutoTokenizer

gpu_device = torch.device("cpu")


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    

class FrozenMedBERTEmbedder(AbstractEncoder):
    """
    Uses a MedBERT transformer encoder for text encoding, with frozen weights.
    Returns token embeddings padded to max_length, then projects to a higher dimension.
    """
    LAYERS = ["last", "penultimate"]

    def __init__(self,
                 model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 device: torch.device = None,
                 max_length: int = 512,
                 proj_dim: int = 1024,
                 freeze: bool = True,
                 layer: str = "last"):
        super().__init__()
        assert layer in self.LAYERS, f"layer must be one of {self.LAYERS}"

        # Device setup
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.max_length = max_length
        self.layer = layer

        # Load tokenizer and model with hidden states
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device)

        # Freeze base model if requested
        if freeze:
            self.freeze()

        # Projection layer: hidden_size -> proj_dim
        hidden_size = self.model.config.hidden_size
        self.projection = nn.Linear(hidden_size, proj_dim).to(self.device)

    def freeze(self):
        """
        Set base model to eval mode and freeze all its parameters.
        Projection remains trainable.
        """
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, texts):
        """
        texts: Union[str, List[str]]
        Returns projected token embeddings with shape [batch_size, max_length, proj_dim].
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        # Extract token embeddings: [batch, max_length, hidden_size]
        token_embs = outputs.hidden_states[-1]
        # Project to desired dimension: [batch, max_length, proj_dim]
        projected = self.projection(token_embs)
        return projected

    def encode(self, text):
        """
        Alias for forward, for compatibility with other encoders.
        """
        return self(text)



class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device=gpu_device, max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        # Load the model onto the specified device (not forced to CPU)
        model, _, _ = open_clip.create_model_and_transforms(arch, device=device, pretrained=version)
        del model.visual
        self.model = model.to(device)  # Move the model to the specified device

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text).to(self.device)  # Ensure tokens are on the same device
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

@Registers.datasets.register_with_name('custom_single')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs = ImagePathDataset(image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/target'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/source'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        img_ori, img_ori_name = self.imgs_ori[i]
        img_cond, img_cond_name = self.imgs_cond[i]
        # print(img_ori_name, img_cond_name)
        # print('--------------------------------')
        return self.imgs_ori[i], self.imgs_cond[i]

@Registers.datasets.register_with_name('custom_aligned_residual')
class CustomAlignedResidualDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/target'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/source'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_residual = None

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        img_ori, img_ori_name = self.imgs_ori[i]
        img_cond, img_cond_name = self.imgs_cond[i]
        # print('we are in custom_aligned_residual')
        # print('--------------------------------')
        img_residual = img_ori - img_cond
        img_residual_name = img_ori_name
        return (img_residual, img_residual_name), (img_cond, img_cond_name)


@Registers.datasets.register_with_name('text_aligned')
class CustomAlignedTextDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()

        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/target'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/source'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.json_path = dataset_config.dataset_path + f'{stage}/prompt.json'
        
        # Read JSON file
        with open(self.json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        # Build mapping from image name to description, preserving original description content (removing quotes)
        self.image_to_description = {item[0]: item[1].strip('"') for item in data if len(item) == 2}
        
        # Use GPU to pre-compute embeddings and cache them
        self.clip_processor = FrozenOpenCLIPEmbedder(arch="ViT-H-14", version="laion2b_s32b_b79k", device='cpu', max_length=77)
        #self.clip_processor = FrozenMedBERTEmbedder(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", device='cpu', max_length=77)
        self.image_to_embedding = {}

        for img_name, description in self.image_to_description.items():
            #print(description)
            embedding = self.clip_processor(description)
            self.image_to_embedding[img_name] = embedding.detach().to('cpu')  # Add detach() to remove gradient information

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        img_ori, img_ori_name = self.imgs_ori[i]
        img_cond, _ = self.imgs_cond[i]

        # First construct two possible keys
        key_jpg = f"{img_ori_name}.jpg"
        key_png = f"{img_ori_name}.png"

        # Try to get text embedding, first try jpg, then try png
        text_embedding = self.image_to_embedding.get(key_jpg)
        description = self.image_to_description.get(key_jpg, "")
        
        if text_embedding is None:
            text_embedding = self.image_to_embedding.get(key_png)
            description = self.image_to_description.get(key_png, "")
            
        if text_embedding is None:
            # If neither format exists, stop code execution
            raise ValueError(f"Text embedding is empty, cannot continue. Please check if the image filename is correct. Tried keys: {key_jpg}, {key_png}")
        
        #print('text_embedding shape', text_embedding.shape)
        
        return img_ori, img_cond, text_embedding, description




@Registers.datasets.register_with_name('custom_colorization_LAB')
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1., 1.)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.datasets.register_with_name('custom_colorization_RGB')
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        cond_image = image.convert('L')
        cond_image = cond_image.convert('RGB')

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
            cond_image = (cond_image - 0.5) * 2.
            cond_image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name('custom_inpainting')
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.
        if index >= self._length:
            index = index - self._length
            p = 1.

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[:, mask_pos_x:mask_pos_x+mask_height, mask_pos_y:mask_pos_y+mask_width] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


import base64
import numpy as np
from torch.utils.data import Dataset
from langchain_core.messages import HumanMessage, SystemMessage

class TextDataset(Dataset):
    def __init__(self, prompts, backup_prompts=None):
        self.prompts = prompts
        if not backup_prompts:
            self.backup_prompts = prompts
        else:
            self.backup_prompts = backup_prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.backup_prompts[idx]

class ImageDataset(Dataset):
    def __init__(self, image_paths, prompt_template):
        self.image_paths = image_paths
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.image_paths)

    # Note: this is a payload for OpenAI API models
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        payload = [
            {
                "type": "text",
                "text": self.prompt_template
            }
        ]
        return payload, image_path


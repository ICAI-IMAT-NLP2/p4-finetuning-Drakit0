import torch
import torch.nn as nn
import math

try:
    from utils import download_and_load_model
except:
    from src.utils import download_and_load_model

class LoRA(nn.Module):
    def __init__(self, original_layer:nn.Linear, r=4, alpha=32):
        """
        Low-Rank Adaptation (LoRA) module.
        
        Args:
            original_layer (nn.Module): The original layer to which LoRA is applied.
            r (int): Rank of the low-rank approximation.
            alpha (int): Scaling factor for the LoRA module.
        """
        super().__init__()
        # Initialize LoRA parameters
        self.r:int = r
        self.alpha: int = alpha
        self.original_layer: nn.Module = original_layer

        # Low-rank matrices A and B for LoRA
        self.A:torch.Tensor = torch.zeros((original_layer.out_features, r))
        self.B: torch.Tensor = torch.zeros((r, original_layer.in_features))

        # Initialize LoRA weights (B is zero-initialized, A is random)
        nn.init.kaiming_uniform_(self.A, a = 5**(1/2))
        
        # Scaling factor alpha 
        self.scaling = alpha/r

        # Freeze the original layer parameters
        for param in original_layer.parameters():
            param.requires_grad = False
                
    def forward(self, x:torch.Tensor):
        # Perform forward pass with low-rank update
        original_out: torch.Tensor = self.original_layer(x)
        LoRa_out: torch.Tensor = x.matmul(self.A@self.B)
        
        return original_out + self.scaling * LoRa_out

def inject_lora_into_model(model:nn.Module, r=4, alpha=32, device='cpu'):
    """
    Inject LoRA layers into the linear layers of the attention modules of the model.
    
    Args:
        model (PreTrainedModel): The pre-trained model.
        r (int): Rank of the low-rank approximation.
        alpha (int): Scaling factor for LoRA.
        device (torch.device): The device to run the model on ('cuda' or 'cpu').
    
    Returns:
        model (PreTrainedModel): The model with LoRA injected into attention layers.
    """
    # Iterate through all child modules of the model
    verify_names:set = set()
    
    for child_name, child_module in model.named_children():
        
        if type(child_module) == nn.Linear:
            verify_names.add(child_name)
            
        # Check if the child module is a linear layer of the attention module
        if child_name.lower() in ["o", "v", "k", "q"]:
            # Create LoRA layer for linear module
            lora_layer = LoRA(child_module, r, alpha)
            setattr(model, child_name, lora_layer)
            
        else:
            inject_lora_into_model(child_module, r, alpha, device)
    
    # print(verify_names)
    return model.to(device)


class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length, model_hidden_size):
        """
        Creates trainable soft prompts to prepend to input embeddings.

        Args:
            prompt_length (int): Number of virtual tokens in the soft prompt.
            model_hidden_size (int): The hidden size of the pre-trained model.
        """
        super().__init__()
        # Initialize soft prompt embeddings
        self.soft_prompt:nn.Parameter = nn.Parameter(torch.zeros((prompt_length, model_hidden_size)))
        nn.init.kaiming_normal_(self.soft_prompt)

    def forward(self, input_embeddings: torch.Tensor):
        """
        Forward pass to prepend soft prompts to input embeddings.

        Args:
            input_embeddings (torch.Tensor): The original input embeddings from the tokenizer.

        Returns:
            torch.Tensor: The concatenated soft prompts and original embeddings.
        """
        # Expand soft prompt to match batch size
        batch_size:int = input_embeddings.shape[0]
        soft_prompt_expanded:torch.Tensor = torch.stack([self.soft_prompt for _ in range(batch_size)])

        # Concatenate soft prompt and input embeddings
        return torch.cat([soft_prompt_expanded, input_embeddings], 1)
from typing import Optional
import torch 
import torch.nn as nn


class siglipVisionConfig: 

    def __init__(
            self,
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: int = 224,
            patch_size: int = 16,
            layer_norm_eps: float = 1e-6,
            attention_dropout: float = 0.1,
            hidden_dropout: float = 0.0,
            num_image_tokens: int = None,
            **kwargs
        ):
            super().__init__()

            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_channels = num_channels
            self.image_size = image_size
            self.patch_size = patch_size
            self.layer_norm_eps = layer_norm_eps
            self.attention_dropout = attention_dropout
            self.hidden_dropout = hidden_dropout
            self.num_image_tokens = num_image_tokens
    
class siglipVisionEmbeddings(nn.Module):
    def __init__(self, config: siglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicate no padding is added 
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent= False
        )
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_size, channels, height, width]
        # Convolve the 'patch_size' kernal over the image, with no overlapping since stride = patch_size
        # The ouput of the convolution will have shape [Batch_size, Embed_dim, Num_patches_height, Num_patches_width]
        # where Num_patches_height = height // patch_size and Num_patches_width = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        # [Batch_size, Embed_dim, Num_patches_height, Num_patches_width] -> [Batch_size, Embed_dim, Num_patches]
        # where Num_patches = Num_patches_height * Num_patches_width
        embeddings = patch_embeds.flatten(2)
        # [Batch_size, Embed_dim, Num_patches] -> [Batch_size, Num_patches, Embed_dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings
    
class SiglipAttention(nn.Module):
    """Multi headed attention"""
    def __init__(self, config: siglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # Equivalent to 1 / sqrt(head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        # hidden_states: [Batch_size, Num_patches, Embed_sim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_size, Num_patches, Embed_dim]
        query_states = self.q_proj(hidden_states) 
        # Key_states : [Batch_size, Num_patches, Embed_dim]
        key_states = self.k_proj(hidden_states)
        # Value_states : [Batch_size, Num_patches, Embed_dim]
        value_states = self.v_proj(hidden_states)
        # Reshape the query, key and value states to [Batch_size, Num_heads, Num_patches, Head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights : [Batch_size, Num_heads, Num_patches, Num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)} but is"
                f" {attn_weights.size()}"
            )
    
        # Apply the softmax row-wise attn_weights: [batch_size, Num_heads, num_patches, Num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1,dtype=torch.float32).to(hidden_states.dtype)

        
        





class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)


    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_size, num_patches, Embed_dim] -> [Batch_size, num_patches, Intermediate_size]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class siglipEncoderLayer(nn.Module):
    def __init__(self, config: siglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: batch_size, Num_patches, Embed_dim
        residual = hidden_states

        # Apllying layer norm 1 without loosing the input tensor shape 
        hidden_states = self.layer_norm1(hidden_states)

        # Self attention
        hidden_states = self.self_attn(hidden_states=hidden_states)

        # Adding the residual connection [Batch_size, num_patches, Embed_dim]
        hidden_states = residual + hidden_states

        # Again making a copy of the input tensor for the skip connection
        residual = hidden_states

        # Apllying layer norm 2 without loosing the input tensor shape
        hidden_states = self.layer_norm2(hidden_states)
        # MLP [Batch_size, num_patches, Embed_dim] -> [Batch_size, num_patches, Embed_dim]
        hidden_states = self.mlp(hidden_states)

        # Adding the residual connection [Batch_size, num_patches, Embed_dim]
        hidden_states = residual + hidden_states

        return hidden_states


class siglipVisionTransformer(nn.Module):
    def __init__(self, config: siglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size


        self.embeddings = siglipVisionEmbeddings(config)
        self.encoder = siglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Pixel_values: [Batch_size, channels, height, width] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class siglipVisionModel(nn.Module):

    
    def __init__(self, config: siglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = siglipVisionTransformer(config)

    def forward(self, pixel_values) -> tuple:
        # [Batch_size, channels, height, width] -> [batch_sizes, Num_patches, Embed_dim]
        return self.vision_model(pixel_values) 
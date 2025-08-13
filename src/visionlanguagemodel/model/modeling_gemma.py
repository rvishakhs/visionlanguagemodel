import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import siglipVisionConfig, siglipVisionModel

class GemmaConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim = 256,
        max_position_embeddings = 8192,
        rms_norm_eps = 1e-6,
        rope_theta = 10000.0,
        attention_bias = False,
        attention_dropout = 0.0,
        pad_token_id = None,
        **kwargs
    ): 
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PalliGemmaConfig:
    def __init__(
        self,
        vision_config= None,
        text_config= None,
        ignore_index= -100,
        image_token_index = 256000,
        vocab_size = 257152,
        projection_dim =2048,
        hidden_size = 2048,
        pad_token_id = None,
        **kwargs
    ): 
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.vision_config = vision_config 
        self.is_encoder_decoder = False


        self.vision_config = siglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2 
        self.vision_config.projection_dim = projection_dim


class GemmaForCausalLM(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self._vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, 
        attention_mask : Optional[torch.Tensor] = None,
        position_ids : Optional[torch.LongTensor] = None, 
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple:
        
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        return_data = {
            "logits": logits
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        return return_data



class PaliGemmaMultiModelProjector(nn.Module):

    def __init__(self, config: PalliGemmaConfig, **kwargs):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.liner(image_features)
        return hidden_states


class paliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PalliGemmaConfig, **kwargs):
        super().__init__()
        self.config = config
        self.vision_tower = siglipVisionModel(config.vision_config)
        self.multi_model_projector = PaliGemmaMultiModelProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1 

    def tie_weights(self): 
        return self.language_model.tie_weights()
    
    def _merge_inputs_ids_with_image_features(
            self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None 
    ): 
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        # combine the embedding of the image tokens, the text tokens and mask out all the padding tokens
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # shape : [batch_size, Seq_len] True for text tokens 
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)

        # shape: [Batch_size, Seq_len] True for image tokens
        image_mask = input_ids == self.config.image_token_index

        # shape : [Batch_size, Seq_len] True for padding tokens 
        pad_mask = input_ids == self.pad_token_id

        # we need to expand the masks 
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embedding 
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Add the image embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### Create the attention mask 
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            casual_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device

            )
        else: 
            assert q_len == 1 
            kv_len = kv_cache.num_items() + q_len

            casual_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        casual_mask = casual_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill((attention_mask == 0), 1).to(device)

        return final_embedding, casual_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple: 
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extract the input embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)


        # 2. Get the vision features
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))

        image_features = self.multi_model_projector(selected_image_feature)

        # 3. Merge the embeddings of the text tokens and the image tokens 
        inputs_embeds, attention_mask, position_ids = self._merge_inputs_ids_with_image_features(image_features, inputs_embeds) 

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs


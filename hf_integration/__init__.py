from transformers import AutoConfig, AutoModel

from .configuration_custom import CustomTransformerConfig
from .model_packing import CustomTransformerModel

AutoConfig.register("custom_transformer", CustomTransformerConfig)
AutoModel.register(CustomTransformerConfig, CustomTransformerModel)
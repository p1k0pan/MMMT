#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig

from .modeling_llama_cd import LlamaModel, LlamaForCausalLM

# from transformers import AutoConfig, AutoModelForCausalLM, \
#                          LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import sys



class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds_icd: Optional[torch.FloatTensor] = None,
        inputs_embeds_vcd: Optional[torch.FloatTensor] = None,
        inputs_embeds_lcd: Optional[torch.FloatTensor] = None,
        sd_am=None,
        image_position=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_position=image_position,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        input_ids_icd: Optional[torch.Tensor] = None,
        images_cd: Optional[torch.Tensor] = None,
        input_ids_lcd: Optional[torch.Tensor] = None,
        sid: Optional[bool] = False,
        img_topk: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if input_ids_icd is not None and input_ids_lcd is not None:
            max_prompt_len = max(inputs.shape[1]-1, input_ids_icd.shape[1]-1, input_ids_lcd.shape[1]-1)
        elif input_ids_icd is not None and input_ids_lcd is None:
            max_prompt_len = max(inputs.shape[1]-1, input_ids_icd.shape[1]-1)
        else:
            max_prompt_len = inputs.shape[1]-1
        inputs_vcd = inputs.clone()
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                image_position
            # ) = self.prepare_inputs_labels_for_multimodal(
            ) = self.prepare_inputs_labels_for_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                max_prompt_len=max_prompt_len
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        if input_ids_icd is not None: 
            (
                input_ids_icd,
                position_ids,
                _,
                _,
                inputs_embeds_icd,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
            # ) = self.prepare_inputs_labels_for_text(
                input_ids_icd,
                position_ids,
                None,
                None,
                None,
                images,
                image_sizes=image_sizes,
                max_prompt_len=max_prompt_len
            ) 
        else:
            inputs_embeds_icd = None
        if images_cd is not None: 
            (
                inputs,
                position_ids,
                _,
                _,
                inputs_embeds_vcd,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs_vcd,
                position_ids,
                None,
                None,
                None,
                images_cd,
                image_sizes=image_sizes,
                max_prompt_len=None
            ) 
        else:
            inputs_embeds_vcd = None
        if input_ids_lcd is not None:
            (
                input_ids_lcd,
                position_ids,
                _,
                _,
                inputs_embeds_lcd,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
            # ) = self.prepare_inputs_labels_for_text(
                input_ids_lcd,
                position_ids,
                None,
                None,
                None,
                images,
                image_sizes=image_sizes,
                max_prompt_len=max_prompt_len
            )
        else:
            inputs_embeds_lcd = None

        if img_topk is not None:
            sd_am = attention_mask.clone()
            img_start = image_position[0][0]
            img_topk = img_topk+img_start
            sd_am[0][img_topk]=False
        else:
            sd_am = None

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            inputs_embeds_icd=inputs_embeds_icd,
            inputs_embeds_vcd=inputs_embeds_vcd,
            inputs_embeds_lcd=inputs_embeds_lcd,
            image_position = image_position[0] if sid else None,
            sd_am = sd_am if img_topk is not None else None,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

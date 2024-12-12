import torch
from typing import Optional
from .modeling_utils import E_INST, BASE_RESPONSE, VICUNA_RESPONSE, INTERNLM_END
from .modeling_utils import add_vector_after_position, find_instruction_end_postion
from .modeling_utils import tokenize_llama_base, tokenize_llama_chat, tokenize_vicuna_v1_5, tokenize_internlm, tokenize_qwen

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, tokenizer, use_interlm=False):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer
        self.use_internlm = use_interlm
        
        if self.use_internlm: 
            self.block.attention = AttnWrapper(self.block.attention)
        else:
            self.block.self_attn = AttnWrapper(self.block.self_attn) 
        self.post_attention_layernorm = self.block.ffn_norm if self.use_internlm else self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.steering_vector = None
        self.after_position = None

        self.save_internal_decodings = False

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]

        if self.steering_vector is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.steering_vector,
                # position_ids: an arange from 0 to the sequence length, with batch size 1 regardless of the input batch
                position_ids=kwargs["position_ids"],
                after=self.after_position,
            )
            output = (augmented_output,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.attention.activations if self.use_internlm else self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.feed_forward(self.post_attention_layernorm(attn_output)) if self.use_internlm else self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def set_steering_vector(self, vector):
        self.steering_vector = vector

    def reset(self):
        self.steering_vector = None
        self.activations = None
        if self.use_internlm:
            self.block.attention.activations = None 
        else:
            self.block.self_attn.activations = None
        self.after_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


class LlamaWrapper:
    def __init__(
        self,
        model,
        tokenizer,
        size: str = "7b",
        use_chat: bool = True,
        use_vicuna: bool = False,
        use_internlm: bool = False,
        use_qwen: bool = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_chat = use_chat
        self.use_vicuna = use_vicuna
        self.use_internlm = use_internlm
        self.use_qwen = use_qwen
        
        self.tokenizer = tokenizer
        self.model = model
        if size != "7b":
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        if use_chat:
            if use_vicuna: # \nASSISTANT:
                self.END_STR = torch.tensor(self.tokenizer.encode(VICUNA_RESPONSE)[1:]).to(self.device)
            elif use_internlm or use_qwen: # <|im_end|> -- 92542
                self.END_STR = torch.tensor(self.tokenizer.encode(INTERNLM_END)[1:]).to(self.device)              
            else: # [/INST]
                self.END_STR = torch.tensor(self.tokenizer.encode(E_INST)[1:]).to(self.device)
        else: # \nResponse:
            self.END_STR = torch.tensor(self.tokenizer.encode(BASE_RESPONSE)[1:]).to(self.device)
        
        for i, layer in enumerate(self.model.model.layers):
            if self.use_internlm:
                self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.output, self.model.model.norm, self.tokenizer, use_interlm=use_internlm)
            else:
                self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm, self.tokenizer)

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos: int):
        for layer in self.model.model.layers:
            layer.after_position = pos

    def generate(self, tokens, max_new_tokens=100):
        with torch.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_after_positions(instr_pos)
            if self.use_internlm:
                eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids([INTERNLM_END])[0]]
            else:
                eos_token_id = [self.tokenizer.eos_token_id]
            generated = self.model.generate(inputs=tokens, max_new_tokens=max_new_tokens, top_k=1, repetition_penalty=1.1, do_sample=True, eos_token_id=eos_token_id,)
            return self.tokenizer.batch_decode(generated)[0]

    def generate_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None, max_new_tokens: int = 50) -> str:
        if self.use_chat:
            if self.use_vicuna:
                tokens = tokenize_vicuna_v1_5(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt)
            elif self.use_internlm:
                # ignore the history input
                tokens = tokenize_internlm(tokenizer=self.tokenizer, user_input=user_input)
            elif self.use_qwen:
                tokens = tokenize_qwen(tokenizer=self.tokenizer, user_input=user_input)
            else:
                tokens = tokenize_llama_chat(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt)
        else:
            tokens = tokenize_llama_base(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def get_logits(self, tokens):
        with torch.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_after_positions(instr_pos)
            logits = self.model(tokens).logits
            return logits

    def get_logits_from_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None) -> torch.Tensor:
        if self.use_chat:
            if self.use_vicuna:
                tokens = tokenize_vicuna_v1_5(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt)
            elif self.use_internlm:
                # ignore the history input
                tokens = tokenize_internlm(tokenizer=self.tokenizer, user_input=user_input)
            elif self.use_qwen:
                tokens = tokenize_qwen(tokenizer=self.tokenizer, user_input=user_input)
            else:
                tokens = tokenize_llama_chat(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt)
        else:
            tokens = tokenize_llama_base(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        return tokens, self.get_logits(tokens)

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_steering_vectors(self, layer, vector):
        self.model.model.layers[layer].set_steering_vector(vector)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

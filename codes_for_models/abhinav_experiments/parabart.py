import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartPretrainedModel as PretrainedBartModel
from typing import Dict, List, Optional, Tuple
from torch import Tensor
from transformers.activations import ACT2FN


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.
    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indicies = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indicies.long() + padding_idx


class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
            attn_mask: Optional[Tensor] = None,
            need_weights=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training, )

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def _prepare_bart_decoder_inputs(
        config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self, num_embeddings: int, embedding_dim: int, padding_idx: int,
    ):
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        assert padding_idx is not None
        num_embeddings += padding_idx + 1  # WHY?
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if use_cache:  # the position is our current step in the decoded sequence
            pos = int(self.padding_idx + input.size(1))
            positions = input.data.new(1, 1).fill_(pos)
        else:
            positions = create_position_ids_from_input_ids(input, self.padding_idx)
        return super().forward(positions)


class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            query=x, key=x, key_padding_mask=encoder_padding_mask, need_weights=self.output_attentions
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
            self,
            x,
            encoder_hidden_states,
            encoder_attn_mask=None,
            layer_state=None,
            causal_mask=None,
            decoder_padding_mask=None,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            need_weights=self.output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


LayerNorm = nn.LayerNorm


class ParaBart(PretrainedBartModel):
    def __init__(self, config):
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)

        self.encoder = ParaBartEncoder(config, self.shared)
        self.decoder = ParaBartDecoder(config, self.shared)

        self.linear = nn.Linear(config.d_model, config.vocab_size)

        self.adversary = Discriminator(config)

        self.init_weights()

    def forward(
            self,
            input_ids,
            decoder_input_ids,
            attention_mask=None,
            decoder_padding_mask=None,
            encoder_outputs=None,
            return_encoder_outputs=False,
    ):
        if attention_mask is None:
            attention_mask = input_ids == self.config.pad_token_id

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)

        if return_encoder_outputs:
            return encoder_outputs

        assert encoder_outputs is not None
        assert decoder_input_ids is not None

        decoder_input_ids = decoder_input_ids[:, :-1]

        _, decoder_padding_mask, decoder_causal_mask = _prepare_bart_decoder_inputs(
            self.config,
            input_ids=None,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_padding_mask,
            causal_mask_dtype=self.shared.weight.dtype,
        )

        attention_mask2 = torch.cat(
            (torch.zeros(input_ids.shape[0], 1).bool().cuda(), attention_mask[:, self.config.max_sent_len + 2:]), dim=1)

        # decoder
        decoder_outputs = self.decoder(
            decoder_input_ids,
            torch.cat((encoder_outputs[1], encoder_outputs[0][:, self.config.max_sent_len + 2:]), dim=1),
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=decoder_causal_mask,
            encoder_attention_mask=attention_mask2,
        )[0]

        batch_size = decoder_outputs.shape[0]
        outputs = self.linear(decoder_outputs.contiguous().view(-1, self.config.d_model))
        outputs = outputs.view(batch_size, -1, self.config.vocab_size)

        # discriminator
        for p in self.adversary.parameters():
            p.required_grad = False
        adv_outputs = self.adversary(encoder_outputs[1])

        return outputs, adv_outputs

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        encoder_outputs = past[0]
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": torch.cat(
                (decoder_input_ids, torch.zeros((decoder_input_ids.shape[0], 1), dtype=torch.long).cuda()), 1),
            "attention_mask": attention_mask,
        }

    def get_encoder(self):
        return self.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)

    def get_input_embeddings(self):
        return self.shared

    @staticmethod
    def _reorder_cache(past, beam_idx):
        enc_out = past[0][0]

        new_enc_out = enc_out.index_select(0, beam_idx)

        past = ((new_enc_out,),)
        return past

    def forward_adv(
            self,
            input_token_ids,
            attention_mask=None,
            decoder_padding_mask=None
    ):
        for p in self.adversary.parameters():
            p.required_grad = True
        sent_embeds = self.encoder.embed(input_token_ids, attention_mask=attention_mask).detach()
        adv_outputs = self.adversary(sent_embeds)

        return adv_outputs


class ParaBartEncoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.config = config

        self.dropout = config.dropout
        self.embed_tokens = embed_tokens

        self.embed_synt = nn.Embedding(77, config.d_model, config.pad_token_id)
        self.embed_synt.weight.data.normal_(mean=0.0, std=config.init_std)
        self.embed_synt.weight.data[config.pad_token_id].zero_()

        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, config.pad_token_id
        )

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.synt_layers = nn.ModuleList([EncoderLayer(config) for _ in range(1)])

        self.layernorm_embedding = LayerNorm(config.d_model)

        self.synt_layernorm_embedding = LayerNorm(config.d_model)

        self.pooling = MeanPooling(config)

    def forward(self, input_ids, attention_mask):

        input_token_ids, input_synt_ids = torch.split(input_ids,
                                                      [self.config.max_sent_len + 2, self.config.max_synt_len + 2],
                                                      dim=1)
        input_token_mask, input_synt_mask = torch.split(attention_mask,
                                                        [self.config.max_sent_len + 2, self.config.max_synt_len + 2],
                                                        dim=1)

        x = self.forward_token(input_token_ids, input_token_mask)
        y = self.forward_synt(input_synt_ids, input_synt_mask)

        encoder_outputs = torch.cat((x, y), dim=1)

        sent_embeds = self.pooling(x, input_token_ids)

        return encoder_outputs, sent_embeds

    def forward_token(self, input_token_ids, attention_mask):
        input_token_embeds = self.embed_tokens(input_token_ids) + self.embed_positions(input_token_ids)
        x = self.layernorm_embedding(input_token_embeds)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)

        for encoder_layer in self.layers:
            x, _ = encoder_layer(x, encoder_padding_mask=attention_mask)

        x = x.transpose(0, 1)
        return x

    def forward_synt(self, input_synt_ids, attention_mask):
        input_synt_embeds = self.embed_synt(input_synt_ids) + self.embed_positions(input_synt_ids)
        y = self.synt_layernorm_embedding(input_synt_embeds)
        y = F.dropout(y, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        y = y.transpose(0, 1)

        for encoder_synt_layer in self.synt_layers:
            y, _ = encoder_synt_layer(y, encoder_padding_mask=attention_mask)

        # T x B x C -> B x T x C
        y = y.transpose(0, 1)
        return y

    def embed(self, input_token_ids, attention_mask=None, pool='mean'):
        if attention_mask is None:
            attention_mask = input_token_ids == self.config.pad_token_id

        x = self.forward_token(input_token_ids, attention_mask)

        sent_embeds = self.pooling(x, input_token_ids)
        return sent_embeds


class MeanPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x, input_token_ids):
        mask = input_token_ids != self.config.pad_token_id
        mean_mask = mask.float() / mask.float().sum(1, keepdim=True)
        x = (x * mean_mask.unsqueeze(2)).sum(1, keepdim=True)
        return x


class ParaBartDecoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()

        self.dropout = config.dropout

        self.embed_tokens = embed_tokens

        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, config.pad_token_id
        )

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(1)])
        self.layernorm_embedding = LayerNorm(config.d_model)

    def forward(
            self,
            decoder_input_ids,
            encoder_hidden_states,
            decoder_padding_mask,
            decoder_causal_mask,
            encoder_attention_mask
    ):
        x = self.embed_tokens(decoder_input_ids) + self.embed_positions(decoder_input_ids)
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        for idx, decoder_layer in enumerate(self.layers):
            x, _, _ = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_attention_mask,
                decoder_padding_mask=decoder_padding_mask,
                causal_mask=decoder_causal_mask)

        x = x.transpose(0, 1)

        return x,


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sent_layernorm_embedding = LayerNorm(config.d_model, elementwise_affine=False)
        self.adv = nn.Linear(config.d_model, 74)

    def forward(self, sent_embeds):
        x = self.sent_layernorm_embedding(sent_embeds).squeeze(1)
        x = self.adv(x)
        return x

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .project4_task2_qa_data import QaBatch


def _run_lstm(sequence: torch.Tensor, mask: torch.Tensor, lstm: nn.LSTM) -> torch.Tensor:
    lengths = mask.sum(dim=1).to(torch.long).cpu()
    packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
    packed_out, _ = lstm(packed)
    output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=sequence.size(1))
    return output


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(torch.bool)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs * mask.to(probs.dtype)
    normalizer = probs.sum(dim=dim, keepdim=True).clamp_min(1e-13)
    return probs / normalizer


class HighwayNetwork(nn.Module):
    def __init__(self, size: int, num_layers: int = 2):
        super().__init__()
        self.transforms = nn.ModuleList(nn.Linear(size, size) for _ in range(num_layers))
        self.gates = nn.ModuleList(nn.Linear(size, size) for _ in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for transform, gate in zip(self.transforms, self.gates):
            transformed = torch.relu(transform(out))
            gate_values = torch.sigmoid(gate(out))
            out = gate_values * transformed + (1.0 - gate_values) * out
        return out


class BidafCore(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        self.highway = HighwayNetwork(embedding_dim, num_layers=2)
        self.contextual_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.attention_linear = nn.Linear(hidden_size * 6, 1, bias=False)
        self.modeling_lstm = nn.LSTM(
            input_size=hidden_size * 8,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.output_lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.start_linear = nn.Linear(hidden_size * 10, 1)
        self.end_linear = nn.Linear(hidden_size * 10, 1)
        self.dropout = nn.Dropout(dropout)

    def _compute_similarity(self, context_encoded: torch.Tensor, question_encoded: torch.Tensor) -> torch.Tensor:
        batch_size, c_len, hidden_dim = context_encoded.shape
        q_len = question_encoded.size(1)
        context_exp = context_encoded.unsqueeze(2).expand(batch_size, c_len, q_len, hidden_dim)
        question_exp = question_encoded.unsqueeze(1).expand(batch_size, c_len, q_len, hidden_dim)
        features = torch.cat([context_exp, question_exp, context_exp * question_exp], dim=-1)
        return self.attention_linear(features).squeeze(-1)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        question_embeddings: torch.Tensor,
        context_mask: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_encoded = _run_lstm(self.dropout(self.highway(context_embeddings)), context_mask, self.contextual_encoder)
        question_encoded = _run_lstm(self.dropout(self.highway(question_embeddings)), question_mask, self.contextual_encoder)

        similarity = self._compute_similarity(context_encoded, question_encoded)
        c2q_attention = _masked_softmax(similarity, question_mask.unsqueeze(1), dim=2)
        c2q = torch.bmm(c2q_attention, question_encoded)

        q2c_attention = _masked_softmax(similarity.max(dim=2).values, context_mask, dim=1)
        q2c = torch.bmm(q2c_attention.unsqueeze(1), context_encoded).squeeze(1)
        q2c = q2c.unsqueeze(1).expand(-1, context_encoded.size(1), -1)

        attention_flow = torch.cat(
            [
                context_encoded,
                c2q,
                context_encoded * c2q,
                context_encoded * q2c,
            ],
            dim=-1,
        )

        modeled = _run_lstm(self.dropout(attention_flow), context_mask, self.modeling_lstm)
        modeled_2 = _run_lstm(self.dropout(modeled), context_mask, self.output_lstm)

        start_logits = self.start_linear(torch.cat([attention_flow, modeled], dim=-1)).squeeze(-1)
        end_logits = self.end_linear(torch.cat([attention_flow, modeled_2], dim=-1)).squeeze(-1)

        large_negative = torch.finfo(start_logits.dtype).min
        start_logits = start_logits.masked_fill(~context_mask, large_negative)
        end_logits = end_logits.masked_fill(~context_mask, large_negative)
        return start_logits, end_logits


class GloveBidafQaModel(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,
            padding_idx=0,
        )
        self.core = BidafCore(
            embedding_dim=embedding_matrix.size(1),
            hidden_size=hidden_size,
            dropout=dropout,
        )

    def forward(self, batch: QaBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.context_token_ids is None or batch.question_token_ids is None:
            raise ValueError("GloveBidafQaModel requires token id tensors in the batch.")
        context_embeddings = self.embedding(batch.context_token_ids)
        question_embeddings = self.embedding(batch.question_token_ids)
        return self.core(context_embeddings, question_embeddings, batch.context_mask, batch.question_mask)


class BertWordEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None,
        bert_max_length: int,
        unfreeze_last_n: int = 0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        try:
            from transformers import BertModel, BertTokenizerFast
        except ImportError as exc:
            raise ImportError("transformers is required for the bert variant.") from exc

        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
        self.bert = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.bert_max_length = bert_max_length
        self.unfreeze_last_n = int(unfreeze_last_n)
        self.is_frozen = self.unfreeze_last_n == 0
        self._configure_finetuning()
        if gradient_checkpointing and not self.is_frozen and hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable()

    @property
    def hidden_size(self) -> int:
        return int(self.bert.config.hidden_size)

    def _configure_finetuning(self) -> None:
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

        if self.unfreeze_last_n < 0:
            for parameter in self.bert.parameters():
                parameter.requires_grad = True
            return

        if self.unfreeze_last_n == 0:
            self.bert.eval()
            return

        encoder_layers = list(self.bert.encoder.layer)
        layers_to_unfreeze = encoder_layers[-self.unfreeze_last_n :]
        for parameter in self.bert.embeddings.parameters():
            parameter.requires_grad = True
        for layer in layers_to_unfreeze:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        if getattr(self.bert, "pooler", None) is not None:
            for parameter in self.bert.pooler.parameters():
                parameter.requires_grad = True

    def forward(self, word_batches: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.bert.parameters()).device
        if self.is_frozen:
            self.bert.eval()
        encoding = self.tokenizer(
            word_batches,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.bert_max_length,
            return_tensors="pt",
        )
        word_id_lists = [encoding.word_ids(batch_index=batch_idx) for batch_idx in range(len(word_batches))]
        model_inputs = {key: value.to(device) for key, value in encoding.items()}

        context = torch.no_grad() if self.is_frozen else torch.enable_grad()
        with context:
            outputs = self.bert(**model_inputs).last_hidden_state

        batch_size = len(word_batches)
        max_words = max(len(words) for words in word_batches)
        pooled = outputs.new_zeros((batch_size, max_words, outputs.size(-1)))
        mask = torch.zeros(batch_size, max_words, dtype=torch.bool, device=device)

        for batch_idx, words in enumerate(word_batches):
            word_ids = word_id_lists[batch_idx]
            seen_word_ids: set[int] = set()
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx in seen_word_ids or word_idx >= len(words):
                    continue
                pooled[batch_idx, word_idx] = outputs[batch_idx, token_idx]
                mask[batch_idx, word_idx] = True
                seen_word_ids.add(word_idx)
            if len(seen_word_ids) != len(words):
                raise ValueError(
                    "BERT word pooling lost tokens after tokenization. "
                    "Reduce the context window or verify preprocessing trimming."
                )

        return pooled, mask


class FrozenBertBidafQaModel(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        cache_dir: str | None,
        projection_dim: int,
        hidden_size: int,
        dropout: float,
        bert_max_length: int,
        bert_unfreeze_last_n: int = 0,
        bert_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.word_encoder = BertWordEncoder(
            model_name=bert_model_name,
            cache_dir=cache_dir,
            bert_max_length=bert_max_length,
            unfreeze_last_n=bert_unfreeze_last_n,
            gradient_checkpointing=bert_gradient_checkpointing,
        )
        self.projection = nn.Linear(self.word_encoder.hidden_size, projection_dim)
        self.core = BidafCore(
            embedding_dim=projection_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )

    def forward(self, batch: QaBatch) -> tuple[torch.Tensor, torch.Tensor]:
        context_embeddings, context_mask = self.word_encoder(batch.context_words)
        question_embeddings, question_mask = self.word_encoder(batch.question_words)
        context_projected = self.projection(context_embeddings)
        question_projected = self.projection(question_embeddings)
        return self.core(context_projected, question_projected, context_mask, question_mask)

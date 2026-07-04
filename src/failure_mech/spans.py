"""
Token-span bookkeeping (task §4.2).

A span is a half-open token-index range ``[start, end)`` into the tokenised
prompt. Spans are computed BY CONSTRUCTION from the character offsets a fast
tokenizer returns, then VERIFIED by decode-and-compare (decode the span, strip,
compare to the intended string). Unit-tested on all four tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """Half-open token range ``[start, end)`` plus the string it should cover."""
    start: int
    end: int
    text: str

    def __len__(self) -> int:
        return self.end - self.start

    def as_tuple(self) -> tuple[int, int]:
        return (self.start, self.end)


class SpanError(RuntimeError):
    """Raised when a span cannot be located or fails decode-verification (§1.9)."""


def char_span_to_token_span(
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """Map a character range ``[char_start, char_end)`` to a token range.

    ``offsets`` is the fast-tokenizer ``offset_mapping`` (one ``(cs, ce)`` per
    token; special tokens are ``(0, 0)`` and skipped). Returns the minimal token
    range whose characters cover the requested range. Raises ``SpanError`` if no
    token overlaps the range.
    """
    tok_start = None
    tok_end = None
    for i, (cs, ce) in enumerate(offsets):
        if cs == ce:  # special / empty token
            continue
        if ce <= char_start or cs >= char_end:
            continue  # no overlap
        if tok_start is None:
            tok_start = i
        tok_end = i + 1
    if tok_start is None or tok_end is None:
        raise SpanError(
            f"No token overlaps char range [{char_start}, {char_end}); "
            "the string was not found at the expected position."
        )
    return tok_start, tok_end


def locate_value_span(
    tokenizer,
    text: str,
    input_ids,
    offsets: list[tuple[int, int]],
    value: str,
    char_start: int,
    *,
    require_clean_boundary: bool = True,
    require_leading_space_merge: bool = True,
) -> Span:
    """Locate ``value`` (known to start at ``char_start``) as a token span.

    Two boundary requirements keep every sample of a cell on the same token
    skeleton (§4.1); either violation raises ``SpanError`` so the value is
    rejected and resampled:

      * ``require_clean_boundary`` (TRAILING): the value's last token ends
        exactly at the value (the following delimiter begins a NEW token), so
        the value never absorbs the period.
      * ``require_leading_space_merge`` (LEADING): the value's first token
        begins exactly one char before (the single preceding space is merged
        into it). This is the crux of alignment — if instead the tokenizer
        emits a STANDALONE space token before some values but not others, those
        values gain a phantom +1 token that shifts every downstream position.
        Requiring the merged form makes the value footprint identical across a
        cell's samples.
    """
    char_end = char_start + len(value)
    tok_start, tok_end = char_span_to_token_span(offsets, char_start, char_end)
    if require_clean_boundary:
        last_ce = offsets[tok_end - 1][1]
        if last_ce != char_end:
            raise SpanError(
                f"Value '{value}' merges with the following character into one "
                f"token (token ends at char {last_ce}, value ends at {char_end}). "
                "Reject this value and resample (skeleton alignment, §4.1)."
            )
    if require_leading_space_merge and char_start > 0:
        first_cs = offsets[tok_start][0]
        if first_cs != char_start - 1:
            raise SpanError(
                f"Value '{value}' has an inconsistent leading boundary "
                f"(first token starts at char {first_cs}, expected {char_start - 1} "
                "= the merged preceding space). Reject and resample (§4.1)."
            )
    span = Span(tok_start, tok_end, value)
    verify_span(tokenizer, input_ids, span, value)
    return span


def verify_span(tokenizer, input_ids, span: Span, expected: str) -> None:
    """Decode ``input_ids[span]`` and confirm it matches ``expected`` (§4.2).

    Comparison strips surrounding whitespace and is exact on the visible glyphs;
    a mismatch raises ``SpanError`` (fail loudly, §1.9).
    """
    ids = list(input_ids[span.start:span.end])
    decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
    if decoded != expected.strip():
        # Fall back to a casefold/alnum comparison message for readability, but
        # still fail: the by-construction span must decode to the intended text.
        raise SpanError(
            f"Span {span.as_tuple()} decodes to {decoded!r}, expected {expected!r}."
        )

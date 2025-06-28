from typing import List

from whisperx.alignment import SingleWordSegment


class TextFormatter:
    @staticmethod
    def format_segments(grouped_word_segments: List[List[SingleWordSegment]]) -> str:
        if not grouped_word_segments:
            return ""

        final_text_segments: List[str] = []
        for single_segment_group in grouped_word_segments:
            words_in_segment = [
                str(ws.get("word", "")).strip() for ws in single_segment_group
            ]
            words_in_segment = [w for w in words_in_segment if w]

            if not words_in_segment:
                continue

            segment_sentence = " ".join(words_in_segment)

            # Capitalize first letter of the sentence if it's not empty
            if segment_sentence:
                formatted_sentence = segment_sentence[0].upper() + segment_sentence[1:]
                final_text_segments.append(formatted_sentence)

        if not final_text_segments:
            return ""

        if len(final_text_segments) == 1:
            return final_text_segments[0]

        return "\n– ".join(final_text_segments)

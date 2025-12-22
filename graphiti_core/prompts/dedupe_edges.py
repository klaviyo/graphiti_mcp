"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(
        ...,
        description='List of idx values of any duplicate facts. If no duplicate facts are found, default to empty list.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of idx values of facts that should be invalidated. If no facts should be invalidated, the list should be empty.',
    )


class Prompt(Protocol):
    resolve_edge: PromptVersion


class Versions(TypedDict):
    resolve_edge: PromptFunction


def resolve_edge(context: dict[str, Any]) -> list[Message]:
    existing_facts_count = len(context.get('existing_edges', []))
    invalidation_candidates_count = len(context.get('edge_invalidation_candidates', []))

    existing_range = f'0 to {existing_facts_count - 1}' if existing_facts_count > 0 else 'none (empty list)'
    invalidation_range = f'0 to {invalidation_candidates_count - 1}' if invalidation_candidates_count > 0 else 'none (empty list)'

    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates facts from fact lists and determines which existing '
                    'facts are contradicted by the new fact.',
        ),
        Message(
            role='user',
            content=f"""
        You will analyze a NEW FACT against two separate lists of existing facts.
        
        ═══════════════════════════════════════════════════════════════
        LIST A: EXISTING FACTS (for duplicate detection)
        ═══════════════════════════════════════════════════════════════
        Count: {existing_facts_count} facts
        Valid idx range: {existing_range}
        
        1. DUPLICATE DETECTION:
           - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
           - Facts with similar information that contain key differences should NOT be marked as duplicates.
           - Return idx values from EXISTING FACTS.
           - If no duplicates, return an empty list for duplicate_facts.

        2. FACT TYPE CLASSIFICATION:
           - Given the predefined FACT TYPES, determine if the NEW FACT should be classified as one of these types.
           - Return the fact type as fact_type or DEFAULT if NEW FACT is not one of the FACT TYPES.

        3. CONTRADICTION DETECTION:
           - Based on FACT INVALIDATION CANDIDATES and NEW FACT, determine which facts the new fact contradicts.
           - Return idx values from FACT INVALIDATION CANDIDATES.
           - If no contradictions, return an empty list for contradicted_facts.

        IMPORTANT:
        - duplicate_facts: Use ONLY 'idx' values from EXISTING FACTS
        - contradicted_facts: Use ONLY 'idx' values from FACT INVALIDATION CANDIDATES
        - These are two separate lists with independent idx ranges starting from 0

        Guidelines:
        1. Some facts may be very similar but will have key differences, particularly around numeric values in the facts.
            Do not mark these facts as duplicates.

        <FACT TYPES>
        {context['edge_types']}
        </FACT TYPES>

        <EXISTING FACTS>
        {context['existing_edges']}

        ═══════════════════════════════════════════════════════════════
        LIST B: FACT INVALIDATION CANDIDATES (for contradiction detection)
        ═══════════════════════════════════════════════════════════════
        Count: {invalidation_candidates_count} facts
        Valid idx range: {invalidation_range}

        {context['edge_invalidation_candidates']}

        ═══════════════════════════════════════════════════════════════
        NEW FACT TO ANALYZE
        ═══════════════════════════════════════════════════════════════
        {context['new_edge']}

        ═══════════════════════════════════════════════════════════════
        FACT TYPES FOR CLASSIFICATION
        ═══════════════════════════════════════════════════════════════
        {context['edge_types']}

        ═══════════════════════════════════════════════════════════════
        YOUR RESPONSE MUST INCLUDE THREE FIELDS
        ═══════════════════════════════════════════════════════════════

        1. duplicate_facts (list of integers)
           SOURCE: Use idx values ONLY from LIST A (EXISTING FACTS)
           VALID RANGE: {existing_range}
           PURPOSE: Identify which facts in LIST A are duplicates of the NEW FACT
           CRITERIA: Facts must represent identical factual information (minor wording differences OK)
           NOTE: Facts with key differences (especially numeric values) are NOT duplicates
           IF NO DUPLICATES: Return empty list []

        2. contradicted_facts (list of integers)
           SOURCE: Use idx values ONLY from LIST B (FACT INVALIDATION CANDIDATES)
           VALID RANGE: {invalidation_range}
           PURPOSE: Identify which facts in LIST B are contradicted by the NEW FACT
           CRITERIA: Facts that are logically incompatible with the NEW FACT
           IF NO CONTRADICTIONS: Return empty list []

        3. fact_type (string)
           SOURCE: Choose from FACT TYPES listed above
           PURPOSE: Classify the NEW FACT's type
           DEFAULT: Return 'DEFAULT' if NEW FACT doesn't match any predefined FACT TYPES

        ═══════════════════════════════════════════════════════════════
        CRITICAL WARNINGS
        ═══════════════════════════════════════════════════════════════
        - LIST A and LIST B are COMPLETELY SEPARATE with INDEPENDENT indexing
        - Do NOT use idx values from LIST B in duplicate_facts field
        - Do NOT use idx values from LIST A in contradicted_facts field
        - Each list starts indexing from 0 independently
        - Verify your idx values are within the valid ranges specified above
        """,
        ),
    ]


versions: Versions = {'resolve_edge': resolve_edge}

"""Markdown parser for extracting hierarchical structure from markdown documents."""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class Token:
    """Represents a parsed token from a markdown document.
    
    Attributes:
        type: The type of token ('header' or 'content').
        content: The text content of the token.
        line_number: The line number where the token appears in the source document.
        metadata: Additional metadata about the token. For headers, includes:
            - marker: The formatting marker ('#' or '*')
            - marker_count: Number of markers (1-6 for '#', 1-3 for '*')
            - case: Text case ('all_caps', 'title_case', 'sentence_case', 'all_lowercase', 'mixed_case')
            - position: Header position ('standalone' or 'inline')
    """
    type: str
    content: str
    line_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchyContext:
    """Represents hierarchical context for a token.
    
    This class encapsulates all hierarchy-related information computed
    during the hierarchy analysis phase, separate from the raw token data.
    
    Attributes:
        token: The Token this context is for.
        level: The computed hierarchical level.
        parents: list of parent header titles (from root to immediate parent).
        parent_types: list of parent header type signatures (corresponding to parents).
            Each signature is a string like "#1", "*2-CAPS", "*2-inline", etc.
    """
    token: Token
    level: int
    parents: list[str]
    parent_types: list[str]


@dataclass
class HierarchyState:
    """Tracks state during hierarchy building.
    
    This class encapsulates all the tracking variables needed to compute
    hierarchical levels for different header types.
    
    Attributes:
        all_caps_level: Fixed level for all-caps headers (set on first encounter).
        last_hash_level: Level of the most recent hash header.
        last_hash_marker_count: Marker count of the most recent hash header.
        last_asterisk_level: Level of the most recent asterisk header.
        last_asterisk_marker_count: Marker count of the most recent asterisk header.
        last_header_level: Level of the most recent header (any type, excluding inline).
        previous_header_was_hash: Whether the previous header was a hash header.
    """
    all_caps_level: Optional[int] = None
    last_hash_level: Optional[int] = None
    last_hash_marker_count: Optional[int] = None
    last_asterisk_level: Optional[int] = None
    last_asterisk_marker_count: Optional[int] = None
    last_header_level: int = 0
    previous_header_was_hash: bool = False


class MarkdownParser:
    """Parser for markdown documents that extracts hierarchical structure.
    
    This parser identifies hash headers (e.g., # Header), asterisk headers
    (e.g., **Bold Header**), inline headers with colon (e.g., **Name:** value),
    all-caps headers, and non-header content, building a hierarchical
    representation of the document structure.
    
    Attributes:
        HEADING_HASH_PATTERN: Regex pattern for hash-style headers.
        HEADING_ASTERISK_PATTERN: Regex pattern for asterisk-style headers.
        INLINE_COLON_PATTERN: Regex pattern for inline headers with colon.
    """
    HEADING_HASH_PATTERN: str = r'^(#{1,6})\s*(.*)'
    HEADING_ASTERISK_PATTERN: str = r'^(\*{1,3})\s*(.*?)\s*\1$'
    # Matches: **Label:** content or **Label**: content or *Label:* content, etc.
    INLINE_COLON_PATTERN: str = r'^(\*{1,3})\s*(.*?):\s*\1\s*(.+)$|^(\*{1,3})\s*(.*?)\s*\4:\s*(.+)$'
    
    def __init__(
        self,
        text: str,
        header_max_words: int = 10,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initializes the MarkdownParser.
        
        Args:
            text: The markdown text to parse.
            header_max_words: Maximum number of words to consider a line as a header
                (applies to both hash and asterisk headers). Defaults to 10.
            metadata: Additional metadata for the document (e.g., id, filepath, type).
                Defaults to None.
        """
        self.text: str = text
        self.max_header_words: int = header_max_words
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}
        self.tokens: list[Token] = []
        self.lines: list[str] = self.text.split('\n')
        self.length: int = len(self.lines)
        self.pos: int = 0
        self._hierarchy_context: Optional[list[HierarchyContext]] = None
    
    ### PARSER METHODS ###
    
    def _is_valid_header_length(self, content: str) -> bool:
        """Checks if header content is within the maximum word count limit.
        
        Args:
            content: The header content to check.
            
        Returns:
            True if the content has at most max_header_words words, False otherwise.
        """
        return len(content.split()) <= self.max_header_words
    
    def _is_valid_header(self, line: str) -> bool:
        """Lightweight check if a line is a valid header (pattern + word count).
        
        This method is optimized for performance in lookahead scenarios where we only
        need to know IF a line is a header, without creating Token objects or computing
        metadata like case detection.
        
        Args:
            line: The line to check.
            
        Returns:
            True if the line matches any header pattern AND passes word count validation.
        """
        # Check inline header pattern
        inline_match = re.match(self.INLINE_COLON_PATTERN, line)
        if inline_match:
            # Extract label from whichever pattern matched
            label = (inline_match.group(2) if inline_match.group(1) 
                    else inline_match.group(5)).strip()
            if self._is_valid_header_length(label):
                return True
        
        # Check hash header pattern
        hash_match = re.match(self.HEADING_HASH_PATTERN, line)
        if hash_match:
            header_content = hash_match.group(2).strip()
            if self._is_valid_header_length(header_content):
                return True
        
        # Check asterisk header pattern
        asterisk_match = re.match(self.HEADING_ASTERISK_PATTERN, line)
        if asterisk_match:
            header_content = asterisk_match.group(2).strip()
            if self._is_valid_header_length(header_content):
                return True
        
        return False
    
    @staticmethod
    def _detect_case(text: str) -> str:
        """Detects the case pattern of the given text using regex patterns.
        
        Args:
            text: The text to analyze.
            
        Returns:
            One of: 'all_caps', 'all_lowercase', 'title_case', 'sentence_case', or 'mixed_case'.
        """
        # Return mixed_case if no letters present
        if not re.search(r'[a-zA-Z]', text):
            return 'mixed_case'
        
        # All caps: all letters are uppercase
        if re.match(r'^[^a-z]*$', text) and re.search(r'[A-Z]', text):
            return 'all_caps'
        
        # All lowercase: all letters are lowercase
        if re.match(r'^[^A-Z]*$', text) and re.search(r'[a-z]', text):
            return 'all_lowercase'
        
        # Title case: each word starts with uppercase letter
        # Pattern: word boundaries followed by uppercase, rest can be lowercase/non-alpha
        if re.match(r'^(\W*[A-Z][a-z]*\W*)+$', text):
            return 'title_case'
        
        # Sentence case: starts with uppercase, rest are lowercase
        # Pattern: optional non-alpha, then uppercase, then only lowercase letters
        if re.match(r'^\W*[A-Z][a-z\W]*$', text) and not re.search(r'[A-Z]', text[1:]):
            return 'sentence_case'
        
        return 'mixed_case'
    
    @staticmethod
    def _get_header_signature(metadata: dict[str, Any]) -> str:
        """Generates a compact string signature for a header based on its metadata.
        
        Format: {marker}{count}[-CAPS][-inline]
        Examples:
            - "#1" → hash header with 1 hash
            - "#2-CAPS" → all-caps hash header with 2 hashes
            - "*2" → bold header
            - "*2-inline" → inline bold header with colon
            - "*2-CAPS" → all-caps bold header
        
        Args:
            metadata: The metadata dictionary containing marker, marker_count, case, and position.
            
        Returns:
            A compact string signature representing the header type.
        """
        marker = metadata.get('marker', '')
        count = metadata.get('marker_count', 0)
        case = metadata.get('case', '')
        position = metadata.get('position', 'standalone')
        
        signature = f"{marker}{count}"
        
        # Add modifiers for special cases
        if case == 'all_caps':
            signature += '-CAPS'
        if position == 'inline':
            signature += '-inline'
        
        return signature
    
    def _try_parse_inline_header(self, line: str, line_number: int) -> Optional[list[Token]]:
        """Attempts to parse a line as an inline header with colon.
        
        Inline headers have the format: **Label:** content or **Label**: content
        They create two tokens: a header token and a content token.
        
        Args:
            line: The line to parse.
            line_number: The line number in the source document.
            
        Returns:
            A list containing [header_token, content_token] if the line matches,
            None otherwise.
        """
        inline_match = re.match(self.INLINE_COLON_PATTERN, line)
        if not inline_match:
            return None
        
        # Pattern has two alternatives, check which matched
        if inline_match.group(1):  # **Label:** content format
            marker_count = len(inline_match.group(1))
            label = inline_match.group(2).strip()
            content = inline_match.group(3).strip()
        else:  # **Label**: content format
            marker_count = len(inline_match.group(4))
            label = inline_match.group(5).strip()
            content = inline_match.group(6).strip()
        
        # Check word count limit
        if not self._is_valid_header_length(label):
            return None
        
        case = self._detect_case(label)
        
        # Create header token
        header_token = Token(
            type='header',
            content=label,
            line_number=line_number,
            metadata={
                'marker': '*',
                'marker_count': marker_count,
                'case': case,
                'position': 'inline',
            },
        )
        
        # Create content token (always follows inline header)
        content_token = Token(
            type='content',
            content=content,
            line_number=line_number,
            metadata={},
        )
        
        return [header_token, content_token]
    
    def _try_parse_hash_header(self, line: str, line_number: int) -> Optional[Token]:
        """Attempts to parse a line as a hash header.
        
        Hash headers have the format: # Header, ## Header, etc.
        
        Args:
            line: The line to parse.
            line_number: The line number in the source document.
            
        Returns:
            A Token if the line matches a hash header, None otherwise.
        """
        hash_match = re.match(self.HEADING_HASH_PATTERN, line)
        if not hash_match:
            return None
        
        marker_count = len(hash_match.group(1))
        header_content = hash_match.group(2).strip()
        
        # Check word count limit
        if not self._is_valid_header_length(header_content):
            return None
        
        case = self._detect_case(header_content)
        
        return Token(
            type='header',
            content=header_content,
            line_number=line_number,
            metadata={
                'marker': '#',
                'marker_count': marker_count,
                'case': case,
                'position': 'standalone',
            },
        )
    
    def _try_parse_asterisk_header(self, line: str, line_number: int) -> Optional[Token]:
        """Attempts to parse a line as an asterisk header (standalone only).
        
        Asterisk headers have the format: *Header*, **Header**, ***Header***
        This method only matches standalone headers (not inline with colon).
        
        Args:
            line: The line to parse.
            line_number: The line number in the source document.
            
        Returns:
            A Token if the line matches an asterisk header, None otherwise.
        """
        asterisk_match = re.match(self.HEADING_ASTERISK_PATTERN, line)
        if not asterisk_match:
            return None
        
        marker_count = len(asterisk_match.group(1))
        header_content = asterisk_match.group(2).strip()
        
        # Check word count limit
        if not self._is_valid_header_length(header_content):
            return None
        
        case = self._detect_case(header_content)
        
        return Token(
            type='header',
            content=header_content,
            line_number=line_number,
            metadata={
                'marker': '*',
                'marker_count': marker_count,
                'case': case,
                'position': 'standalone',
            },
        )
    
    def parse(self) -> list[Token]:
        """Parses the markdown text and extracts tokens.
        
        This method processes the markdown text line by line, identifying:
        - Hash headers (e.g., # Header, ## Header)
        - Asterisk headers (e.g., **Bold Header**, *Italic Header*)
        - Inline headers with colon (e.g., **Name:** value)
        - All-caps headers (any header with all uppercase content)
        - Non-header content
        
        All headers are assigned type='header' with metadata distinguishing them:
        - marker: '#' or '*'
        - marker_count: Number of markers
        - case: 'all_caps', 'title_case', 'sentence_case', 'all_lowercase', or 'mixed_case'
        - position: 'standalone' or 'inline'
        
        Returns:
            A list of Token objects representing the parsed document structure.
        """
        while self.pos < self.length:
            line = self.lines[self.pos]
            line_number = self.pos + 1
            
            # Try to parse as inline header with colon (highest priority)
            inline_tokens = self._try_parse_inline_header(line, line_number)
            if inline_tokens:
                self.tokens.extend(inline_tokens)
                self.pos += 1
                continue
            
            # Try to parse as hash header
            hash_token = self._try_parse_hash_header(line, line_number)
            if hash_token:
                self.tokens.append(hash_token)
                self.pos += 1
                continue
            
            # Try to parse as asterisk header (standalone)
            asterisk_token = self._try_parse_asterisk_header(line, line_number)
            if asterisk_token:
                self.tokens.append(asterisk_token)
                self.pos += 1
                continue
            
            # If we reach here, the line is not a header (either doesn't match pattern
            # or matches but exceeds word count). Collect it and subsequent non-header lines.
            non_header_lines: list[str] = [line]
            self.pos += 1
            
            # Continue collecting non-header lines until we find an actual header
            while self.pos < self.length:
                line = self.lines[self.pos]
                
                # Check if the line is a valid header
                if self._is_valid_header(line):
                    break
                    
                non_header_lines.append(line)
                self.pos += 1
            
            # Create content token if we have non-header content
            non_header_content = '\n'.join(non_header_lines).strip()
            if non_header_content:
                token = Token(
                    type='content',
                    content=non_header_content,
                    line_number=line_number,
                    metadata={},
                )
                self.tokens.append(token)
        
        return self.tokens
    
    ### HIERARCHY METHODS ###
    
    def _compute_all_caps_level(
        self,
        state: HierarchyState,
        header_stack: list[tuple[int, str, dict[str, Any]]]
    ) -> int:
        """Computes level for all-caps headers.
        
        First all-caps header sets the level contextually, all subsequent
        all-caps headers use that same fixed level.
        
        Args:
            state: The current hierarchy state.
            header_stack: The current header stack.
            
        Returns:
            The computed level for the all-caps header.
        """
        if state.all_caps_level is None:
            level = 1 if not header_stack else state.last_header_level + 1
            state.all_caps_level = level
        else:
            level = state.all_caps_level
        
        # Reset asterisk state when we encounter an all-caps header
        # This ensures that the next asterisk header after an all-caps header
        # starts fresh without reference to previous asterisk levels
        state.last_asterisk_level = None
        state.last_asterisk_marker_count = None
        
        return level
    
    def _compute_inline_level(
        self,
        header_stack: list[tuple[int, str, dict[str, Any]]]
    ) -> int:
        """Computes level for inline headers.
        
        Inline headers are one level deeper than the last header on the stack.
        
        Args:
            header_stack: The current header stack.
            
        Returns:
            The computed level for the inline header.
        """
        return 1 if not header_stack else header_stack[-1][0] + 1
    
    def _compute_hash_level(
        self,
        marker_count: int,
        state: HierarchyState,
        header_stack: list[tuple[int, str, dict[str, Any]]]
    ) -> int:
        """Computes level for hash headers.
        
        Uses formula: level = last_hash_level + (marker_count - last_hash_marker_count)
        
        Args:
            marker_count: Number of hash markers in the current header.
            state: The current hierarchy state.
            header_stack: The current header stack.
            
        Returns:
            The computed level for the hash header.
        """
        if not header_stack:
            level = 1
        elif state.last_hash_level is not None and state.last_hash_marker_count is not None:
            level = state.last_hash_level + (marker_count - state.last_hash_marker_count)
        else:
            level = state.last_header_level + 1
        
        # Update state
        state.last_hash_level = level
        state.last_hash_marker_count = marker_count
        state.previous_header_was_hash = True
        
        return level
    
    def _compute_asterisk_level(
        self,
        marker_count: int,
        state: HierarchyState,
        header_stack: list[tuple[int, str, dict[str, Any]]]
    ) -> int:
        """Computes level for asterisk headers.
        
        Uses custom ordering for binary comparison: bold (2) < bold+italic (3) < italic (1)
        Level changes by +1 or -1 based on whether current header is deeper or shallower
        in the hierarchy compared to the previous asterisk header.
        
        The mapping establishes a hierarchy where:
        - 2 asterisks (bold) → order 1 (highest level in hierarchy)
        - 3 asterisks (bold+italic) → order 2 (middle level)
        - 1 asterisk (italic) → order 3 (lowest level in hierarchy)
        
        Args:
            marker_count: Number of asterisk markers in the current header.
            state: The current hierarchy state.
            header_stack: The current header stack.
            
        Returns:
            The computed level for the asterisk header.
        """
        # Asterisk count to hierarchical order mapping
        order_map: dict[int, int] = {2: 1, 3: 2, 1: 3}
        
        if not header_stack:
            level = 1
        elif state.previous_header_was_hash:
            level = state.last_header_level + 1
        elif state.last_asterisk_level is not None and state.last_asterisk_marker_count is not None:
            current_order = order_map.get(marker_count, marker_count)
            prev_order = order_map.get(state.last_asterisk_marker_count, state.last_asterisk_marker_count)
            
            # Binary comparison: only increment/decrement by 1
            if current_order > prev_order:
                # Current is deeper in hierarchy (e.g., bold -> italic)
                level = state.last_asterisk_level + 1
            elif current_order < prev_order:
                # Current is shallower in hierarchy (e.g., italic -> bold)
                level = state.last_asterisk_level - 1
            else:
                # Same level (e.g., bold -> bold)
                level = state.last_asterisk_level
        else:
            level = state.last_header_level + 1
        
        # Update state
        state.last_asterisk_level = level
        state.last_asterisk_marker_count = marker_count
        state.previous_header_was_hash = False
        
        return level
    
    def _compute_header_level(
        self,
        token: Token,
        state: HierarchyState,
        header_stack: list[tuple[int, str, dict[str, Any]]]
    ) -> int:
        """Computes the hierarchical level for a header token.
        
        Delegates to specific computation methods based on header type.
        
        Args:
            token: The header token to compute level for.
            state: The current hierarchy state.
            header_stack: The current header stack.
            
        Returns:
            The computed hierarchical level.
        """
        marker = token.metadata.get('marker', '')
        marker_count = token.metadata.get('marker_count', 0)
        case = token.metadata.get('case', '')
        position = token.metadata.get('position', 'standalone')
        
        if case == 'all_caps' and position == 'standalone':
            return self._compute_all_caps_level(state, header_stack)
        elif position == 'inline':
            return self._compute_inline_level(header_stack)
        elif marker == '#':
            return self._compute_hash_level(marker_count, state, header_stack)
        elif marker == '*':
            return self._compute_asterisk_level(marker_count, state, header_stack)
        else:
            return state.last_header_level + 1 if state.last_header_level > 0 else 1
    
    def _create_hierarchy_context(
        self,
        token: Token,
        level: int,
        header_stack: list[tuple[int, str, dict[str, Any]]]
    ) -> HierarchyContext:
        """Creates a HierarchyContext for a token.
        
        Args:
            token: The token to create context for.
            level: The hierarchical level of the token.
            header_stack: The current header stack.
            
        Returns:
            A HierarchyContext object with parent information extracted from the stack.
        """
        parents = [h[1] for h in header_stack]
        parent_types = [self._get_header_signature(h[2]) for h in header_stack]
        
        return HierarchyContext(
            token=token,
            level=level,
            parents=parents,
            parent_types=parent_types,
        )
    
    def _update_header_stack(
        self,
        header_stack: list[tuple[int, str, dict[str, Any]]],
        level: int,
        token: Token
    ) -> None:
        """Updates the header stack with a new header.
        
        Pops headers with level >= current level, then pushes the new header.
        
        Args:
            header_stack: The header stack to update.
            level: The level of the new header.
            token: The new header token.
        """
        while header_stack and header_stack[-1][0] >= level:
            header_stack.pop()
        header_stack.append((level, token.content, token.metadata))
    
    def _should_pop_inline_header(self, context_list: list[HierarchyContext]) -> bool:
        """Checks if the previous token was an inline header that should be popped.
        
        Args:
            context_list: The list of hierarchy contexts built so far.
            
        Returns:
            True if the previous token was an inline header, False otherwise.
        """
        if not context_list or len(context_list) < 2:
            return False
        
        prev_context = context_list[-2]
        return (
            prev_context.token.type == 'header' and
            prev_context.token.metadata.get('position') == 'inline'
        )
    
    def _build_hierarchy_context(self) -> list[HierarchyContext]:
        """Builds hierarchical context for all tokens.
        
        This method traverses tokens once and computes complete hierarchical information:
        - Assigns levels to all tokens based on header hierarchy
        - Tracks parent headers and their metadata
        - Maintains a stack to track the current position in the hierarchy
        
        The level assignment follows these rules:
        - First header gets level 1
        - All-caps headers: First encounter sets the level contextually, all subsequent
          all-caps headers use that same fixed level
        - Inline headers: Treated as one level above their content (content is always leaf)
        - For other headers:
          - Hash headers: level = last_hash_level + (marker_count - last_hash_marker_count)
          - Asterisk headers: Use asterisk ordering (bold < bold+italic < italic)
        - Content tokens: level = last_header_level + 1 (or 1 if no headers)
        
        Returns:
            A list of HierarchyContext objects, one for each token.
        """
        if not self.tokens:
            return []
        
        context_list: list[HierarchyContext] = []
        header_stack: list[tuple[int, str, dict[str, Any]]] = []
        state = HierarchyState()
        
        for token in self.tokens:
            if token.type == 'header':
                # Compute level for this header
                level = self._compute_header_level(token, state, header_stack)
                
                # Create context
                context = self._create_hierarchy_context(token, level, header_stack)
                context_list.append(context)
                
                # Update header stack
                self._update_header_stack(header_stack, level, token)
                
                # Update state (but NOT for inline headers - they don't affect hierarchy)
                if token.metadata.get('position') != 'inline':
                    state.last_header_level = level
                
            elif token.type == 'content':
                # Calculate level: Use the level of the last header on stack + 1
                # (The stack includes inline headers, while last_header_level doesn't)
                level = header_stack[-1][0] + 1 if header_stack else 1
                
                # Create context
                context = self._create_hierarchy_context(token, level, header_stack)
                context_list.append(context)
                
                # If the previous token was an inline header, pop it from stack
                # since content after inline headers is always a leaf
                if self._should_pop_inline_header(context_list):
                    header_stack.pop()
        
        return context_list
    
    def _get_or_build_context(self) -> list[HierarchyContext]:
        """Gets or builds (and caches) the hierarchy context.
        
        Returns:
            The cached or newly built hierarchy context list.
        """
        if self._hierarchy_context is None:
            self._hierarchy_context = self._build_hierarchy_context()
        return self._hierarchy_context
    
    def to_hierarchical_dict(self) -> dict[str, Any]:
        """Converts parsed tokens into a hierarchical dictionary structure.
        
        This method builds a nested dictionary representation where each header
        contains its content and child sections, preserving the document hierarchy.
        
        Returns:
            A hierarchical dictionary representation where each header contains
            its content and child sections. Returns an empty dict if no tokens exist.
        """
        if not self.tokens:
            return {}
        
        context_list = self._get_or_build_context()
        
        root: dict[str, Any] = {**self.metadata, 'sections': []}
        
        # Each stack entry: (level, section_dict)
        stack: list[tuple[int, dict[str, Any]]] = [(0, root)]
        
        # For each token, place it in the correct parent by level
        for ctx in context_list:
            token = ctx.token
            
            if token.type == 'header':
                section = {
                    'type': token.type,
                    'text': token.content,
                    'level': ctx.level,
                    'line_number': token.line_number,
                    'metadata': token.metadata,
                    'sections': []
                }
                
                # Find parent: pop until parent level < current level
                # But always keep at least the root in the stack
                while len(stack) > 1 and stack[-1][0] >= ctx.level:
                    stack.pop()
                
                parent = stack[-1][1]
                parent['sections'].append(section)
                stack.append((ctx.level, section))
            
            elif token.type == 'content':
                # Find parent: pop until parent level < current level
                # But always keep at least the root in the stack
                while len(stack) > 1 and stack[-1][0] >= ctx.level:
                    stack.pop()
                
                parent = stack[-1][1]
                
                content_item = {
                    'type': 'content',
                    'text': token.content,
                    'level': ctx.level,
                    'line_number': token.line_number,
                }
                
                parent['sections'].append(content_item)
        
        return root
    
    ### OUTPUT METHODS ###
    
    def export_to_json(
        self,
        filepath: str,
        indent: int = 2,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Exports the parsed tokens to a JSON file with hierarchical structure.
        
        Args:
            filepath: Path to the output JSON file.
            indent: Indentation level for pretty printing. Defaults to 2.
            metadata: Optional metadata dictionary to add at the root level of the JSON.
                If provided, these key-value pairs will be added alongside the 'sections'
                key. Defaults to None.
        
        Returns:
            Path to the created file.
        """
        hierarchical_data = self.to_hierarchical_dict()
        
        # Add metadata to root level if provided
        if metadata:
            # Insert metadata before 'sections' key for better readability
            output_data: dict[str, Any] = {}
            for key, value in metadata.items():
                output_data[key] = value
            # Add all other keys from hierarchical_data
            for key, value in hierarchical_data.items():
                output_data[key] = value
        else:
            output_data = hierarchical_data
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=indent, ensure_ascii=False)
        
        return filepath
    
    def print_tree(
        self,
        show_line_numbers: bool = True,
        show_type: bool = True,
        return_string: bool = False,
        metadata_header: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Prints the header structure as an ASCII tree.
        
        This method displays the document's hierarchical structure using
        box-drawing characters to create a visual tree representation.
        
        Args:
            show_line_numbers: Whether to show line numbers. Defaults to True.
            show_type: Whether to show header type indicators. Defaults to True.
            return_string: If True, returns the tree as a string instead of printing.
                Defaults to False.
            metadata_header: Optional dictionary of metadata to display at the top
                of the tree output. Defaults to None.
        
        Returns:
            The tree structure as a string if return_string is True, None otherwise.
        """
        # Get hierarchy context
        context_list = self._get_or_build_context()
        
        # Filter to only header contexts
        header_contexts = [
            ctx for ctx in context_list
            if ctx.token.type == 'header'
        ]
        
        lines: list[str] = []
        
        # Add metadata header if provided
        if metadata_header:
            lines.append("Metadata")
            lines.append("-" * 80)
            for key, value in metadata_header.items():
                lines.append(f"{key}: {value}")
            lines.append("")
        
        if not header_contexts:
            lines.append("No headers found")
            if return_string:
                return '\n'.join(lines)
            else:
                print('\n'.join(lines))
                return None
        
        lines.append("Document Structure")
        lines.append("=" * 80)
        
        # Track the levels and whether they have more siblings
        level_has_more: dict[int, bool] = {}
        
        for i, ctx in enumerate(header_contexts):
            header = ctx.token
            
            # Determine if this header has siblings after it at the same level
            has_more_siblings = False
            for j in range(i + 1, len(header_contexts)):
                if header_contexts[j].level < ctx.level:
                    break
                if header_contexts[j].level == ctx.level:
                    has_more_siblings = True
                    break
            
            # Update the level_has_more tracking
            level_has_more[ctx.level] = has_more_siblings
            
            # Build the tree prefix
            prefix = ""
            for level in range(1, ctx.level):
                if level in level_has_more and level_has_more[level]:
                    prefix += "│   "
                else:
                    prefix += "    "
            
            # Add the branch character
            if ctx.level > 1:
                if has_more_siblings:
                    prefix += "├── "
                else:
                    prefix += "└── "
            
            # Build the label
            label = header.content
            
            # Add type indicator
            if show_type:
                type_signature = self._get_header_signature(header.metadata)
                label += f" [{type_signature}]"
            
            # Add line number
            if show_line_numbers:
                label += f" (line {header.line_number})"
            
            lines.append(f"{prefix}{label}")
        
        if return_string:
            return '\n'.join(lines)
        else:
            print('\n'.join(lines))
            return None
    
    def extract_non_header_rows(self) -> list[dict[str, Any]]:
        """Extracts content tokens with their hierarchical context.
        
        For each content token, this method computes:
        - start_line: The line number where the token starts
        - level: The hierarchical level of the content
        - length: Number of characters in the token content
        - parents: Ordered list of parent header titles (from root to immediate parent)
        - parent_types: Ordered list of parent header type signatures (corresponding to parents)
        - content: The actual text content of the token
        
        Returns:
            A list of dictionaries, one for each content token, containing the fields above.
        """
        # Get hierarchy context
        context_list = self._get_or_build_context()
        
        rows: list[dict[str, Any]] = []
        
        for ctx in context_list:
            if ctx.token.type == 'content':
                # Create row dictionary from context
                row = {
                    'start_line': ctx.token.line_number,
                    'level': ctx.level,
                    'length': len(ctx.token.content),
                    'parents': ctx.parents,
                    'parent_types': ctx.parent_types,
                    'content': ctx.token.content,
                }
                
                rows.append(row)
        
        return rows

### BATCH PROCESSING FUNCTION ###

def batch_process_markdown_dataframe(
    df: 'pd.DataFrame',
    content_column: str = 'content',
    id_column: Optional[str] = None,
    metadata_columns: Optional[list[str]] = None,
    header_max_words: int = 10,
    export_tree: bool = False,
    export_json: bool = False,
    export_parquet: bool = False,
    output_dir: Optional[str] = None,
) -> 'pd.DataFrame':
    """Batch processes a dataframe containing markdown documents.
    
    This function processes each row in the dataframe, parses the markdown content,
    and creates a new dataframe where each row represents a non-header token with
    its hierarchical context.
    
    The output dataframe has the following structure:
    - id: Document identifier (from id_column if provided, otherwise a hash of the content)
    - metadata_columns: list of additional columns from the input dataframe (if specified)
    - start_line: Line number where the token starts
    - level: Hierarchical level (last header level + 1, or 1 if no headers)
    - length: Number of characters in the token content
    - parents: list of parent header titles (from root to immediate parent)
    - parent_types: list of parent header types (corresponding to parents)
    - content: The actual text content of the token
    
    Args:
        df: Input dataframe with markdown content.
        content_column: Name of the column containing markdown text. Defaults to 'content'.
        id_column: Name of the column to use as document ID. If None, a hash of the
            content will be generated. Defaults to None.
        metadata_columns: list of additional column names to include in the output.
            These columns will be copied from the input dataframe and placed after
            the id column. Defaults to None.
        header_max_words: Maximum number of words to consider a line as a header
            (applies to both hash and asterisk headers). Defaults to 10.
        export_tree: If True, exports the tree structure of each document to individual
            text files in output_dir/tree/. Defaults to False.
        export_json: If True, exports the hierarchical structure of each document to
            individual JSON files in output_dir/json/. Defaults to False.
        export_parquet: If True, exports the parsed dataframe to a parquet file in
            output_dir/parsed_data.parquet. Defaults to False.
        output_dir: Directory path where exported files will be saved. Required if
            export_tree, export_json, or export_parquet is True. Defaults to None.
    
    Returns:
        A new dataframe where each row represents a non-header token with its
        hierarchical context. Multiple rows will have the same ID if they come
        from the same source document.
    
    Raises:
        ValueError: If the specified content_column, id_column, or any metadata_columns
            don't exist in the dataframe, or if export_tree/export_json/export_parquet
            is True but output_dir is not provided.
    """    
    if content_column not in df.columns:
        raise ValueError(
            f"Column '{content_column}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )
    
    if id_column is not None and id_column not in df.columns:
        raise ValueError(
            f"Column '{id_column}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )
    
    if metadata_columns is not None:
        missing_columns = [col for col in metadata_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Metadata columns {missing_columns} not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )
    
    if (export_tree or export_json or export_parquet) and output_dir is None:
        raise ValueError(
            "output_dir must be provided when export_tree, export_json, or export_parquet is True"
        )
    
    # Create output directories if needed
    tree_dir = None
    json_dir = None
    if output_dir is not None:
        if export_tree:
            tree_dir = os.path.join(output_dir, 'tree')
            os.makedirs(tree_dir, exist_ok=True)
        if export_json:
            json_dir = os.path.join(output_dir, 'json')
            os.makedirs(json_dir, exist_ok=True)
    
    # Process each row
    all_rows: list[dict[str, Any]] = []
    
    for idx, row in df.iterrows():
        # Get or generate document ID
        if id_column is not None:
            doc_id = row[id_column]
        else:
            # Generate hash from content
            content = row[content_column]
            doc_id = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Extract metadata values
        metadata_values = {}
        if metadata_columns is not None:
            for col in metadata_columns:
                metadata_values[col] = row[col]
        
        # Parse the markdown content
        parser = MarkdownParser(
            text=row[content_column],
            header_max_words=header_max_words,
        )
        parser.parse()
        
        # Export tree structure if requested
        if export_tree and tree_dir is not None:
            # Create metadata header for the tree output
            tree_metadata = {'id': doc_id}
            if metadata_columns is not None:
                for col in metadata_columns:
                    tree_metadata[col] = row[col]
            
            # Get tree structure as string
            tree_str = parser.print_tree(
                show_line_numbers=True,
                show_type=True,
                return_string=True,
                metadata_header=tree_metadata,
            )
            
            # Write to file using doc_id as filename
            if tree_str is not None:
                tree_filename = f"{doc_id}.txt"
                tree_filepath = os.path.join(tree_dir, tree_filename)
                with open(tree_filepath, 'w', encoding='utf-8') as f:
                    f.write(tree_str)
        
        # Export JSON structure if requested
        if export_json and json_dir is not None:
            # Create metadata for JSON export
            json_metadata = {'id': doc_id}
            if metadata_columns is not None:
                for col in metadata_columns:
                    json_metadata[col] = row[col]
            
            # Export to JSON file
            json_filename = f"{doc_id}.json"
            json_filepath = os.path.join(json_dir, json_filename)
            parser.export_to_json(
                filepath=json_filepath,
                indent=2,
                metadata=json_metadata,
            )
        
        # Extract non-header rows
        non_header_rows = parser.extract_non_header_rows()
        
        # Add document ID and metadata to each row
        for nh_row in non_header_rows:
            nh_row['id'] = doc_id
            # Add metadata columns
            for col, value in metadata_values.items():
                nh_row[col] = value
            all_rows.append(nh_row)
    
    # Convert to dataframe with desired column order
    if all_rows:
        result_df = pd.DataFrame(all_rows)
        # Build column order: id, metadata_columns, then the rest
        column_order = ['id']
        if metadata_columns is not None:
            column_order.extend(metadata_columns)
        column_order.extend(['start_line', 'level', 'length', 'parents', 'parent_types', 'content'])
        result_df = result_df[column_order]
    else:
        # Create empty dataframe with correct schema
        base_columns = ['id']
        if metadata_columns is not None:
            base_columns.extend(metadata_columns)
        base_columns.extend(['start_line', 'level', 'length', 'parents', 'parent_types', 'content'])
        result_df = pd.DataFrame(columns=base_columns)
    
    # Export to parquet if requested
    if export_parquet and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        parquet_filepath = os.path.join(output_dir, 'parsed_data.parquet')
        result_df.to_parquet(parquet_filepath, index=False)
    
    return result_df

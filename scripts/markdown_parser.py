"""Markdown parser for extracting hierarchical structure from markdown documents."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Token:
    """Represents a parsed token from a markdown document.
    
    Attributes:
        type: The type of token ('hash_header', 'asterisk_header', or 'non_header').
        content: The text content of the token.
        line_number: The line number where the token appears in the source document.
        metadata: Additional metadata about the token (e.g., number of hashes/asterisks).
    """
    type: str
    content: str
    line_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchyContext:
    """Represents hierarchical context for a token.
    
    This class encapsulates all hierarchy-related information computed
    during the hierarchy analysis phase, separate from the raw token data.
    
    Attributes:
        token: The Token this context is for.
        level: The computed hierarchical level.
        parents: List of parent header titles (from root to immediate parent).
        parent_types: List of parent header types (corresponding to parents).
    """
    token: Token
    level: int
    parents: List[str]
    parent_types: List[str]


class MarkdownParser:
    """Parser for markdown documents that extracts hierarchical structure.
    
    This parser identifies hash headers (e.g., # Header), asterisk headers
    (e.g., **Bold Header**), and non-header content, building a hierarchical
    representation of the document structure.
    
    Attributes:
        HEADING_HASH_PATTERN: Regex pattern for hash-style headers.
        HEADING_ASTERISK_PATTERN: Regex pattern for asterisk-style headers.
    """
    HEADING_HASH_PATTERN: str = r'^(#{1,6})\s*(.*)'
    HEADING_ASTERISK_PATTERN: str = r'^(\*{1,3})\s*(.*?)\s*\1$'
    
    def __init__(
        self,
        text: str,
        asterisk_header_max_words: int = 7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the MarkdownParser.
        
        Args:
            text: The markdown text to parse.
            asterisk_header_max_words: Maximum number of words to consider a line
                with asterisks as a header. Defaults to 7.
            metadata: Additional metadata for the document (e.g., id, filepath, type).
                Defaults to None.
        """
        self.text: str = text
        self.max_asterisk_words: int = asterisk_header_max_words
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.tokens: List[Token] = []
        self.lines: List[str] = self.text.split('\n')
        self.length: int = len(self.lines)
        self.pos: int = 0
        self._hierarchy_context: Optional[List[HierarchyContext]] = None
    
    def _is_asterisk_header(self, line: str) -> bool:
        """Checks if a line is an asterisk header based on pattern and word count.
        
        Args:
            line: The line to check.
            
        Returns:
            True if the line matches the asterisk pattern and has at most
            max_asterisk_words words, False otherwise.
        """
        match = re.match(self.HEADING_ASTERISK_PATTERN, line)
        if match:
            content = match.group(2).strip()
            return len(content.split()) <= self.max_asterisk_words
        return False
       
    @staticmethod
    def _asterisk_order(num_asterisks: int) -> int:
        """Converts asterisk count to a comparable hierarchical order value.
        
        The mapping establishes a hierarchy where:
        - 2 asterisks (bold) → order 1 (highest level in hierarchy)
        - 3 asterisks (bold+italic) → order 2 (middle level)
        - 1 asterisk (italic) → order 3 (lowest level in hierarchy)
        
        Args:
            num_asterisks: The number of asterisks in the header.
            
        Returns:
            The hierarchical order value, where lower values indicate higher hierarchy.
        """
        order_map: Dict[int, int] = {2: 1, 3: 2, 1: 3}
        return order_map.get(num_asterisks, num_asterisks)
    
    def parse(self) -> List[Token]:
        """Parses the markdown text and extracts tokens.
        
        This method processes the markdown text line by line, identifying
        hash headers, asterisk headers, and non-header content.
        
        Returns:
            A list of Token objects representing the parsed document structure.
        """
        while self.pos < self.length:
            line = self.lines[self.pos]
            line_number = self.pos + 1
            
            # Check for hash headers
            hash_heading_match = re.match(self.HEADING_HASH_PATTERN, line)
            if hash_heading_match:
                non_header_content = hash_heading_match.group(2).strip()
                token = Token(
                    type='hash_header',
                    content=non_header_content,
                    line_number=line_number,
                    metadata={'num_hashes': len(hash_heading_match.group(1))},
                )
                self.tokens.append(token)
                self.pos += 1
                continue
            
            # Check for asterisk headers
            if self._is_asterisk_header(line):
                asterisk_match = re.match(self.HEADING_ASTERISK_PATTERN, line)
                assert asterisk_match is not None  # Guaranteed by _is_asterisk_header
                non_header_content = asterisk_match.group(2).strip()
                token = Token(
                    type='asterisk_header',
                    content=non_header_content,
                    line_number=line_number,
                    metadata={'num_asterisks': len(asterisk_match.group(1))},
                )
                self.tokens.append(token)
                self.pos += 1
                continue
            
            # Collect non-header lines until the next header
            non_header_lines: List[str] = []
            while self.pos < self.length:
                line = self.lines[self.pos]
                
                if re.match(self.HEADING_HASH_PATTERN, line) or self._is_asterisk_header(line):
                    break
                non_header_lines.append(line)
                self.pos += 1
            
            non_header_content = '\n'.join(non_header_lines).strip()
            if non_header_content:
                token = Token(
                    type='non_header',
                    content=non_header_content,
                    line_number=line_number,
                )
                self.tokens.append(token)
        
        return self.tokens
    
    def _build_hierarchy_context(self) -> List[HierarchyContext]:
        """Builds hierarchical context for all tokens.
        
        This method traverses tokens once and computes complete hierarchical information:
        - Assigns levels to all tokens based on header hierarchy
        - Tracks parent headers and their metadata
        - Maintains a stack to track the current position in the hierarchy
        
        The level assignment follows these rules:
        - First header gets level 1
        - For subsequent headers:
          - Hash headers:
            - If there is a previous hash header:
              level = last_hash_level + (num_hashes - last_hash_num_hashes)
            - If no previous hash header: level = last_header_level + 1
          - Asterisk headers:
            - If previous header is hash: level = previous_level + 1
            - If previous header is asterisk:
              level = prev_asterisk_level + (asterisk_order_current - asterisk_order_prev)
              where asterisk_order follows: 2 (bold) < 3 (bold+italic) < 1 (italic)
        - Non-header content: level = last_header_level + 1 (or 1 if no headers)
        
        Returns:
            A list of HierarchyContext objects, one for each token.
        """
        if not self.tokens:
            return []
        
        context_list: List[HierarchyContext] = []
        
        # Stack to track current header hierarchy
        # Each element is (level, title, type)
        header_stack: List[tuple[int, str, str]] = []
        
        # Track last hash header info for level computation
        last_hash_level: Optional[int] = None
        last_hash_num_hashes: Optional[int] = None
        
        # Track last asterisk header info for level computation
        last_asterisk_level: Optional[int] = None
        last_asterisk_num_asterisks: Optional[int] = None
        
        # Track last header level seen (for any header type)
        last_header_level: int = 0
        previous_header_type: Optional[str] = None
        
        for token in self.tokens:
            if token.type in ['hash_header', 'asterisk_header']:
                # Compute level for this header
                level: int
                if not header_stack:
                    # First header gets level 1
                    level = 1
                elif token.type == 'hash_header':
                    num_hashes: int = token.metadata['num_hashes']
                    
                    if last_hash_level is not None and last_hash_num_hashes is not None:
                        # Use formula with last hash header
                        level = last_hash_level + (num_hashes - last_hash_num_hashes)
                    else:
                        # No previous hash header, use last header level + 1
                        level = last_header_level + 1
                    
                    # Update last hash header info
                    last_hash_level = level
                    last_hash_num_hashes = num_hashes
                    
                else:  # token.type == 'asterisk_header'
                    num_asterisks: int = token.metadata['num_asterisks']
                    
                    if previous_header_type == 'hash_header':
                        # Previous is hash, assign previous level + 1
                        level = last_header_level + 1
                    elif last_asterisk_level is not None and last_asterisk_num_asterisks is not None:
                        # Previous is asterisk, use formula with custom ordering
                        current_order: int = self._asterisk_order(num_asterisks)
                        prev_order: int = self._asterisk_order(last_asterisk_num_asterisks)
                        
                        level = last_asterisk_level + (current_order - prev_order)
                    else:
                        # First asterisk header after no headers, use last header level + 1
                        level = last_header_level + 1
                    
                    # Update last asterisk header info
                    last_asterisk_level = level
                    last_asterisk_num_asterisks = num_asterisks
                
                # Update header stack - pop headers with level >= current level
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                
                # Extract parent information from stack
                parents = [h[1] for h in header_stack]
                parent_types = [h[2] for h in header_stack]
                
                # Push current header onto stack
                header_stack.append((level, token.content, token.type))
                
                # Create context for this header
                context = HierarchyContext(
                    token=token,
                    level=level,
                    parents=parents,
                    parent_types=parent_types,
                )
                context_list.append(context)
                
                # Update tracking variables
                last_header_level = level
                previous_header_type = token.type
                
            elif token.type == 'non_header':
                # Calculate level: last header level + 1, or 1 if no headers
                level = last_header_level + 1 if last_header_level > 0 else 1
                
                # Extract parent information from stack
                parents = [h[1] for h in header_stack]
                parent_types = [h[2] for h in header_stack]
                
                # Create context for this content
                context = HierarchyContext(
                    token=token,
                    level=level,
                    parents=parents,
                    parent_types=parent_types,
                )
                context_list.append(context)
        
        return context_list
    
    def _get_or_build_context(self) -> List[HierarchyContext]:
        """Gets or builds (and caches) the hierarchy context.
        
        Returns:
            The cached or newly built hierarchy context list.
        """
        if self._hierarchy_context is None:
            self._hierarchy_context = self._build_hierarchy_context()
        return self._hierarchy_context
    
    def to_hierarchical_dict(self) -> Dict[str, Any]:
        """Converts parsed tokens into a hierarchical dictionary structure.
        
        This method builds a nested dictionary representation where each header
        contains its content and child sections, preserving the document hierarchy.
        
        Returns:
            A hierarchical dictionary representation where each header contains
            its content and child sections. Returns an empty dict if no tokens exist.
        """
        if not self.tokens:
            return {}
        
        # Get hierarchy context
        context_list = self._get_or_build_context()
        
        root: Dict[str, Any] = {
            **self.metadata,
            'sections': [],
        }
        
        # Stack to keep track of the current hierarchy
        # Each element is a tuple: (level, section_dict)
        stack: List[tuple[int, Dict[str, Any]]] = [(0, root)]
        current_section: Optional[Dict[str, Any]] = None
        
        for ctx in context_list:
            token = ctx.token
            
            if token.type in ['hash_header', 'asterisk_header']:
                # Create new section
                section: Dict[str, Any] = {
                    'type': token.type,
                    'title': token.content,
                    'level': ctx.level,
                    'line_number': token.line_number,
                    'metadata': token.metadata,
                    'content': [],
                    'sections': []
                }
                
                # Pop from stack until we find the right parent
                while len(stack) > 1 and stack[-1][0] >= ctx.level:
                    stack.pop()
                
                # Add section to parent
                parent: Dict[str, Any] = stack[-1][1]
                parent['sections'].append(section)
                
                # Push current section onto stack
                stack.append((ctx.level, section))
                current_section = section
                
            elif token.type == 'non_header':
                # Add content to the current section
                content_item: Dict[str, Any] = {
                    'type': 'content',
                    'text': token.content,
                    'line_number': token.line_number,
                    'level': ctx.level,
                }
                
                if current_section is not None:
                    current_section['content'].append(content_item)
                else:
                    # Content before any header goes to root
                    root['sections'].append(content_item)
        
        return root
    
    def export_to_json(self, filepath: str, indent: int = 2) -> str:
        """Exports the parsed tokens to a JSON file with hierarchical structure.
        
        Args:
            filepath: Path to the output JSON file.
            indent: Indentation level for pretty printing. Defaults to 2.
        
        Returns:
            Path to the created file.
        """
        hierarchical_data = self.to_hierarchical_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(hierarchical_data, f, indent=indent, ensure_ascii=False)
        
        return filepath
    
    def print_tree(self, show_line_numbers: bool = True, show_type: bool = True) -> None:
        """Prints the header structure as an ASCII tree.
        
        This method displays the document's hierarchical structure using
        box-drawing characters to create a visual tree representation.
        
        Args:
            show_line_numbers: Whether to show line numbers. Defaults to True.
            show_type: Whether to show header type indicators. Defaults to True.
        """
        # Get hierarchy context
        context_list = self._get_or_build_context()
        
        # Filter to only header contexts
        header_contexts = [
            ctx for ctx in context_list
            if ctx.token.type in ['hash_header', 'asterisk_header']
        ]
        
        if not header_contexts:
            print("No headers found")
            return
        
        print("Document Structure")
        print("=" * 80)
        
        # Track the levels and whether they have more siblings
        level_has_more: Dict[int, bool] = {}
        
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
                type_indicator = ""
                if header.type == 'hash_header':
                    type_indicator = f" [#{header.metadata['num_hashes']}]"
                elif header.type == 'asterisk_header':
                    type_indicator = f" [*{header.metadata['num_asterisks']}]"
                label += type_indicator
            
            # Add line number
            if show_line_numbers:
                label += f" (line {header.line_number})"
            
            print(f"{prefix}{label}")
    
    def extract_non_header_rows(self) -> List[Dict[str, Any]]:
        """Extracts non-header tokens with their hierarchical context.
        
        For each non-header token, this method computes:
        - start_line: The line number where the token starts
        - level: The level of the last header before this token + 1 (or 1 if no headers)
        - length: Number of characters in the token content
        - parents: Ordered list of parent header titles (from root to immediate parent)
        - parent_types: Ordered list of parent header types (corresponding to parents)
        - content: The actual text content of the token
        
        Returns:
            A list of dictionaries, one for each non-header token, containing the fields above.
        """
        # Get hierarchy context
        context_list = self._get_or_build_context()
        
        rows: List[Dict[str, Any]] = []
        
        for ctx in context_list:
            if ctx.token.type == 'non_header':
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


def batch_process_markdown_dataframe(
    df: 'pd.DataFrame',
    content_column: str = 'content',
    id_column: Optional[str] = None,
    metadata_columns: Optional[List[str]] = None,
    asterisk_header_max_words: int = 7,
) -> 'pd.DataFrame':
    """Batch processes a dataframe containing markdown documents.
    
    This function processes each row in the dataframe, parses the markdown content,
    and creates a new dataframe where each row represents a non-header token with
    its hierarchical context.
    
    The output dataframe has the following structure:
    - id: Document identifier (from id_column if provided, otherwise a hash of the content)
    - metadata_columns: List of additional columns from the input dataframe (if specified)
    - start_line: Line number where the token starts
    - level: Hierarchical level (last header level + 1, or 1 if no headers)
    - length: Number of characters in the token content
    - parents: List of parent header titles (from root to immediate parent)
    - parent_types: List of parent header types (corresponding to parents)
    - content: The actual text content of the token
    
    Args:
        df: Input dataframe with markdown content.
        content_column: Name of the column containing markdown text. Defaults to 'content'.
        id_column: Name of the column to use as document ID. If None, a hash of the
            content will be generated. Defaults to None.
        metadata_columns: List of additional column names to include in the output.
            These columns will be copied from the input dataframe and placed after
            the id column. Defaults to None.
        asterisk_header_max_words: Maximum words for asterisk headers. Defaults to 7.
    
    Returns:
        A new dataframe where each row represents a non-header token with its
        hierarchical context. Multiple rows will have the same ID if they come
        from the same source document.
    
    Raises:
        ValueError: If the specified content_column, id_column, or any metadata_columns
            don't exist in the dataframe.
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
    
    # Process each row
    all_rows: List[Dict[str, Any]] = []
    
    for _, row in df.iterrows():
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
            asterisk_header_max_words=asterisk_header_max_words,
        )
        parser.parse()
        
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
    
    return result_df

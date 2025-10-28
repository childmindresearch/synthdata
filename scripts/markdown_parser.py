"""Markdown parser for extracting hierarchical structure from markdown documents."""

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd


class Token:
    """Represents a parsed token from a markdown document.
    
    Attributes:
        type: The type of token ('hash_header', 'asterisk_header', or 'non_header').
        content: The text content of the token.
        line_number: The line number where the token appears in the source document.
        metadata: Additional metadata about the token (e.g., number of hashes/asterisks).
        level: The hierarchical level of the token (for headers).
    """
    
    def __init__(
        self,
        type_: str,
        content: str,
        line_number: int,
        metadata: Optional[Dict[str, Any]] = None,
        level: int = -1,
    ) -> None:
        """Initializes a Token instance.
        
        Args:
            type_: The type of token.
            content: The text content of the token.
            line_number: The line number in the source document.
            metadata: Additional metadata. Defaults to None.
            level: The hierarchical level. Defaults to -1.
        """
        self.type: str = type_
        self.content: str = content
        self.line_number: int = line_number
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.level: int = level

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
    
    def parse(self) -> List[Token]:
        """Parses the markdown text and extracts tokens.
        
        This method processes the markdown text line by line, identifying
        hash headers, asterisk headers, and non-header content. After parsing,
        it assigns hierarchical levels to all headers.
        
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
                    type_='hash_header',
                    content=non_header_content,
                    line_number=line_number,
                    metadata={'num_hashes': len(hash_heading_match.group(1))},
                )
                self.tokens.append(token)
                self.pos += 1
                continue
            
            # Check for asterisk headers
            asterisk_match = re.match(self.HEADING_ASTERISK_PATTERN, line)
            if asterisk_match:
                non_header_content = asterisk_match.group(2).strip()
                if len(non_header_content.split()) <= self.max_asterisk_words:
                    token = Token(
                        type_='asterisk_header',
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
                asterisk_match = re.match(self.HEADING_ASTERISK_PATTERN, line)
                is_asterisk_header = (
                    asterisk_match is not None
                    and len(asterisk_match.group(2).strip().split()) <= self.max_asterisk_words
                )
                
                if re.match(self.HEADING_HASH_PATTERN, line) or is_asterisk_header:
                    break
                non_header_lines.append(line)
                self.pos += 1
            
            non_header_content = '\n'.join(non_header_lines).strip()
            if non_header_content:
                token = Token(
                    type_='non_header',
                    content=non_header_content,
                    line_number=line_number,
                )
                self.tokens.append(token)
        
        # After parsing all tokens, assign hierarchical levels to headers
        self._assign_levels()
        
        return self.tokens
    
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
    
    def _assign_levels(self) -> None:
        """Assigns hierarchical levels to header tokens retrospectively.
        
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
              where asterisk_order follows number of asterisks: 
              2 (bold) < 3 (bold+italic) < 1 (italic)
        """
        # Get all header tokens
        headers: List[Token] = [
            token for token in self.tokens
            if token.type in ['hash_header', 'asterisk_header']
        ]
        
        if not headers:
            return
        
        # Assign level 1 to first header
        headers[0].level = 1
        
        # Track last hash header info
        last_hash_level: Optional[int] = None
        last_hash_num_hashes: Optional[int] = None
        
        if headers[0].type == 'hash_header':
            last_hash_level = 1
            last_hash_num_hashes = headers[0].metadata['num_hashes']
        
        # Process remaining headers
        for i in range(1, len(headers)):
            current_header = headers[i]
            previous_header = headers[i - 1]
            
            if current_header.type == 'hash_header':
                num_hashes: int = current_header.metadata['num_hashes']
                
                if last_hash_level is not None and last_hash_num_hashes is not None:
                    # Use formula with last hash header
                    current_header.level = last_hash_level + (num_hashes - last_hash_num_hashes)
                else:
                    # No previous hash header, use last header level + 1
                    current_header.level = previous_header.level + 1
                
                # Update last hash header info
                last_hash_level = current_header.level
                last_hash_num_hashes = num_hashes
                
            elif current_header.type == 'asterisk_header':
                num_asterisks: int = current_header.metadata['num_asterisks']
                
                if previous_header.type == 'hash_header':
                    # Previous is hash, assign previous level + 1
                    current_header.level = previous_header.level + 1
                else:
                    # Previous is asterisk, use formula with custom ordering
                    prev_asterisk_level: int = previous_header.level
                    prev_asterisk_num_asterisks: int = previous_header.metadata['num_asterisks']
                    
                    # Convert to comparable order values
                    current_order: int = self._asterisk_order(num_asterisks)
                    prev_order: int = self._asterisk_order(prev_asterisk_num_asterisks)
                    
                    current_header.level = prev_asterisk_level + (current_order - prev_order)
    
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
        
        root: Dict[str, Any] = {
            **self.metadata,
            'sections': [],
        }
        
        # Stack to keep track of the current hierarchy
        # Each element is a tuple: (level, section_dict)
        stack: List[tuple[int, Dict[str, Any]]] = [(0, root)]
        current_section: Optional[Dict[str, Any]] = None
        
        for token in self.tokens:
            if token.type in ['hash_header', 'asterisk_header']:
                # Create new section
                section: Dict[str, Any] = {
                    'type': token.type,
                    'title': token.content,
                    'level': token.level,
                    'line_number': token.line_number,
                    'metadata': token.metadata,
                    'content': [],
                    'sections': []
                }
                
                # Pop from stack until we find the right parent
                while len(stack) > 1 and stack[-1][0] >= token.level:
                    stack.pop()
                
                # Add section to parent
                parent: Dict[str, Any] = stack[-1][1]
                parent['sections'].append(section)
                
                # Push current section onto stack
                stack.append((token.level, section))
                current_section = section
                
            elif token.type == 'non_header':
                # Add content to the current section
                content_item: Dict[str, Any] = {
                    'type': 'content',
                    'text': token.content,
                    'line_number': token.line_number
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
        headers: List[Token] = [
            token for token in self.tokens
            if token.type in ['hash_header', 'asterisk_header']
        ]
        
        if not headers:
            print("No headers found")
            return
        
        print("Document Structure")
        print("=" * 80)
        
        # Track the levels and whether they have more siblings
        level_has_more: Dict[int, bool] = {}
        
        for i, header in enumerate(headers):
            # Determine if this header has siblings after it at the same level
            has_more_siblings = False
            for j in range(i + 1, len(headers)):
                if headers[j].level < header.level:
                    break
                if headers[j].level == header.level:
                    has_more_siblings = True
                    break
            
            # Update the level_has_more tracking
            level_has_more[header.level] = has_more_siblings
            
            # Build the tree prefix
            prefix = ""
            for level in range(1, header.level):
                if level in level_has_more and level_has_more[level]:
                    prefix += "│   "
                else:
                    prefix += "    "
            
            # Add the branch character
            if header.level > 1:
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
    
    def _flatten_recursive(
        self,
        sections: List[Dict[str, Any]],
        path: List[str],
        flat_dict: Dict[str, Any],
        separator: str,
    ) -> None:
        """Recursively flattens sections into the flat_dict.
        
        Args:
            sections: List of section dictionaries to process.
            path: Current path in the hierarchy (list of section titles).
            flat_dict: Dictionary to populate with flattened paths and content.
            separator: String to use between section names in paths.
        """
        for section in sections:
            if section.get('type') == 'content':
                # This is content, not a section
                if path:
                    # Content under a section
                    content_path = separator.join(path)
                else:
                    # Headerless content at root - use line number
                    line_num = section.get('line_number', 'unknown')
                    content_path = f"content_line_{line_num}"
                
                # Append to existing content or create new
                if content_path in flat_dict:
                    flat_dict[content_path] += '\n' + section['text']
                else:
                    flat_dict[content_path] = section['text']
            else:
                # This is a section with a title
                section_title = section.get('title', '')
                new_path = path + [section_title]
                
                # Add the section's content
                if section.get('content'):
                    content_texts = [item['text'] for item in section['content']]
                    combined_content = '\n'.join(content_texts)
                    content_path = separator.join(new_path)
                    flat_dict[content_path] = combined_content
                
                # Recursively process subsections
                if section.get('sections'):
                    self._flatten_recursive(section['sections'], new_path, flat_dict, separator)
    
    def flatten_sections(
        self,
        separator: str = " > ",
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Flattens the hierarchical structure into a flat dictionary.
        
        This method converts the nested section structure into a flat dictionary
        where each key represents the full path to a section. For headerless content
        (content at the root level without any headers), the path will use the line
        number where the content appears.
        
        Args:
            separator: String to use between section names in paths. Defaults to " > ".
            include_metadata: Whether to include document metadata in output. Defaults to True.
        
        Returns:
            A flat dictionary where keys are paths like "Section > Subsection"
            and values are the corresponding text content. Headerless content will
            have keys like "content_line_1".
        """
        hierarchical_data = self.to_hierarchical_dict()
        flat_dict: Dict[str, Any] = {}
        
        # Add document metadata if requested
        if include_metadata:
            for key, value in self.metadata.items():
                flat_dict[key] = value
        
        # Start flattening from root sections
        if 'sections' in hierarchical_data:
            self._flatten_recursive(hierarchical_data['sections'], [], flat_dict, separator)
        
        return flat_dict


def process_markdown_row(
    row: Dict[str, Any],
    content_column: str = 'content',
    metadata_columns: Optional[List[str]] = None,
    document_type: str = 'clinical_report',
    separator: str = ' > ',
    asterisk_header_max_words: int = 7,
) -> Dict[str, Any]:
    """Processes a single row containing markdown text.
    
    This function parses markdown content from a row and flattens it into
    a dictionary with paths as keys and content as values.
    
    Args:
        row: Dictionary representing a dataframe row.
        content_column: Name of the column containing markdown text. Defaults to 'content'.
        metadata_columns: List of column names to include as metadata. If None,
            all columns except content_column are included. Defaults to None.
        document_type: Type of document to add to metadata. Defaults to 'clinical_report'.
        separator: String to use between section names in paths. Defaults to ' > '.
        asterisk_header_max_words: Maximum words for asterisk headers. Defaults to 7.
    
    Returns:
        A flat dictionary with section paths as keys and content as values,
        including metadata from the original row.
    """
    # Extract content
    text = row.get(content_column, '')
    
    # Determine metadata columns
    if metadata_columns is None:
        metadata_columns = [col for col in row.keys() if col != content_column]
    
    # Build metadata dictionary
    metadata = {col: row.get(col) for col in metadata_columns}
    metadata['type'] = document_type
    
    # Parse and flatten
    parser = MarkdownParser(
        text=text,
        asterisk_header_max_words=asterisk_header_max_words,
        metadata=metadata,
    )
    parser.parse()
    
    return parser.flatten_sections(separator=separator, include_metadata=True)


def batch_process_markdown_dataframe(
    df: 'pd.DataFrame',
    content_column: str = 'content',
    metadata_columns: Optional[List[str]] = None,
    document_type: str = 'clinical_report',
    separator: str = ' > ',
    asterisk_header_max_words: int = 7,
    fill_missing: Any = None,
) -> 'pd.DataFrame':
    """Batch processes a dataframe containing markdown documents.
    
    This function processes each row in the dataframe, parses the markdown content,
    and creates a new dataframe where each unique section path becomes a column.
    
    Args:
        df: Input dataframe with markdown content.
        content_column: Name of the column containing markdown text. Defaults to 'content'.
        metadata_columns: List of column names to include as metadata. If None,
            all columns except content_column are included. Defaults to None.
        document_type: Type of document to add to metadata. Defaults to 'clinical_report'.
        separator: String to use between section names in paths. Defaults to ' > '.
        asterisk_header_max_words: Maximum words for asterisk headers. Defaults to 7.
        fill_missing: Value to use for missing sections. Defaults to None.
    
    Returns:
        A new dataframe where each row represents a processed document and each
        column represents either metadata or a section path.
    
    Raises:
        ValueError: If the specified content_column doesn't exist in the dataframe.
    """    
    if content_column not in df.columns:
        raise ValueError(
            f"Column '{content_column}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Process each row
    processed_rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        flattened = process_markdown_row(
            row=row_dict,
            content_column=content_column,
            metadata_columns=metadata_columns,
            document_type=document_type,
            separator=separator,
            asterisk_header_max_words=asterisk_header_max_words,
        )
        processed_rows.append(flattened)
    
    # Convert to dataframe
    result_df = pd.DataFrame(processed_rows)
    
    # Fill missing values if specified
    if fill_missing is not None:
        result_df = result_df.fillna(fill_missing)
    
    return result_df

"""myzip.py

In this task you will be writing an encoder (compression method) and decoder (uncompression method) implementing the Lempel and Ziv’s (LZ77) algorithm introduced in your Week 9 lecture.

name: Kevin Yu
student_id: 306128531

"""
import sys
import os
from bitarray import bitarray, util
from heapq import heappush, heappop
from math import inf
from pathlib import Path
# Constants
ALPHABET_LENGTH = 256

# Types
Node = type["LeafNode"] | type["InternalNode"]
child = Node | None

class GenericNode:
  """
  We use the below Node classes when creating our Huffman coding tree
  Both LeafNodes and InternalNodes store a "frequency" attribute, though this means different things
  - Frequency on a LeafNode refers to the number of times that the character appears in the text
  - Frequency on an InternalNode refers to the combined frequency of all LeafNodes below that InternalNode
  """

  def __init__(self, is_leaf: bool, frequency: int):
    self.is_leaf = is_leaf
    self.frequency = frequency

  def __lt__(self, other: Node):
    # We implement __lt__ so that Nodes can be sorted when using the min heap
    return self.frequency < other.frequency

  # Used for debugging
  def __str__(self):
    return f'{self.frequency}'

  # Used for debugging
  def __repr__(self):
    return self.__str__()

class LeafNode(GenericNode):
  def __init__(self, character: str, frequency: int):
    # LeafNodes also store a unique character in the text along with the frequency of that character
    # This is so that when we make our tree traversal and end up at a LeafNode
    # we can figure out the character that our path represents
    self.character = character
    super().__init__(is_leaf=True, frequency=frequency)

  # Used for debugging
  def __str__(self):
    return f'Leaf | {self.frequency}'

class InternalNode(GenericNode):
  def __init__(self, left: child, right: child, frequency: int):
    # Internal nodes store a left and a right Node (can be a Leaf or InternalNode)
    # self.left is always the Node with a smaller frequency
    self.left = left
    self.right = right
    super().__init__(is_leaf=False, frequency=frequency)

  # Used for debugging
  def __str__(self):
    return f'[InternalNode | {self.frequency}] -> L({self.left}) R({self.right})'


class TextSearchHelper:
  """
  TextSearchHelper uses a modified version of my implementation of modified Boyer Moore from Assignment 1
  The find_match function is new, and the exec_Modified_BoyerMoore has been modified
  The rest of the functions are the same
  """
  length = int
  distance = int
  following_char = str
  match = (distance, length, following_char)
  """A match found during lz77 encoding

  Returns:
      _type_: (distance, length, following_char)

  Return Types:
      distance (int): The distance from the search_start_index to the start of the matching string in the search window
      length (int): The length of the matching string found in the search window
      following_char (str): The character following the matching string in the lookahead buffer. Can also be ""
  """

  def find_match(self, text: str, search_start_index: int, lookahead_start_index: int, lookahead_end_index: int) -> match:
    """Finds the longest matching string in text[lookahead_start_index: lookahead_end_index] that belongs in text[search_start_index: lookahead_start_index]

    Args:
        text (str): The text to be searched
        search_start_index (int): The starting index of the search window
        lookahead_start_index (int): The starting index of the lookahead section
        lookahead_end_index (int): The ending index of the lookahead section

    Returns:
        match: A match found during lz77 encoding. Defined above

    How this function works:
     - We utilise the modified Boyer Moore algorithm from Assignment 1
     - We pass in the:
        1. Text as the full string between search_start_index and lookahead_end_index = text[search_start_index: lookahead_end_index]
        2. Pattern as the string between lookahead_start_index and lookahead_end_index = text[lookahead_start_index: lookahead_end_index]

    However, we can't use the original modified Boyer Moore immediately; there are a couple of issues/solutions:
      1. Boyer Moore returns full matches of pattern within the text, while we want the longest substring within text that matches
        Solution: Since at any point, the pointer on the pattern tells us the longest match thus far, and the text pointer tells us where, we can record this
        and keep updating the longest match value until the algorithm completes, and then return it
      2. Boyer Moore iterates through the pattern from back to front. Whereas we'd want matches beginning from the start of the string
        Solution: You could make a reversed implementation of the entire Boyer Moore algorithm. That would be more efficient. However I opted to 
        instead reverse the inputs to the Boyer Moore function, the expense being an overhead of reversing the strings before passing them in
      3. The pattern in the search window can extend into the buffer so we need to include the buffer in the text passed into Boyer Moore. However, 
        we also don't want the pattern to just match on the entire buffer section.
        Solution: Use a custom text start pointer. This will make sure that we effectively start matching from the edge of the lookahead_start_index - 1, and move backwards,
        meaning that we never cross into the buffer territory
    """

    # Handling the special case when our text is empty, though i'm pretty sure we'll never actually reach this block. Its a safeguard
    if len(text) == 0:
      return (0, 0, "")

    text_len = len(text)

    # We are finding the negative indexed positions of the start and end of pattern, and text (the entire window)
    # This is so that we can do a reverse slice of text, and retrieve the reversed slices of pattern and text in one go
    pat_start_index = - text_len + (lookahead_end_index - 1)
    pat_end_index = - text_len + (lookahead_start_index - 1)
    window_start_index = pat_start_index
    window_end_index = pat_end_index - (lookahead_start_index - search_start_index)

    # Next we need to get the pattern slice.
    # First we get the pattern that we are looking for in reverse. This takes O(B) time complexity where B is the size of the buffer
    pat = text[pat_start_index: pat_end_index: -1]

    # Next we get the text that we are searching through in reverse. This takes O(W + B) time complexity where W is the size of the window
    # We want window and buffer insteae of just buffer as the pattern in window can extend into the buffer as well
    extended_window = text[window_start_index: window_end_index: -1]

    # The max length of your match is len(lookahead_buffer_size). This means that the matched string can extend up to i + lookahead_buffer_size - 1, since
    # the matched string still has to begin in the dictionary (search window)
    # In order to make the sure the pattern doesn't just match on the buffer itself (since our extended_window (txt) is now window + buffer)
    # we need to add a custom text start index, which is buffer_length + 1
    buffer_length = (lookahead_end_index - 1 - lookahead_start_index)
    custom_start_txt_end_pointer = buffer_length + 1

    # The output gives us the position in the reversed(window + buffer)
    (matched, txt_end_ponter, pattern_pointer) = self.exec_Modified_BoyerMoore(
        txt=extended_window, pattern=pat, custom_start_txt_end_pointer=custom_start_txt_end_pointer)

    if (not matched):
      # If we didn't find a match, then return nothing with the next char
      if lookahead_start_index < text_len:
        # Pretty sure we can never have a situation where lookahead_start_index >= text_len here, but just in case ^^
        return (0, 0, text[lookahead_start_index])
      else:
        return (0, 0, "")

    # Calculate real length, distance, and next_char_index from the matched reversed output indexes
    length = len(pat) - pattern_pointer
    distance = txt_end_ponter - buffer_length
    next_char_index = lookahead_start_index + length

    next_char = ""
    if next_char_index < text_len:
      # If we haven't reached the end of the text, then we want to add the next character.
      # Otherwise, just use the empty string
      next_char = text[next_char_index]

    return (distance, length, next_char)

  def exec_Modified_BoyerMoore(self, txt: str, pattern: str, custom_start_txt_end_pointer: int):
    """Executes the actual BoyerMoore algorithm given a pattern and a text to match against

    Args:
        txt (str): txt to scan for pat
        pattern (str): pat that is to be found in txt
    """
    # Preprocessing the pattern
    extended_bad_character_matrix = self.compute_extended_bad_character_matrix(pattern)
    zSuffixes = self.compute_z_suffix_values(pattern)
    good_suffix_values = self.compute_good_suffix_values(pattern)
    mismatched_suffix_values = self.compute_mismatched_suffix_values(pattern)
    matched_prefix_values = self.compute_matched_prefix_values(pattern)

    pattern_len = len(pattern)
    txt_len = len(txt)

    # As we will be iterating through the pattern/txt, these pointers are used to keep track of our location
    # We add the custom start txt end pointer so that we only match from the position of buffer - 1
    txt_end_pointer = custom_start_txt_end_pointer
    pattern_pointer = pattern_len - 1

    # Used for Galil's optimisation to skip over parts of the pattern
    stop = -1
    start = -1

    # (has_matched?, txt_end_pointer, pattern_pointer)
    # This stores whether we have matched, and if we have, the longest match thus far
    longest_match_indexes = (False, None, None)

    while txt_end_pointer < txt_len:
      txt_pointer = txt_end_pointer

      # Matching phase
      while txt_pointer < txt_len and pattern_pointer < pattern_len and txt[txt_pointer] == pattern[pattern_pointer]:
        # If we have reached the end of our pattern, we have found a match!

        # We take the longest match index pair depending on how far into the pattern we are (found through pattern_len - pattern_pointer which is x[1])
        # We also take any match if we haven't found a match yet (through x[0])
        longest_match_indexes = max(longest_match_indexes, (True, txt_end_pointer,
                                    pattern_pointer), key=lambda x: -inf if not x[0] else pattern_len - x[2])

        if pattern_pointer == 0:
          # If we have reached the end of our pattern, that means we have found a match. Since we are only looking for one complete match, we can return here
          length = pattern_len - pattern_pointer
          return longest_match_indexes
        # Otherwise, we decrement the pointers since the while loop has checked for equivalency between txt[txt_pointer] and pattern[pattern_pointer]
        txt_pointer -= 1
        pattern_pointer -= 1

        # If we have reached the stop index, this indicates that we need to make a skip
        if pattern_pointer == stop:
          # Shift the pattern_pointer to the start that we set and txt_pointer by the same amount
          shift_amount = stop - start
          pattern_pointer = max(start, 0)
          txt_pointer -= shift_amount
          new_txt = txt[txt_pointer]
          new_pat = pattern[pattern_pointer]

      # We've either found a match, or had a mistmatch. Either way, we need to figure out how much we're going to shift pattern by
      # Calculate all the different shifts we can make
      mismatched_character = txt[txt_pointer]
      bad_character_index = self.get_bad_character_value(
          extended_bad_character_matrix, mismatched_character, pattern_pointer)
      bad_character_shift = pattern_pointer - bad_character_index
      good_suffix_shift = good_suffix_values[pattern_pointer]

      if pattern_pointer + 1 == pattern_len:
        suffix_shift = 1
      # If we can't find a good suffix, we use the matched_prefix value
      elif good_suffix_values[pattern_pointer + 1] == -1:
        suffix_shift = pattern_len - matched_prefix_values[pattern_pointer + 1]
        stop = matched_prefix_values[pattern_pointer + 1] - 1
        start = 0
      else:
        mismatched_suffix_value = mismatched_suffix_values[pattern_pointer + 1][ord(
            mismatched_character)]
        # If there is a mismatched_suffix_value that matches the current suffix with the mismatched character at the start of it
        if mismatched_suffix_value != -1:
          suffix_shift = pattern_pointer - mismatched_suffix_value
          stop = mismatched_suffix_value + (pattern_len - pattern_pointer) - 1
          start = mismatched_suffix_value + 1
        # Otherwise, just use the usual good_suffix_value
        else:
          suffix_shift = pattern_len - 1 - good_suffix_values[pattern_pointer + 1]
          stop = good_suffix_values[pattern_pointer + 1] - 1
          start = good_suffix_values[pattern_pointer + 1] - pattern_len + pattern_pointer + 1

      shift = max(bad_character_shift, suffix_shift)

      # If the shift != suffix_shift that means that we are using bad_character_shift
      # In this case we don't want to skip past parts of the string so we reset stop and start
      if shift != suffix_shift:
        stop = -1
        start = -1

      txt_end_pointer += shift
      pattern_pointer = pattern_len - 1

    length = pattern_len - pattern_pointer

    # We return the indexes of the longest match we have found
    return longest_match_indexes

  # The rest of the functions below have no changes
  def compute_extended_bad_character_matrix(self, pattern: str) -> list[list[int]]:
    """Computes the bad character matrix given a pattern
    This matrix is used to find the rightmost instance of a character given a position
    in the pattern

    Here we are assuming that the inputs are ASCII and within

    Args:
        pattern (str): the pattern to be processed
    """
    r_matrix = [[-1 for _ in range(ALPHABET_LENGTH)] for _ in range(len(pattern))]
    for i in range(len(pattern) - 1):
      character = pattern[i]
      r_matrix[i+1] = r_matrix[i].copy()
      r_matrix[i+1][ord(character)] = i
    return r_matrix

  def get_bad_character_value(self, matrix: list[list[int]], character: str, position: int) -> int:
    """Helper function to get the index of a character given a bad character matrix and a position in pattern

    Args:
        matrix (list[list[int]]): bad character matrix
        character (str): character to be found
        position (int): position in pattern

    Returns:
        int: the position of the bad character
    """
    return matrix[position][ord(character)]

  def compute_z_values(self, input_str: str) -> list[int]:
    """computes the z values for an input str

    Args:
        input_str (str): string to compute z values for

    Returns:
        list[int]: the list of z values
    """
    if len(input_str) == 0:
      return []

    if len(input_str) == 1:
      return [1]

    output = [0 for i in range(len(input_str))]
    output[1] = self.get_z_value_by_direct_comparison(input_str, 1)
    r = 0
    l = 0
    if output[1] > 0:
      # Right is output[1] + 1 as the index of the current Z value is 1.
      # output[1] would be correct if we were at index 0
      r = output[1]
      # Left is one as this is the index of the current Z value
      l = 1
    for i in range(2, len(input_str)):
      current = input_str[i]
      # i is the current index that we are trying to compute a Z value for
      # If i > r, then this means we are not inside a z box.
      # We will have to check for a matching substring through explicit comparison
      if i > r:
        z_value = self.get_z_value_by_direct_comparison(input_str, i)
        output[i] = z_value
        if output[i] > 0:
          r = i + output[i] - 1
          l = i
      else:
        prefix_index = i - l
        # If we are in a zbox still
        if output[prefix_index] + i <= r:
          output[i] = output[prefix_index]
        # Otherwise, we are at the end or beyond the zbox
        # In this case, we will need to get the z value by direct comparison since we have no knowledge of values beyond the box
        else:
          # This is index of the matching starting char at the string's prefix
          # We want to compare the char at r + 1 with the prefix chars
          prefix_box_end_index = r - i + 1
          z_value = self.get_z_value_by_direct_comparison(input_str, prefix_box_end_index, r + 1)
          output[i] = r - i + 1 + z_value
          l = i
    return output

  def get_z_value_by_direct_comparison(self, input_str: str, index: int, input_index: int = 0) -> int:
    """function that computes z values given an input string and two indexes to compare
    """
    substring_index = index
    count = 0
    while input_index < len(input_str) and substring_index < len(input_str) and input_str[input_index] == input_str[substring_index]:
      input_index += 1
      substring_index += 1
      count += 1
    return count

  def compute_z_suffix_values(self, input_str: str) -> str:
    """Computes the z suffix values
    """
    return self.compute_z_values(input_str[::-1])[::-1]

  def compute_good_suffix_values(self, input_str: str) -> list[int]:
    """Computes the good suffix values

    Args:
        input_str (str): input string to compute good suffixes for

    Returns:
        list[int]: list of good suffixes
    """
    input_len = len(input_str)
    z_suffix_values = self.compute_z_suffix_values(input_str)
    good_suffix_values = [-1 for i in range(len(input_str) + 1)]
    for index, z_value in enumerate(z_suffix_values):
      if z_value > 0:
        good_suffix_values[input_len - z_value] = index + 1
    return good_suffix_values

  def compute_mismatched_suffix_values(self, input_str: str) -> list[list[int]]:
    """Computes the mismatched suffix values for an input string

    Given a string input_str, let m be its computed mismatched_suffix_values

    With a character c and index between [0, len(input_str)],  m[index][ord(c)] returns the rightmost instance 
    of a matching substring that matches the suffix [index, len(input_str)], and is also preceeded by the character c
    """
    input_len = len(input_str)
    z_suffix_values = self.compute_z_suffix_values(input_str)
    mismatched_suffix_values = [[-1 for i in range(ALPHABET_LENGTH)] for j in range(len(input_str) + 1)]

    for index, z_value in enumerate(z_suffix_values):
      # For each substring input_str[a,b] that matches a suffix input_str[x,y], we want to store
      # the index a - 1 (index - z_value), the character after the leftmost part of the substring
      # This index is stored at m[x][ord(input_str[index - z_value])], which allows O(1) access
      # This way, when we mismatch at the left end of a suffix (x), we can easily access the mismatched
      # suffix value. This allows us to shift per the modified shift rules in the assignment
      if z_value > 0 and index != 0:
        mismatched_suffix_values[input_len -
                                 z_value][ord(input_str[index - z_value])] = index - z_value
    return mismatched_suffix_values

  def compute_matched_prefix_values(self, input_str: str) -> list[int]:
    input_len = len(input_str)

    matched_prefix_values = [-1 for i in range(len(input_str) + 1)]

    z_values = self.compute_z_values(input_str)

    current_largest_prefix_length = 0

    for index in range(len(z_values) - 1, -1, -1):
      z_value = z_values[index]
      if z_value + index == input_len:
        current_largest_prefix_length = z_value
      matched_prefix_values[index] = current_largest_prefix_length
    matched_prefix_values[0] = input_len
    return matched_prefix_values


class Encoder:
  """Encoder
  Contains the main functions used to encode a file
  """
  encoding: type(bitarray)

  def __init__(self, filename: str, max_search_window_size: int, max_lookahead_size: int):
    # Get the finalpath component
    self.filename = Path(filename).name
    self.file_contents = FileHelper.read_file(filename)
    self.max_search_window_size = max_search_window_size
    self.max_lookahead_size = max_lookahead_size
    # This encoding object is what we will be extending throughout the encoding process
    self.encoding = bitarray()
    # This is a special flag to handle encoding differently if we only have one distinct character
    self.has_only_one_distinct_character = False
    # TextSearchHelper object used during lz77 encoding
    self.TextSearchHelper = TextSearchHelper()

  def encode(self) -> type(bitarray):
    # Encode the header
    self.encode_header()
    # Encode the data
    self.encode_data()

    return self.encoding

  def encode_header(self):
    # Encode the length of the filename
    self.encoding.extend(self.encode_number(len(self.filename)))

    # Encode filename
    for char in self.filename:
      self.encoding.extend(util.int2ba(ord(char), length=8))

    # Encode file size
    self.encoding.extend(self.encode_number(len(self.file_contents)))

    # Create Huffman codewords and add as instance variables
    (self.distinct_character_count, self.characters,
     self.char_lookup) = self.generate_huffman_coding()

    # Encode Huffman codewords into the header using the generated codewords above
    self.encoding.extend(self.generate_huffman_coding_header())

  def encode_data(self):
    # Encode the data using lz77 algorithm, utilising the Huffman Codewords created earlier
    self.encode_with_lz77()

  num_chars = int
  characters = [str]
  char_lookup = [type(bitarray)]
  generate_huffman_coding_output_type = (num_chars, characters, char_lookup)

  def generate_huffman_coding(self) -> generate_huffman_coding_output_type:
    """generates the huffman coding for the text

    Returns:
        generate_huffman_coding_output_type: (num_chars, characters, char_lookup)
        num_chars (int): number of unique characters
        characters ([str]): a list of the unique characters
        char_lookup ([type(bitarray)]): a list of size ALPHABET_LENGTH (256) that allows O(1) lookup of a character's Huffman codeword
    """
    # Get the count of each character in the text
    character_frequencies = self.count_character_frequencies()

    # List that will be used as a priority queue while creating the Huffman Encoding
    queue = []

    # We want to insert each unique character into the queue with its frequency as a LeafNode
    for frequency, character in character_frequencies:
      heappush(queue, LeafNode(character, frequency))

    # With all the LeafNodes in the priority queue, we can begin the process of joining Leaves and Internal Nodes until there is only one
    # Internal node left
    while len(queue) > 1:
      # Get the two nodes with smallest frequency from the queue using heap pop
      # Always make node_left the smaller node, which will be popped off the heap first
      node_left = heappop(queue)
      node_right = heappop(queue)

      # Create a new node that has frequency equal to the two children nodes combined
      # Make the smaller of the two children the left node and the larger the right node
      new_node = InternalNode(
          node_left, node_right, node_left.frequency + node_right.frequency)
      # Push this new internal node onto the priority queue
      heappush(queue, new_node)

    # Next we want to traverse the tree created above to get the Huffman Codewords for each character
    # We create two pieces of data,
    # 1. char_lookup, which allows you to access the codeword for a character c by char_lookup[ord(c)] -> O(1) access
    # 2. characters, which allows you to loop through the characters

    char_lookup = [None for _ in range(ALPHABET_LENGTH)]
    characters = []

    if len(queue) > 0:
      # The only node left in the queue should be the root node
      root_node = heappop(queue)

      # Generate the paths
      # A path is a tuple (character, bitarray), being a character from the text and the corresponding bitarray it is represented by
      paths = self.build_path(root_node)

      # For each character and its corresponding bitarray
      for character, bitarray in paths:
        # Add the pair to our char_lookup table and characters list
        char_lookup[ord(character)] = bitarray
        characters.append(character)

      return (len(paths), characters, char_lookup)

    return (0, characters, char_lookup)

  def count_character_frequencies(self):
    total_character_counts = [0 for i in range(ALPHABET_LENGTH)]

    # Count occurences of each character
    for character in self.file_contents:
      total_character_counts[ord(character)] += 1

    # Return an array of tuples (frequency, character) for each unique character in text
    result = [(frequency, chr(index)) for index, frequency in enumerate(
        total_character_counts) if frequency != 0]

    return result

  # This recursive path builder was replaced by the implementation underneath as it is more memory heavy than the iterative approach
  # def build_path(self, node: Node):
  #   paths = []
  #   def aux_build_path(node: Node, current: type[bitarray], paths: [type(bitarray)]):
  #     if node.is_leaf:
  #       paths.append((node.character, current))
  #       return current
  #     else:
  #       current_left = current.copy()
  #       current_left.append(0)
  #       current_right = current.copy()
  #       current_right.append(1)

  #       aux_build_path(node.left, current_left, paths)
  #       aux_build_path(node.right, current_right, paths)

  #   aux_build_path(node, bitarray(), paths)
  #   return paths

  codeword = type(bitarray)
  character = str

  def build_path(self, node: Node) -> [(character, codeword)]:
    """build codewords for each character

    Returns:
        [(character, codeword)]: List of tuples, where first index is character and second index is the corresponding codeword
    """
    # Use a stack to implement an in order traversal of the tree
    stack = []
    paths = []
    # Append the root node with empty bitarray, as it is the root node
    stack.append((node, bitarray()))

    while len(stack) > 0:
      # Get the next node to look at
      current_node, path = stack.pop()

      if current_node.is_leaf:
        # If the node is a leaf, we've reach the end of a path, so we append the character at the node along with the path we've created
        paths.append((current_node.character, path))
      else:
        # Otherwise, we want to add the left and right children onto the stack
        new_right_path = path.copy()
        new_right_path.append(1)
        new_left_path = path.copy()
        new_left_path.append(0)

        stack.append((current_node.right, new_right_path))
        # Append the left node last so that it is popped first
        stack.append((current_node.left, new_left_path))

    return paths

  def generate_huffman_coding_header(self):
    huffman_coding_header = bitarray()
    huffman_coding_header.extend(
        self.encode_number(self.distinct_character_count))
    for character in self.characters:
      # For each character;
      # Add 8-bit ASCII code of each character
      huffman_coding_header.extend(util.int2ba(ord(character), length=8))
      # Add length of Huffman codeword
      huffman_codeword = self.char_lookup[ord(character)]
      huffman_codeword_length = len(huffman_codeword)
      huffman_coding_header.extend(
          self.encode_number(huffman_codeword_length))
      # Add the Huffman codeword itself
      huffman_coding_header.extend(huffman_codeword)

    return huffman_coding_header

  def encode_with_lz77(self):
    """encode the data using LZ77 algorithm

    Returns:
        _type_: _description_
    """
    text_length = len(self.file_contents)
    # Initialise the window variables, lookahead buffer start and end index, and search window start index
    lookahead_buffer_start_index = 0
    lookahead_buffer_end_index = lookahead_buffer_start_index + self.max_lookahead_size
    search_window_start_index = 0

    # While we still have characters to encode in the lookahead buffer
    while lookahead_buffer_start_index < len(self.file_contents):
      # Find the longest matching substring in the buffer that also exists in the search window. Get the distance from the buffer start index, the length of this string,
      # as well as the following character. The following character can be empty string ""
      (distance, length, character) = self.TextSearchHelper.find_match(text=self.file_contents, search_start_index=search_window_start_index,
                                                                       lookahead_start_index=lookahead_buffer_start_index, lookahead_end_index=lookahead_buffer_end_index)

      # Get the character_code for the ending character
      character_code = self.get_bitarray_for_character(character)

      # Add the encoded distance of the match from the start of the lookahead buffer
      self.encoding.extend(self.encode_number(distance))
      # Add the encoded length of the match
      self.encoding.extend(self.encode_number(length))
      # Add the character following the matched substring
      self.encoding.extend(character_code)

      # Compute the new window values for the lookahead buffer start and end index, and search window start index
      lookahead_buffer_start_index = min(
          text_length, lookahead_buffer_start_index + length + 1)
      lookahead_buffer_end_index = min(
          text_length, lookahead_buffer_end_index + length + 1)
      search_window_start_index = max(
          0, lookahead_buffer_start_index - self.max_search_window_size)

  def get_bitarray_for_character(self, character: str) -> type(bitarray):
    # We have a special function for this get action, as we need to handle the empty string case.
    # This is because the empty string does not exist in our char_lookup table
    if len(character) > 0:
      return self.char_lookup[ord(character)]
    # If the len(character) == 0 then it is the empty string.
    # We will return an empty bitarray as there is nothing to encode
    return bitarray()

  def encode_number(self, n: int) -> type(bitarray):
    """Encode a number into a sequence of bits using elias encoding

    Args:
      n (int): The number to be encoded

    Returns:
        type(bitarray) : a bitarray representing the elias encoded number in binary
    """
    # Since we need to encode numbers n where n >= 0, we need to shift all n by 1: n = n + 1
    # We will also shift back in the decoding process
    code = util.int2ba(n + 1)
    # We work with it in reverse the complexity of appending/extending to the end of the list is better than that of appending to the head
    code.reverse()

    # Initialise the current component that we are working on
    current_length_component = util.int2ba(len(code) - 1)

    while len(current_length_component) > 1:
      # Encode the current component by inverting the last (first when reversed) index
      current_length_component.invert(0)
      current_length_component.reverse()
      # Extend our code by this component
      code.extend(current_length_component)
      # Calculate the new current length component
      current_length_component = util.int2ba(
          len(current_length_component) - 1)

    # If we're here, then we've reached the length 1.
    # We need to encode the length component 1 by appending 0, but only if n != 0, as that is a special case.
    if n != 0:
      code.append(0)
    code.reverse()

    return code


class FileHelper:
  """Helper functions for reading from and writing to files
  """
  @staticmethod
  def read_file(filename):
    with open(filename) as f:
      return "".join(f.readlines())

  @staticmethod
  def write_to_file(bitarray: type(bitarray), filename: str):
    with open(filename, "wb") as f:
      # bitarray's tofile method pads the bitarray to a whole number of bytes
      # automatically when writing to the file
      bitarray.tofile(f)

def main(input_filename: str, w: str, l: str):
  search_window_size = int(w)
  lookahead_buffer_size = int(l)

  encoder = Encoder(filename=input_filename, max_search_window_size=search_window_size,
                    max_lookahead_size=lookahead_buffer_size)
  encoding = encoder.encode()

  # Note that this will output the file to the directory that you call this python script in
  new_filename = f"{Path(input_filename).name}.bin"

  FileHelper.write_to_file(encoding, new_filename)


if __name__ == "__main__":
  if len(sys.argv[1:]) > 0:
    args = [arg for arg in sys.argv[1:]]
    main(args[0], args[1], args[2])

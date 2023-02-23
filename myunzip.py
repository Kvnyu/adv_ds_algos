"""myunzip.py

In this task you will be writing an encoder (compression method) and decoder (uncompression method) implementing the Lempel and Ziv’s (LZ77) algorithm introduced in your Week 9 lecture.

name: Kevin Yu
student_id: 306128531

"""
import sys
import os
from heapq import heappush, heappop
from bitarray import bitarray, util
from pathlib import Path

ALPHABET_LENGTH = 256

Node = type["LeafNode"] | type["InternalNode"]
child = Node | None

class GenericNode:
  """
  We use the below Node classes when creating our Huffman coding tree
  Creating the huffman coding tree on the decoding side gives us an efficient data structure to parse Huffman encoded characters
  """

  def __init__(self, is_leaf: bool):
    self.is_leaf = is_leaf

  def __str__(self):
    return 'GenericNode'

  def __repr__(self):
    return self.__str__()

class LeafNode(GenericNode):
  # LeafNodes store a unique character in the text
  # This is so that when we make our tree traversal when decoding and end up at a LeafNode
  # we can figure out the character that our path represents
  def __init__(self, character: str):
    self.character = character
    super().__init__(is_leaf=True)

  def __str__(self):
    return f'Leaf {self.character}'

class InternalNode(GenericNode):
  def __init__(self, left: child = None, right: child = None):
    self.left = left
    self.right = right
    super().__init__(is_leaf=False)

  def get_or_create_left_node(self):
    if not self.left:
      self.left = InternalNode()
    return self.left

  def get_or_create_right_node(self):
    if not self.right:
      self.right = InternalNode()
    return self.right

  def __str__(self):
    return f'[InternalNode] -> L({self.left}) R({self.right})'

class Decoder:
  """Decoder
  Contains the main functions used to encode a file
  """
  # encoding is the bitarray that we read in fromthe file
  encoding: type(bitarray)

  def __init__(self, encoding: type(bitarray)):
    # This is an important variable, used throughout the decoding process
    # It stores the entire contents of the binary file as a bitarray
    self.encoding = encoding
    # Current bit pointer represents the last bit that we have processed
    # Each function keeps the promise that by the end of its call, self.current_bit_pointer will be pointing to the last bit that it processed
    # This keeps it clean what the main logic of each function is, and also gives us a 'constant condition' we can depend on
    # Likewise, each function, when called, expects an input index that is the next bit we haven't yet looked at
    # So whenevever we pass the self.current_bit_pointer into a function the starting index, we need to + 1  to current_bit_poitner
    # to increment it to a bit that we haven't yet looked at
    self.current_bit_pointer = 0
    self.has_only_one_distinct_character = False

  filename = str
  data = str
  decode_output_type = (filename, data)

  def decode(self) -> decode_output_type:
    # Decode the header
    self.decode_header()
    # Decode the data
    self.decode_data()

    return (self.filename, self.data)

  def decode_header(self):
    # Decode the length of the filename
    filename_length = self.decode_number_from_bitarray(
        read_position=self.current_bit_pointer)

    # Decode the filename
    self.filename = self.read_filename(
        start_index=self.current_bit_pointer + 1,
        length=filename_length)

    # Decode the file size
    self.file_length = self.decode_number_from_bitarray(
        read_position=self.current_bit_pointer + 1)

    # Decode the character codes
    # character_codes is an array of (character, huffman_codeword)
    character_codes = self.decode_huffman_coding_header(
        self.current_bit_pointer + 1)

    # Build the Huffman tree
    # This Huffman tree allows us O(n) access to characters with their respective code of length n
    # This is used when we are decoding characters during the data decoding stage
    self.huffman_tree_root_node = self.build_huffman_tree(character_codes)

  def decode_data(self):
    # Decode the data using lz77 algorithm, utilising the Huffman codewords decoded earlier
    self.data = self.decode_with_lz77(
        start_index=self.current_bit_pointer + 1)

  character = str
  huffman_codeword = type(bitarray)

  def decode_huffman_coding_header(self, start_index: int) -> [(character, huffman_codeword)]:
    """decodes the portion of the header that stores Huffman codes for the characters in the file

    Returns: [(character, huffman_codeword)]
    """
    # A list of tuples, storing each character along with its huffman_codeword
    character_codes: [(character, huffman_codeword)] = []

    # Get the number of distinct characters in the text
    distinct_character_count = self.decode_number_from_bitarray(
        read_position=start_index)

    # Initialise the bit pointer for this function
    current_index = self.current_bit_pointer

    # For each distinct character, we want to read the character and the codeword
    for i in range(distinct_character_count):
      # We know that each character is encoded in 8 bit ascii
      character = chr(util.ba2int(
          self.encoding[current_index + 1: current_index + 1 + 8]))

      # Increment the current index and instance variable current_bit_pointer
      # Incrementing the current_bit_pointer is necessary as the
      # decode_number_from_bitarray function utilises this instance variable
      current_index = current_index + 8
      self.current_bit_pointer = current_index

      # Get the length of the Huffman codeword
      huffman_codeword_length = self.decode_number_from_bitarray(
          read_position=current_index + 1)
      # The decode_number_from_bitarray updates the self.current_bit_pointer, so we need to
      # resync our current_index
      current_index = self.current_bit_pointer
      # Get the corresponding Huffman codeword using the length of the codeword we got earlier
      huffman_codeword = self.encoding[current_index +
                                       1: current_index + 1 + huffman_codeword_length]
      current_index += huffman_codeword_length

      # Add the character, huffman_codeword pair to our output list
      character_codes.append((character, huffman_codeword))

      # Update the instance variable current_bit_pointer
      self.current_bit_pointer = current_index

    return character_codes

  def build_huffman_tree(self, character_codes: [(character, huffman_codeword)]):
    """builds a huffman tree (more minimal than the encoding one for lookup)

    Args:
        character_codes [(character, huffman_codeword)]: The array of character to Huffman codewords we built earlier

    Returns:
        root_node (InternalNode) : The root node of the Huffman tree
    """
    root_node = InternalNode()
    if (len(character_codes) != 1):
      # If we have two or more characters, then insert characters
      for character, code in character_codes:
        self.insert_character_code(root_node, character, code)
    else:
      # If we only have one distinct character or none, then our Huffman tree should be empty (no leaves)
      # Set the flags to true and set the distinct character which we will use in the lookup later instead of using this tree
      self.has_only_one_distinct_character = True
      self.distinct_character = character_codes[0][0]
    return root_node

  def insert_character_code(self, root_node: type(GenericNode), character: character, code: huffman_codeword) -> type(GenericNode):
    """Inserts a character -> codeword pair into the Huffman tree. This allows for easy lookup when decoding codewords in the data 
      section. 

    Args:
        root_node (type(GenericNode)): root node to start walking from
        character (character): character we are inserting at a leaf
        code (huffman_codeword): Huffman codeword of the character we are inserting. Also the path that we will be following

    Returns:
        root_node (type(GenericNode)): root node of the Huffman tree
    """
    # We want to look at the code starting from the right side, therefore we first reverse it
    code.reverse()
    # Initialise the current_node variable
    current_node = root_node
    while len(code) > 1:
      # Look at the right most bit
      current_bit = code.pop()
      # If the current bit is a one, go right
      if current_bit:
        current_node = current_node.get_or_create_right_node()
      # Else, go left
      else:
        current_node = current_node.get_or_create_left_node()

    # Since our while loop runs while len(code) > 1, we will have 1 bit here
    current_bit = code.pop()
    # If current_bit == 1, then set the right child with the character
    if current_bit:
      current_node.right = LeafNode(character)
    # Otherwise, set the left child with the character
    else:
      current_node.left = LeafNode(character)
    return root_node

  def decode_number_from_bitarray(self, read_position: int) -> int:
    """decodes an Elias encoded number from the bitarray, starting from read_position

    Args:
        read_position (int): position to start decoding from

    Returns:
        int: the integer that was encoded
    """
    # start by reading 1 bit
    read_length = 1

    # While the bit we are looking at isn't a 1
    while not self.encoding[read_position]:
      # Get the next component by adding the read length onto the current position
      next_component = self.encoding[read_position: read_position + read_length]

      # Update the read position
      read_position += read_length

      # Invert the head of the next component
      next_component.invert(0)

      # Get the next read length and add 1
      read_length = util.ba2int(next_component) + 1

    # Update the instance variable self.current_bit_pointer
    self.current_bit_pointer = read_position + read_length - 1
    # If we're here, it means we hit a 1. We can then read the next read_length bits and turn it to a dec int
    # Minus one as we added one when first encoding to account for 0
    return util.ba2int(
        self.encoding[read_position: read_position + read_length]) - 1

  def read_filename(self, start_index: int, length: int):
    """decodes the filename of the original file

    Args:
        start_index (int): start index to begin reading from
        length (int): length of bits to be decoded

    Returns:
        filename (str): the original filename
    """
    filename = []

    for i in range(length):
      # For each character in the length of the filename
      # Convert the ascii to its integer representation and then into the character itself
      # Built the filename through the array of characters
      char = chr(util.ba2int(
          self.encoding[start_index + (i * 8): start_index + (i + 1) * 8]))
      self.current_bit_pointer += 8
      filename.append(char)

    # Return the filename as a string
    return "".join(filename)

  def decode_with_lz77(self, start_index: int) -> str:
    """decode the data using the LZ77 algorithm

    Args:
        start_index (int): the index to start looking at in self.encoding

    Returns:
        str: the data, as a string
    """
    char_array = []
    current_index = self.current_bit_pointer

    while len(char_array) < self.file_length:
      distance = self.decode_number_from_bitarray(
          read_position=self.current_bit_pointer + 1)
      length = self.decode_number_from_bitarray(
          read_position=self.current_bit_pointer + 1)
      character = self.read_huffman_code(self.current_bit_pointer + 1)

      start = len(char_array) - distance
      for i in range(start, start + length):
        assert i < len(char_array)
        # if len(char_array) < self.file_length - 1:
        char_array.append(char_array[i])

      if not len(char_array) < self.file_length:
        # We need to check if we can still append to the end of the char_array, as
        # There can be situations where the string is a repeating sequence of characters,
        # And the last next_char is an empty string
        # With encodings with only one repeated character, the char_code for the
        # character is empty. We also encode empty (nothing) for the empty string
        # This means that we need to check if we've reached the end when adding the next_char
        # as both the empty string and the single repeated character both have the same 'code',
        # that code being no code
        # To avoid this, we stop decoding when we've filled the char_array up to the file_length
        # that got decoded earlier
        break
      # Otherwise, we append the character at the end
      char_array.append(character)

    # Return the data as a string
    return ''.join(char_array)

  def read_huffman_code(self, start_index: int) -> str:
    """reads a huffman code

    Args:
        start_index (int): index to start reading from in self.encoding 

    Returns:
        str: the character represented by the huffman code beginning at start_index in self.encoding
    """
    # If we have the special case where there is only one distinct character,
    # then the Huffman tree will be empty, so we need to return the character
    # that we stored separately in self.distinct_character before
    if self.has_only_one_distinct_character:
      return self.distinct_character

    # Set current node in the Huffman tree
    current_node = self.huffman_tree_root_node
    # Set current index in self.encoding
    current_index = start_index
    # Start traversing the tree using the Huffman code
    # While we have not reached the leaf,
    while not current_node.is_leaf:
      # If the current bit is 1, go right
      if self.encoding[current_index]:
        current_node = current_node.right
      # Otherwise, go left
      else:
        current_node = current_node.left

      # We don't want to increment if we have hit a leaf after updating current_node

      if not current_node.is_leaf:
        current_index += 1

    # Update the instance variable self.current_bit_pointer
    self.current_bit_pointer = current_index
    return current_node.character

class FileHelper:
  """Helper functions for reading from and writing to files
  """
  @staticmethod
  def read_binary_to_bitarray(filename: str):
    result = bitarray()
    with open(filename, "rb") as f:
      result.fromfile(f)
    return result

  @staticmethod
  def write_to_file(filename: str, data: str):
    with open(filename, "w") as f:
      f.write(data)


def main(input_filename: str):
  encoding = FileHelper.read_binary_to_bitarray(input_filename)

  decoder = Decoder(encoding=encoding)

  (base_filename, data) = decoder.decode()

  # Note that this will output the file to the directory that you call this python script in
  FileHelper.write_to_file(base_filename, data)


if __name__ == "__main__":
  if len(sys.argv[1:]) > 0:
    args = [arg for arg in sys.argv[1:]]
    main(args[0])

"""gst.py

In the real-world, it is a routine task to search a collection (corpora) of texts and identify all
occurrences of patterns (often, by ignoring characters’ upper/lower case).
To render the search of any pattern efficient, an approach is to preprocess the collection of
texts beforehand, and then use that processed data structure to support efficient identification
of the locations of user-specified pattern(s).
For this question, you are required to construct a single suffix tree containing information of
suffixes of multiple (and not just one) input (text) strings: {txt1,txt2, . . . ,txtN }. In principle,
N (number of texts in the collection) can grow arbitrarily large.
Once the suffix tree for the set {txt1,txt2, ... ,txtN } is constructed, you will have to use it
to search and identify all the locations (positions) of exact occurrences of each pattern from a
given set of pattern strings: {pat1, pat2, ... , patM}.
In this task, all strings (i.e., text strings and pattern strings) are read from their respective
standard ASCII files (with 7-bit fixed-width ASCII code per character, where each character
takes 1 Byte of storage). Also, it is perfectly safe to assume that none of the (text and pattern)
strings being read from the ASCII files contain a ‘$’ (ASCII value 36) as one of its characters.
Finally, in this exercise, although we are doing exact matching of patterns, you should handle any pattern matching case-insensitively. 
That is, if you were seaching for ‘FIT3155’ in some
collection of texts, you should also report matches for ‘fit3155’ or ‘Fit3155’ or ‘fIt3155’ etc.
along with the ‘FIT3155’.


name: Kevin Yu
student_id: 306128531

"""

import sys

"""Notes
  There are some annoyances I have with my implementation
  1. The use of class variables. It seems a bit hacky as the algorithm can only be run once in the same script,
  and to rerun it you'd have to implement a reset function for each class. Classes that use class vairables include
  EdgeList, TrieBuilder

  2. The Generalized suffix trie isn't "clean". Since in my approach I concatenate the strings into an array of chars
  and using a special class UniqueEndCharacter as a separator and then do a suffix, the Tries created have edges that
  breach the end character. This isn't a problem to find substrings, as the Leaf will always store the initial text that
  lead to its creation, and the fact that patterns don't have $ (per the spec), and wouldn't be able to match it even if they did
  because of my UniqueEndCharacter class. However it is still annoying. I didn't have time to implement a pruning algorithm that would
  alleviate this.

  3. Adding texts to the trie can't be done "online". It requires the text strings to be known in advance to initialise values like edge array size etc
"""

occurence = tuple[int, int]


class UniqueEndCharacter:
  """UniqueEndCharacter is used to mark the end of a text.
  It allows us to create end markers for each text that are unique
  This uniqueness is important as it means that when the same substring is inserted from two different texts,
  the algorithm will create a leaf node for each unique end character, thus recording the occurence of the substring
  in both texts
  """

  def __init__(self, id_: int):
    # id is the text number, starts from 0
    self.id = id_

  def __eq__(self, other):
    # Per the spec, the input texts/patterns shouldn't have the $
    # character, so a str comparison will never be equal
    if isinstance(other, str):
      return False

    return self.id == other.id

  def __str__(self):
    return '$'

  def __repr__(self):
    return self.__str__()

  def get_id(self):
    return self.id


class GenericNode:
  """GenericNode
  We don't actually really need this, but I'm just going to keep it
  Class for Leaf and Node to inherit from
  """

  def __init__(self, is_leaf: bool = True):
    self.is_leaf = is_leaf

  def __repr__(self):
    return self.__str__()


class Leaf(GenericNode):
  """Class representing a leaf node
  """
  # Each leaf records the text that it is a part of. This just makes it
  # easier when we are trying to find matches
  text_index: int
  # Value is the classical leaf value, it is the index of the start of the substring that the path from the root to this leaf
  # within the entire string (the concatenated text strings) that is being
  # inserted
  value: int
  # True value is the index of the start of the substring that the path from the root to this leaf represents
  # within the text that it is being inserted from. It makes producing the
  # results of substring search much easier
  true_value: int

  def __init__(self, value: int, true_value: int, text_index: int = -1):
    super().__init__(True)
    self.value = value
    self.text_index = text_index
    self.true_value = true_value

  def __str__(self):
    return f'(Leaf {self.value} t{self.text_index})'


class Node(GenericNode):
  # label_num is just used to
  # label_num = 65
  def __init__(self, is_root: bool = False):
    self.is_root = is_root
    self.edge_list = EdgeList()
    self.suffix_link = None

    # Label is mainly used for debugging so that we can check which nodes are suffix linked to which other nodes
    # self.label = chr(Node.label_num)
    # Node.label_num += 1

    if self.is_root:
      self.suffix_link = self

  def add_leaf(
          self,
          edge_char: str,
          leaf_value: int,
          start_index: int,
          text_index: int,
          true_value: int) -> type[Leaf]:
    # end_index is not inclusive
    new_edge = self.edge_list.add_leaf_edge(edge_char)
    new_edge.set_start(start_index)
    new_leaf = Leaf(
        value=leaf_value,
        text_index=text_index,
        true_value=true_value)
    new_edge.set_next_node(new_leaf)

    return new_leaf

  def add_suffix_link(self, node: 'Node') -> 'Node':
    self.suffix_link = node

  def split_edge_at_index(
          self,
          edge: type["GenericEdge"],
          edge_first_char: str,
          new_node_first_char: str,
          split_index: int):
    # Create initial edge with index from
    new_node = Node()

    initial_edge = NodeEdge(
        start_index=edge.start_index,
        end_index=edge.get_index_at(split_index),
        next_node=new_node)
    edge.set_start(edge.get_index_at(split_index))

    self.edge_list.add_node_edge(edge_first_char, initial_edge)

    # Point the new middle node to the shortened old edge
    new_node.edge_list.add_node_edge(
        character=new_node_first_char, edge=edge)
    # index is the index we are splitting at
    # first character is the first character of the leaf edge?

    return new_node

  # This was mainly used for debugging suffix links
  # def __str__(self):
  # return f'(Node {self.label} Link_{self.suffix_link.label if
  # bool(self.suffix_link) else None} {self.edge_list})'

  def __str__(self):
    return f'(Node {self.label} Link_{bool(self.suffix_link)} {self.edge_list})'

  def __repr__(self):
    return self.__str__()


class GlobalEnd:
  """GlobalEnd
  As integer variables are passed by value, rather than reference, we need a GlobalEnd class to act as a pointer
  This allows all instances of the LeafEdge class to have a single reference to the global end

  The numerical python operations have been implemented as well so that this can be used like a normal number
  """

  def __init__(self, value=0):
    self.value = value

  def increment(self):
    self.value += 1

  def reset(self):
    self.value = 0

  def __add__(self, num: int):
    return self.value + num

  def __sub__(self, num: int):
    return self.value - num

  def __rsub__(self, num: int):
    return num - self.value

  def __radd__(self, num: int):
    return self.value + num

  def __lt__(self, num: int):
    return self.value < num

  def __le__(self, num: int):
    return self.value <= num

  def __eq__(self, num: int):
    return self.value == num

  def __ne__(self, num: int):
    return self.value != num

  def __gt__(self, num: int):
    return self.value > num

  def __ge__(self, num: int):
    return self.value >= num

  def __str__(self):
    return f'{self.value}'

  def __repr__(self):
    return self.__str__()


class GenericEdge:
  """GenericEdge
  Class representing a generic edge
  Node Edge and Leaf Edge inherit from this class

  """

  def __init__(
          self,
          start_index: int = 0,
          end_index: int = 0,
          next_node: type[GenericNode] = None):
    # start_index is inclusive and end_index is exclusive
    self.start_index = start_index
    self.end_index = end_index
    self.next_node = next_node

  def set_start(self, start_index: int):

    self.start_index = start_index

    return self

  def set_end(self, end_index: int):
    self.end_index = end_index
    return self

  def set_next_node(self, node: type[GenericNode]):
    self.next_node = node
    return self

  def get_next_node(self):
    return self.next_node

  def get_length(self):
    return self.end_index - self.start_index

  def get_index_at(self, index):
    return self.start_index + index

  def __repr__(self):
    return self.__str__()


class LeafEdge(GenericEdge):
  """LeafEdge
    Represents an edge that leads to a leaf
  """
  # We want each LeafEdge to share a pointer to the same GlobalEnd
  global_end = GlobalEnd()

  def __init__(
          self,
          start_index: int = 0,
          next_node: type[GenericNode] = None):
    # We initialise the end_index as the GlobalEnd object. Since this is an
    # object, the end_index will be a pointer
    super().__init__(
        start_index=start_index,
        end_index=self.global_end,
        next_node=next_node)

  def get_length(self):
    return self.end_index.value - self.start_index

  def __str__(self):
    return f'L_edge {self.start_index + 1}..{self.end_index} -> {self.next_node}'


class NodeEdge(GenericEdge):
  """NodeEdge
  Represents an edge leading up to a Node
  """

  def __init__(
          self,
          start_index: int = 0,
          end_index: int = 0,
          next_node: type[GenericNode] = None):
    super().__init__(
        start_index=start_index,
        end_index=end_index,
        next_node=next_node)

  def __str__(self):
    return f'N_edge {self.start_index + 1}..{self.end_index} -> {self.next_node}'

  def set_start(self, start_index: int):
    self.start_index = start_index


class EdgeList:
  """EdgeList

  Represents a list of edges outgoing from a node
  Uses a fixed size array self.array to store the edges

  For an index i smaller than the alphabet size 128, the edge at self.array[i] will start with the character chr(i)
  That is, index 65 will be the character A

  Our array is extended to store edges that will have the UniqueEndCharacter, self.array[128] will store $0, self.array[129] will store $1 etc..

  """
  alphabet_start_decimal = 0
  # The end decimal is exclusive
  alphabet_end_decinmal = 128
  alphabet_size = alphabet_end_decinmal - alphabet_start_decimal
  # This implementation to store the terminal characters in the edge
  num_terminal_characters = 0

  def __init__(self):
    self.array = [
        None for character in range(
            self.alphabet_size +
            self.num_terminal_characters)
    ]

  def add_leaf_edge(self, character: str):
    new_edge = LeafEdge()
    index = self.__get_index_for_char(character)
    self.array[self.__get_index_for_char(character)] = new_edge

    return new_edge

  def add_node_edge(self, character: str, edge: type[NodeEdge]):
    index = self.__get_index_for_char(character)
    self.array[index] = edge

    return edge

  def get_edge_for_character(self, character: str) -> type[GenericEdge]:
    return self.array[self.__get_index_for_char(character)]

  def get_edge_length_for_character(self, character: str):
    edge = self.array[self.__get_index_for_char(character)]
    if edge is not None:
      return edge.get_length()

    return 0

  def __get_index_for_char(self, character: str | type[UniqueEndCharacter]):
    # If the character is a UniqueEndCharacter, then we will need to handle
    # it specially
    if isinstance(character, UniqueEndCharacter):
      return self.__get_end_character_index(character)

    return ord(str(character)) - (self.alphabet_start_decimal + 1)

  def __get_end_character_index(self, character: type[UniqueEndCharacter]):
    # Since UniqueEndCharacters are stored at the end of the array, we can just use the UniqueEndCharacter id as a "shift" from the end of the ascii alphabet
    # portion of the array
    shift = character.get_id()

    return self.alphabet_size + shift

  def __str__(self):
    serialised = [edge for edge in self.array if edge is not None]

    return f'edges={serialised}'

  def __repr__(self):
    return self.__str__()

  # This function is used for the DFS traversal of the trie to find matches
  def get_non_null_edges(self):
    return [edge for edge in self.array if bool(edge)]

  @classmethod
  def set_num_terminal_characters(cls, num_terminal_characters: int):
    cls.num_terminal_characters = num_terminal_characters


class Trie:
  """Trie
  Represents a Trie by storing the root node
  """
  text: str

  def __init__(self):
    self.root_node = Node(True)

  def get_root_node(self):
    return self.root_node

  def set_text(self, text: str): self.text = text

  def __str__(self):
    return self.root_node.__str__()

  def __repr__(self):
    return self.__str__()


class Solution:
  def __init__(self, text_strings: list[str], pattern_strings: list[str]):
    self.text_strings = text_strings
    self.pattern_strings = pattern_strings
    self.trie = Trie()

  def run(self):
    trie_builder = TrieBuilder()
    [trie, processed_text] = trie_builder.add_text(self.text_strings)
    trie_searcher = TrieSearcher(trie=trie, text=processed_text)
    results = [trie_searcher.find(pattern)
               for pattern in self.pattern_strings]

    return results


text_index = int
true_value = int
result = (text_index, true_value)


class TrieSearcher:
  """TrieSearcher
  This class is used to search for a string (pattern) within the texts
  """
  trie: type[Trie]
  text: [str]

  def __init__(
      self, trie: type[Trie], text: [
          str | type[UniqueEndCharacter]]):
    self.trie = trie
    self.text = text

  def find(self, pattern: str) -> [result]:
    """find
    This function traverses down the tree character by character of the pattern
    If it isn't able to reach the end of the pattern (i.e the pattern doesn't exist in the Trie), it returns false
    Once we reach the end of the pattern, it does a BFS to find all leaf nodes that are children of the node it is currently at

    Args:
        pattern (string): patterntern we are looking for

    Returns:

    """
    node = self.trie.get_root_node()
    pattern_char_pointer = 0
    text_length = len(self.text)
    pattern_length = len(pattern)
    edge_char_pointer = 0
    while pattern_char_pointer < pattern_length:
      edge = node.edge_list.get_edge_for_character(
          pattern[pattern_char_pointer])
      if not bool(edge):
        return []
      edge_char_pointer = 0
      edge_length = edge.get_length()
      while edge_char_pointer < edge_length and pattern_char_pointer < pattern_length:
        index = edge.get_index_at(edge_char_pointer)
        if self.text[index] != pattern[pattern_char_pointer]:
          return []
        pattern_char_pointer += 1
        edge_char_pointer += 1
      # We've reached the end of an edge
      # Check if we have reached the end of our patterntern
      if pattern_char_pointer >= pattern_length:
        return self.collect_terminals(edge)
      else:
        node = edge.get_next_node()

    return self.collect_terminals(edge)

  # This function does a DFS to find leaves, and then adds the data stored
  # on the leaves to a results array
  def collect_terminals(self, edge: type[GenericEdge()]) -> [result]:
    # start looking from the edge_char_pointer
    results = []
    stack = []
    stack.append(edge.next_node)
    while len(stack) > 0:
      current = stack.pop()
      if isinstance(current, Leaf):
        results.append((current.text_index, current.true_value))
        # The results are added as an array [[text_index, true_value]]
        # where text_index is the zero indexed position of the text within the texts
        # and true_value is the true (zero indexed) index value of the
        # result within the text itself
      else:
        for edge in current.edge_list.get_non_null_edges():
          stack.append(edge.next_node)

    return results


class TrieBuilder:
  """TrieBuilder

  Class used to build a trie given an input string
  """

  # Stores the trie that we are building
  trie: type[Trie]
  # Stores the global end instance that the LeafEdges are pointing to
  global_end = type[GlobalEnd]
  # last_node stores the last active node so that we can suffix link to it
  # in the next iteration
  last_node: type[GenericNode]
  # pending_last_node is used to set the last_node after we have resolved suffix links.
  # This is because we can figure out what the last_node should be at the make_extension portion of the algorithm,
  # however, we don't want to set the last_node immediately, as we need to first resolve suffix links. This temp storage for last
  # node allows us to do this
  pending_last_node: type[GenericNode]
  # The length of the text
  text_length: int
  # The texts, broken down into chars, with the UniqueEndCharacters included
  # It looks something like {text1}${text2}$...
  text: [str | type[UniqueEndCharacter]]
  # This is the node below which we make extensions
  active_node: type[Node]
  # Remainder start and end index (end index is exclusive) help us determine where to make the extension
  # these indexes refer to indexes of self.text
  remainder_start_index: int
  remainder_end_index: int
  # Concat here means that we are referring to the "supertext" that is concatenated together in self.text
  # concat_next_text_index is the index of self.text when we move onto the
  # next text
  concat_next_text_index: int
  # concat_current_text_index is the index of the start of the current text within self.text
  # it helps us to get the "true_value" for the leaves, i.e the position of the substring within the text itself,
  # not the larger concatenated text
  concat_current_text_index: int
  # text_counter keeps track of the number of texts are being inserted
  text_counter: int
  # current_text stores the index of the current text within the texts whose
  # suffix is being inserted. It is 0 indexed (the 1st input text is a 0th
  # text here)
  current_text: int

  def __init__(self):
    self.text = []
    # We want to set global_end to the GlobalEnd instance that all the
    # LeafEdges are pointing to
    self.global_end = LeafEdge.global_end
    self.last_node: GenericNode = None
    self.pending_last_node = None
    self.text_counter = 0
    self.current_text = 0
    self.text_lengths = []

  def prepare_text(self, texts: [str]):
    """prepare_text

    Args:
        texts ([str]): a list of texts to be inserted into the trie
    """
    for text in texts:
      for char in text:
        self.text.append(char)
      # Append the unique end character
      self.text.append(UniqueEndCharacter(self.text_counter))
      self.text_lengths.append(len(text))
      self.text_counter += 1

    # We set the number of terminal characters so that EdgeList knows the
    # size of the array to initialise
    EdgeList.set_num_terminal_characters(self.text_counter)

  def add_text(self, texts: [str]):
    # prepare_text has to be run before any other variable that includes a
    # node is initialised as it sets the EdgeList array length that Nodes
    # rely on
    self.prepare_text(texts)

    self.trie = Trie()

    self.trie.set_text(self.text)

    self.text_length = len(self.text)

    self.last_j = 0

    # The active node is the node we are currently at
    self.active_node = self.trie.get_root_node()

    # The remainder tells us the string below the active node we need to
    # get to to make the extension for the current i interation
    self.remainder_start_index = 0
    self.remainder_end_index = 0

    # These variables help us track which texts we are currently inserting
    self.current_text = 0

    # Current text index tells us the start index of the current text we
    # are adding within the concatenated text
    self.concat_current_text_index = 0

    # Next text index tracks the index of the next text in the concatenated text
    # We add one to account for the dollar sign
    self.concat_next_text_index = self.text_lengths[0] + 1

    # construct I_1
    self.trie.get_root_node().add_leaf(
        edge_char=self.text[0],
        leaf_value=0,
        start_index=0,
        text_index=self.current_text,
        true_value=self.get_leaf_true_value(0))
    self.global_end.increment()

    for index in range(1, self.text_length):
      # Starting at self.last_j means that we skip past 0 .. last_j
      # iterations
      self.global_end.increment()
      self.calibrate_text(index)
      for j in range(self.last_j + 1, index + 1):
        # Skip count to the extension point
        self.traverse()

        # Make the extension
        # make_extension also returns True/False depending on whether
        # the extension was a rule 2 extension or not
        is_rule_2_extension = self.make_extension(index, j)

        # Resolve any pending suffix links from the previous extension
        self.resolve_suffix_links()

        # Move to the next extension, i.e by traversing suffix links,
        # or using showstopper
        self.move_to_next_extension(is_rule_2_extension, j)

        # If we aren't a rule 2 extension, then we are a rule 3
        # extension, in which case we use the showstopper rule
        if not is_rule_2_extension:
          break

    return [self.trie, self.text]

  def traverse(self):
    """traverse
    Skip counts down to the extension point, updating active_node and remainder
    """
    first_character = self.text[self.remainder_start_index]
    edge_length = self.active_node.edge_list.get_edge_length_for_character(
        first_character)

    # If edge_length is 0 that means that the character does not exist on
    # the edge at all
    while self.get_remainder_length() >= edge_length and edge_length > 0:
      self.shift_remainder_start_index_right(
          self.active_node.edge_list.get_edge_length_for_character(first_character))
      self.active_node = self.active_node.edge_list.get_edge_for_character(
          first_character).get_next_node()
      first_character = self.text[self.remainder_start_index]
      edge_length = self.active_node.edge_list.get_edge_length_for_character(
          first_character)

  def follow_suffix_links(self):
    if self.active_node.is_root:
      self.shift_remainder_start_index_right(1)
    self.active_node = self.active_node.suffix_link

  def make_extension(self, index: int, j: int) -> bool:
    """make_extension

    Args:
        index (int): The current index
        j (int): The current j value

    Returns:
        bool: True if rule 2, False otherwise
    """
    edge_first_char = self.text[index] if self.get_remainder_length(
    ) == 0 else self.text[self.remainder_start_index]

    character = self.text[index]

    edge: GenericEdge = self.active_node.edge_list.get_edge_for_character(
        edge_first_char)

    remainder_length = self.get_remainder_length()

    # If our remainder is '' and there is no edge beginning with that
    # character, we want to create a new leaf from the node directly

    if remainder_length == 0 and edge is None:
      self.active_node.add_leaf(
          edge_char=character,
          leaf_value=j,
          start_index=index,
          text_index=self.current_text,
          true_value=self.get_leaf_true_value(j))
      return True
    else:
      existing_character_at_extension_point = self.text[edge.get_index_at(
          remainder_length)]
      # Otherwise, if the edge does exist, then we want to compare the character at the extension point
      # If the character matches, its a rule 3
      if character == existing_character_at_extension_point:
        # Set the remainder values to the substring (remainder) that we
        # looked at
        self.set_remainder_start_index(edge.get_index_at(0))
        self.set_remainder_end_index(
            edge.get_index_at(remainder_length) + 1)

        return False

      # If the character doesn't match, its a rule 2 case 2
      if remainder_length != 0:
        # Split the edge and make the new node
        new_node = self.active_node.split_edge_at_index(
            edge=edge,
            edge_first_char=edge_first_char,
            new_node_first_char=existing_character_at_extension_point,
            split_index=self.get_remainder_length())

        # Add the new leaf to this new node
        new_node.add_leaf(
            edge_char=character,
            leaf_value=j,
            start_index=index,
            text_index=self.current_text,
            true_value=self.get_leaf_true_value(j))

        self.pending_last_node = new_node
        # If we just made a new node, we want to connect the last node
        # to this new one immediately
        if self.last_node:
          self.last_node.add_suffix_link(new_node)
          self.last_node = None

        return True

  def resolve_suffix_links(self):
    """resolve_suffix_links
    Resolves the suffix_links that are pending
    """
    if bool(self.last_node):
      self.last_node.add_suffix_link(self.active_node)
      self.last_node = None

  def get_remainder_length(self):
    return self.remainder_end_index - self.remainder_start_index

  def move_to_next_extension(self, is_rule_2_extension: bool, j: int):
    # Here we handle all the remainder index shifts
    # Set the last_node so that we can make suffix links in the next
    # iteration
    self.last_node = self.pending_last_node
    self.pending_last_node = None

    # If we made a rule_2 extension, then we want to set last_j, and also
    # follow the suffix link to the next node
    if is_rule_2_extension:
      self.last_j = j
      self.follow_suffix_links()

  def shift_remainder_start_index_right(self, amount: int):
    self.remainder_start_index = min(
        self.remainder_start_index + amount,
        self.remainder_end_index)

  def shift_remainder_end_index_left(self, amount: int):
    self.remainder_end_index = max(
        self.remainder_start_index,
        self.remainder_end_index - amount)

  def shift_remainder_end_index_right(self, amount: int):
    self.remainder_end_index = min(
        self.remainder_end_index + amount, len(self.text))

  def set_remainder_start_index(self, index: int):
    self.remainder_start_index = index

  def set_remainder_end_index(self, index: int):
    self.remainder_end_index = index

  def calibrate_text(self, index: int):
    # This function calibrates our text pointers
    # It makes sure that the current text, and next text index variables
    # are up to date
    if index == self.concat_next_text_index:
      self.current_text += 1
      # Make the current text index equal to the next text index
      self.concat_current_text_index = self.concat_next_text_index
      # Shift the next text index to the next value
      self.concat_next_text_index = self.concat_next_text_index + \
          self.text_lengths[self.current_text] + 1

  def get_leaf_true_value(self, index: int):
    """Given an index in the concatenated text, this function returns the index within the current text being scanned only
    """
    return index - self.concat_current_text_index


class FileHelper:
  """FileHelper helps with any file operations we need
  """

  def __init__(self, filename: str):
    with open(filename, "r") as f:
      self.file_contents = [line.rstrip('\n') for line in f.readlines()]

      self.number_text_files = int(self.file_contents[0])

      self.text_files = [line.split(
          " ")[1] for line in self.file_contents[1: self.number_text_files + 1]]

      self.number_pattern_files = int(
          self.file_contents[self.number_text_files + 1])

      self.pattern_files = [
          line.split(" ")[1]
          for line in self.file_contents[self.number_text_files + 2: self.number_text_files + self.number_pattern_files + 3]
      ]

  # For the text and pattern, we convert both to lower() so that we can
  # match case insensitively

  def get_text_files(self) -> list[str]:
    return [self.__open_and_read_all_lines_from_file(
        file).lower() for file in self.text_files]

  def get_pattern_files(self) -> list[str]:
    return [self.__open_and_read_line_from_file(
        file).lower() for file in self.pattern_files]

  def __open_and_read_line_from_file(self, filename: str) -> str:
    with open(filename, "r") as f:
      return f.readline()

  def __open_and_read_all_lines_from_file(self, filename: str) -> str:
    with open(filename, "r") as f:
      return "".join(f.readlines())

  def write_occurences_to_file(occurences: list[occurence]):
    with open('output_gst.txt', 'w') as f:
      for index in range(len(occurences)):
        # +1 as all the values I use in the algorithm are 0 indexed
        if (occurences[index]):
          for occurence in occurences[index]:
            f.write(
                f'{index + 1} {occurence[0] + 1} {occurence[1] + 1}\n')


def main(filename: str):
  file_parser = FileHelper(filename)

  text_strings = file_parser.get_text_files()

  pattern_strings = file_parser.get_pattern_files()

  results = Solution(text_strings, pattern_strings).run()

  FileHelper.write_occurences_to_file(results)


if __name__ == "__main__":
  if len(sys.argv[1:]) > 0:
    args = [arg for arg in sys.argv[1:]]

    main(args[0])

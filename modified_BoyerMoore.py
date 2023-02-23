"""Modified BoyerMoore

Given any text txt[1 ... n] and any pattern pat[1 ... m], your first task is to implement a modified 
version of the Boyer-Moore’s algorithm, and report the positions of all occurrences of pat in txt.

name: Kevin Yu
student_id: 306128531

"""

import sys

def compute_extended_bad_character_matrix(pattern: str) -> list[list[int]]:
  """computes the bad character matrix given a pattern
  This matrix is used to find the rightmost instance of a character given a position
  in the pattern

  Here we are assuming that the inputs are ASCII and within

  Args:
      pattern (str): the pattern to be processed
  """
  r_matrix = [[-1 for _ in range(128)] for _ in range(len(pattern))]
  for i in range(len(pattern) - 1):
    character = pattern[i]
    r_matrix[i+1] = r_matrix[i].copy()
    r_matrix[i+1][ord(character)] = i
  return r_matrix

def get_bad_character_value(matrix: list[list[int]], character: str, position: int) -> int:
  """helper function to get the index of a character given a bad character matrix and a position in pattern

  Args:
      matrix (list[list[int]]): bad character matrix
      character (str): character to be found
      position (int): position in pattern

  Returns:
      int: the position of the bad character
  """
  return matrix[position][ord(character)]

def compute_z_values(input_str: str) -> list[int]:
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
  output[1] = get_z_value_by_direct_comparison(input_str, 1)
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
      z_value = get_z_value_by_direct_comparison(input_str, i)
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
        z_value = get_z_value_by_direct_comparison(input_str, prefix_box_end_index, r + 1)
        output[i] = r - i + 1 + z_value
        l = i
  return output

def get_z_value_by_direct_comparison(input_str: str, index: int, input_index: int = 0) -> int:
  """function that computes z values given an input string and two indexes to compare
  """
  substring_index = index
  count = 0
  while input_index < len(input_str) and substring_index < len(input_str) and input_str[input_index] == input_str[substring_index]:
    input_index += 1
    substring_index += 1
    count += 1
  return count

def compute_z_suffix_values(input_str: str) -> str:
  """Computes the z suffix values
  """
  return compute_z_values(input_str[::-1])[::-1]

def compute_good_suffix_values(input_str: str) -> list[int]:
  """Computes the good suffix values

  Args:
      input_str (str): input string to compute good suffixes for

  Returns:
      list[int]: list of good suffixes
  """
  input_len = len(input_str)
  z_suffix_values = compute_z_suffix_values(input_str)
  good_suffix_values = [-1 for i in range(len(input_str) + 1)]
  for index, z_value in enumerate(z_suffix_values):
    if z_value > 0:
      good_suffix_values[input_len - z_value] = index + 1
  return good_suffix_values

def compute_mismatched_suffix_values(input_str: str) -> list[list[int]]:
  """Computes the mismatched suffix values for an input string

  Given a string input_str, let m be its computed mismatched_suffix_values

  With a character c and index between [0, len(input_str)],  m[index][ord(c)] returns the rightmost instance 
  of a matching substring that matches the suffix [index, len(input_str)], and is also preceeded by the character c
  """
  input_len = len(input_str)
  z_suffix_values = compute_z_suffix_values(input_str)
  mismatched_suffix_values = [[-1 for i in range(128)] for j in range(len(input_str) + 1)]

  for index, z_value in enumerate(z_suffix_values):
    # For each substring input_str[a,b] that matches a suffix input_str[x,y], we want to store 
    # the index a - 1 (index - z_value), the character after the leftmost part of the substring
    # This index is stored at m[x][ord(input_str[index - z_value])], which allows O(1) access
    # This way, when we mismatch at the left end of a suffix (x), we can easily access the mismatched
    # suffix value. This allows us to shift per the modified shift rules in the assignment
    if z_value > 0 and index != 0:
      mismatched_suffix_values[input_len - z_value][ord(input_str[index - z_value])] = index - z_value
  return mismatched_suffix_values

def compute_matched_prefix_values(input_str: str) -> list[int]:
  input_len = len(input_str)
  
  matched_prefix_values = [-1 for i in range(len(input_str) + 1)]
  
  z_values = compute_z_values(input_str)

  current_largest_prefix_length = 0

  for index in range(len(z_values) - 1, -1, -1):
    z_value = z_values[index]
    if z_value + index == input_len:
      current_largest_prefix_length = z_value
    matched_prefix_values[index] = current_largest_prefix_length
  matched_prefix_values[0] = input_len
  return matched_prefix_values

def exec_Modified_BoyerMoore(txt: str, pattern: str):
  """executes the actual BoyerMoore algorithm given a pattern and a text to match against

  Args:
      txt (str): txt to scan for pat
      pattern (str): pat that is to be found in txt
  """

  # Preprocessing the pattern 
  extended_bad_character_matrix = compute_extended_bad_character_matrix(pattern)
  zSuffixes = compute_z_suffix_values(pattern)
  good_suffix_values = compute_good_suffix_values(pattern)
  mismatched_suffix_values = compute_mismatched_suffix_values(pattern)
  matched_prefix_values = compute_matched_prefix_values(pattern)

  pattern_len = len(pattern)
  txt_len = len(txt)

  # Used to store the results
  matches = []
  
  # As we will be iterating through the pattern/txt, these pointers are used to keep track of our location
  txt_end_pointer = pattern_len - 1
  pattern_pointer = pattern_len - 1

  # Used for Galil's optimisation to skip over parts of the pattern
  stop = -1 
  start = -1
  
  while txt_end_pointer < txt_len:
    txt_pointer = txt_end_pointer

  # Matching phase
    while txt_pointer < txt_len and pattern_pointer < pattern_len and txt[txt_pointer] == pattern[pattern_pointer]:
      # If we have reached the end of our pattern, we have found a match!
      if pattern_pointer == 0:
        matches.append(txt_pointer + 1)
        break
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
    bad_character_index = get_bad_character_value(extended_bad_character_matrix, mismatched_character, pattern_pointer)
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
      mismatched_suffix_value = mismatched_suffix_values[pattern_pointer + 1][ord(mismatched_character)]
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
  return matches

def open_and_read_line_from(file: str):
  with open(file, 'r', encoding = 'utf-8') as f:
    return f.read() 
  
def create_and_write_results_to(results: list[int], filename: str):
  with open(filename, 'w') as f:
    for match in results:
      f.write(f'{match}\n')

def modified_BoyerMoore(txt_file: str, pat_file: str):
  txt = open_and_read_line_from(txt_file)

  pat = open_and_read_line_from(pat_file)

  if len(txt) == 0 or len(pat) == 0:
    return 

  results = exec_Modified_BoyerMoore(txt, pat)

  create_and_write_results_to(results, "output_modified_BoyerMoore.txt")

if __name__ == "__main__":
  if len(sys.argv[1:]) > 0:
    args = [arg for arg in sys.argv[1:]]
    modified_BoyerMoore(args[0], args[1])
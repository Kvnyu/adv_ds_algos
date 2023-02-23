"""Hamming distance patmatch

Given some text txt[1..n] and a pattern pat[1..m], write a program to identify all positions within txt[1..n] that matches the pat[1..m] within a Hamming distance <= 1. 
The Hamming distance between two strings of the same length is the number of corresponding positions where the characters between the two strings disagree.

name: Kevin Yu
student_id: 306128531

"""

import sys

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

Match = (int, int)
Results = list[Match]

def patmatch(txt: str, pat: str) -> Results:
  """executes the actual patmatch 

  Args:
      txt (str): the txt to match against
      pat (str): the pattern to find in txt (with hamming distance <= 1)
  """
  # We assume that chr(0) NULL is an ASCII character that will not be provided in the input txt or pat
  # as it is a non-printable character (as per edstem posts)
  # Therefore it can effectively be used as a separator between pat and txt
  input_str = pat + chr(0) + txt
  # We compute z values forwards to see how many characters in a row match going forwards
  z_values = compute_z_values(input_str)
  # We compute z values of the reverse of pat and txt together to see how many characters in a row match going backwards
  reverse_input_str = pat[::-1] + chr(0) + txt[::-1] 
  reverse_z_values = compute_z_values(reverse_input_str)[::-1]

  results = []

  # Below we want to combine the results for z_value and reverse_z_values for txt (not pat)
  # For the z_value array, we want to start at the first char in txt, and for reverse_z_values we want to start at the len(pat) th character

  # The string corresponding to reverse_z_values looks like txt(NULL)pat 
  # The first char in reverse_z_values that we look at is len(pat) - 1. Since we are going from 1 in the for loop (for easier results recording)
  # this is reduced to len(pat) - 2
  reverse_z_value_starting_index = len(pat) - 2

  # The string corresponding to z_values looks like pat(NULL)txt
  # The first char in z_values that we look at is len(pat) + 1
  # Since we are going from 1 in the for loop, this is reduced to len(pat)
  z_value_starting_index = len(pat)

  for i in range(1, len(txt) - len(pat) + 2):
    reverse_z_value_index = reverse_z_value_starting_index + i
    z_value_index = z_value_starting_index + i

    # current_matches is how many chars between txt[i, i + len(pat)] match
    current_matches = reverse_z_values[reverse_z_value_index] + z_values[z_value_index]

    # since Hamming distance <= 1, we use len(pat) - 1 for the threshold for a match
    threshold = len(pat) - 1

    if current_matches == threshold:
      results.append((i, 1))

    # if we are above the threshold, it means that we have a a hamming distance of 0, since our threshold 
    # was hamming distance <= 1
    elif current_matches > threshold:
      results.append((i, 0))

  return results


def hd1_patmatch(txt_file: str, pat_file: str) :
  txt = open_and_read_line_from(txt_file)

  pat = open_and_read_line_from(pat_file)

  if len(txt) == 0 or len(pat) == 0:
    return 

  results = patmatch(txt, pat)

  create_and_write_results_to(results, "output_hd1_patmatch.txt")


def open_and_read_line_from(file: str):
  with open(file, 'r') as f:
    return f.read() 
  
def create_and_write_results_to(results: Results, filename: str):
  with open(filename, 'w') as f:
    for match in results:
      f.write("{position_in_txt} {hamming_distance}\n".format(position_in_txt = match[0], hamming_distance = match[1]))

if __name__ == "__main__":
  if len(sys.argv[1:]) > 0:
    args = [arg for arg in sys.argv[1:]]
    hd1_patmatch(args[0], args[1])
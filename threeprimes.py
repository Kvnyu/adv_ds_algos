"""threeprimes.py

Consider the following conjecture on integers:
Every odd number N > 7 is a sum of three odd prime numbers.
For example, when N = {9, 11, 13, 15, 17}, it is easy to see this conjecture holds

In this exercise you will be writing a computer program that verifies whether this conjecture holds for any given odd input N. If it holds, the program should output any valid set of three odd prime numbers (space separated) that add up to the input N. Otherwise, do not report anything (but you should doubt your program’s correctness).
Importantly, your implementation must use the Miller-Rabin algorithm for primality testing introduced in this unit.


name: Kevin Yu
student_id: 306128531

"""
import sys
from random import randrange

K_VALUE = 64

# This function is adapted from the lecture slides
# Returns true if prime, false otherwise
def miller_rabin_randomized_primality(n: int, k: int):
  """A non deterministic algorithm to check if a number n is prime.

  Args:
      n (int): a number n that is being checked for primeness or compositeness
      k (int): the number of times to repeat the witness loop. Higher k means more accuracy

  Returns:
      (bool): true if n is prime, false otherwise (though true is more like 'maybe' true)
  """
  if n < 2:
    return False

  if n == 2 or n == 3:
    return True

  if n % 2 == 0:
    return False

  s = 0
  t = n - 1

  while (not (t % 2)):
    s += 1
    t //= 2

  while k > 0:
    # Generate a random number between 2 and n-2
    a = randrange(2, n - 1)
    if (not is_congruent(a**(n-1), 1, n)):
      return False

    # Calculate our initial value for prev
    prev = modular_exponentiation(a, t, n)
    for i in range(1, s + 1):
      # Keep the current mod_exp as we can use it next iteration as 'prev'
      mod_exp = modular_exponentiation(a, t * (2 ** i), n)
      first_condition = modular_exponentiation(a, t * (2 ** i), n) == 1 % n
      second_condition = prev == 1 % n
      third_condition = prev == -1 % n

      if (first_condition and not (second_condition or third_condition)):
        return False
      prev = mod_exp
    k -= 1
  return True


def is_congruent(a: int, b: int, n: int):
  return (a - b) % n == 0


def is_prime(n: int) -> bool:
  return miller_rabin_randomized_primality(n, K_VALUE)

# This function is adapted from Taylor's notes
# This function calculates a^b mod n efficiently
def modular_exponentiation(a: int, b: int, n: int):
  # base case
  exponent = b
  current = a % n
  result = 1

  if exponent & 1:
    result = current

  exponent >>= 1

  while exponent:
    current = (current * current) % n

    if exponent & 1:
      result = (result * current) % n

    exponent >>= 1

  return result

def pregenerate_primes(n: int):
  # We pregerate all primes up to the input number n so that we can efficiently retrieve and iterate through the prime numbers to build the number n
  # The prime bitarray allows O(1) checking for whether a given number is prime
  prime_bitarray = [0 for i in range(n)]
  # The primes list is a list containing all primes < n
  primes = []
  for i in range(n):
    if is_prime(i):
      prime_bitarray[i] = 1
      primes.append(i)
  return (prime_bitarray, primes)


output = [int, int, int]

def verify_goldbach_weak_conjecture(n: int) -> output:
  # Precalculate all primes up to n, get the bitarray and list of primes
  (prime_bitarray, primes) = pregenerate_primes(n)
  num_primes = len(primes)
  # For each prime
  for i in range(num_primes):
    # Get the first prime number
    prime_1 = primes[i]
    # For each prime larger than or equal to this prime number in the list of primes
    for j in range(i, num_primes):
      # Get the second prime number
      prime_2 = primes[j]
      maybe_prime_3 = n - prime_1 - prime_2
      # Check if n - prime_1 - prime_2 is a prime number using our prime bitarray
      if prime_bitarray[maybe_prime_3]:
        prime_3 = maybe_prime_3
        # If it is prime, then return the three primes
        return [prime_1, prime_2, prime_3]
  raise Exception("It is more probable that all oxygen molecules in my home would conspire against me and move away from me into a corner than this happening ... but it has?")


class FileHelper:
  @staticmethod
  def write_to_file(output: output, filename: str):
    with open(filename, "w") as f:
      f.write(" ".join(map(str, output)))

def main(n: str):
  n = int(n)
  output = verify_goldbach_weak_conjecture(n)

  # Note that this will output the file to the directory that you call this python script in
  filename = "output_threeprimes.txt"
  FileHelper.write_to_file(output, filename)


if __name__ == "__main__":
  if len(sys.argv[1:]) > 0:
    args = [arg for arg in sys.argv[1:]]
    main(args[0])

import random
import time
import matplotlib.pyplot as plt
from functools import cmp_to_key
from math import log2

# Sorting algorithms (same as before)
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# def heap_sort(arr):
#     heapq.heapify(arr)
#     return [heapq.heappop(arr) for _ in range(len(arr))]

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[i] < arr[left]:
        largest = left

    if right < n and arr[largest] < arr[right]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        merge_sort(left)
        merge_sort(right)

        i = j = k = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

def insertion_sort(arr, left=0, right=None):
    if right is None:
        right = len(arr) - 1

    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1



def timsort(arr):
    MIN_MERGE = 32

    def insertion_sort(arr, left, right):
        for i in range(left + 1, right + 1):
            j = i
            while j > left and arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                j -= 1

    def merge(arr, left, mid, right):
        len_left, len_right = mid - left + 1, right - mid
        left_copy, right_copy = arr[left:mid + 1], arr[mid + 1:right + 1]

        i = j = 0
        k = left

        while i < len_left and j < len_right:
            if left_copy[i] <= right_copy[j]:
                arr[k] = left_copy[i]
                i += 1
            else:
                arr[k] = right_copy[j]
                j += 1
            k += 1

        while i < len_left:
            arr[k] = left_copy[i]
            i += 1
            k += 1

        while j < len_right:
            arr[k] = right_copy[j]
            j += 1
            k += 1

    def timsort_util(arr, left, right):
        if right - left + 1 < MIN_MERGE:
            insertion_sort(arr, left, right)
        else:
            mid = (left + right) // 2
            timsort_util(arr, left, mid)
            timsort_util(arr, mid + 1, right)
            merge(arr, left, mid, right)

    timsort_util(arr, 0, len(arr) - 1)

def radix_sort(arr):
    # Radix sort for positive integers
    if not arr:
        return arr
    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= 10
    return arr

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i]

def bucket_sort(arr):
    if not arr:
        return arr
    min_val, max_val = min(arr), max(arr)
    bucket_size = max(1, (max_val - min_val) // len(arr))
    buckets = [[] for _ in range((max_val - min_val) // bucket_size + 1)]

    for num in arr:
        buckets[(num - min_val) // bucket_size].append(num)

    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(sorted(bucket))

    return sorted_arr

# def timsort(arr):
#     return sorted(arr)

# Function to measure execution time
def measure_time(algorithm, input_array):
    start_time = time.time()
    algorithm(input_array)
    end_time = time.time()
    return end_time - start_time

# Function to compare sorting algorithms for a specific input size
def compare_algorithms_for_input(algorithms, input_type, size, k=None):
    plt.figure(figsize=(14, 6))

    x_values = [algorithm.__name__ for algorithm in algorithms]
    y_values = []

    for algorithm in algorithms:
        input_array = generate_input(input_type, size, k)
        time_taken = measure_time(algorithm, input_array)
        y_values.append(time_taken)

    plt.bar(x_values, y_values, label=f"Size: {size}")

    plt.title(f"Sorting Algorithm Comparison - {input_type} - Size: {size}")
    plt.xlabel("Sorting Algorithm")
    plt.ylabel("Time Taken (seconds)")
    plt.legend()
    plt.show()

# Function to generate various types of input arrays
def generate_input(input_type, size, k=None):
    if input_type == "random":
        return [random.randint(0, size) for _ in range(size)]
    elif input_type == "limited_range":
        return [random.randint(0, k) for _ in range(size)]
    elif input_type == "cubed_range":
        return [random.randint(0, size**3) for _ in range(size)]
    elif input_type == "log_range":
        return [random.randint(0, int(log2(size))) for _ in range(size)]
    elif input_type == "multiples_of_1000":
        return [random.randint(0, n) * 1000 for n in range(size)]
    elif input_type == "partially_sorted":
        arr = list(range(size))
        log_n_half = int(log2(size) / 2)
        for _ in range(log_n_half):
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    # Add other input types as needed

# Function to compare sorting algorithms for each input size on the same graph
def compare_algorithms_for_all_sizes(algorithms, input_type, sizes, k=None):
    plt.figure(figsize=(14, 6))

    for size in sizes:
        x_values = [algorithm.__name__ for algorithm in algorithms]
        y_values = []

        plotting_points = []

        for algorithm in algorithms:
            input_array = generate_input(input_type, size, k)
            time_taken = measure_time(algorithm, input_array)
            y_values.append(time_taken)

            plotting_points.append((algorithm.__name__, time_taken))

        print(f"Size: {size}")
        print("Algorithm\tTime Taken")
        for x, y in plotting_points:
            print(f"{x}\t\t{y:.4f}")

        plt.bar(x_values, y_values, label=f"Size: {size}")

    plt.title(f"Sorting Algorithm Comparison - {input_type}")
    plt.xlabel("Sorting Algorithm")
    plt.ylabel("Time Taken (seconds)")
    plt.legend()
    plt.show()

# Example usage
algorithms = [quick_sort, heap_sort, merge_sort, radix_sort, bucket_sort, timsort]
input_types = ["random", "cubed_range", "limited_range","log_range", "multiples_of_1000", "partially_sorted"]
sizes = [1000000, 900000, 700000, 500000, 200000, 600000]  # Adjust sizes for a larger input
k = 1000  # Adjust k as needed

# Compare algorithms for each input size on the same graph
compare_algorithms_for_all_sizes(algorithms, input_types[0], sizes, k)


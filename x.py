def remove_lines_starting_with_set(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Filter out lines that start with "Set"
    filtered_lines = [line for line in lines if not line.startswith("Set")]

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)

# Example usage
input_file = "xx"   # Replace with your input file
output_file = "output.txt" # Replace with your desired output file
remove_lines_starting_with_set(input_file, output_file)

import os

# Specify the directory containing the text files
directory = 'data'

# Specify the output file
output_file = 'dataset.txt'

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a text file
        if filename.endswith('.txt'):
            # Open the file and read its contents
            with open(os.path.join(directory, filename), 'r') as infile:
                # Write the contents to the output file
                outfile.write(infile.read())

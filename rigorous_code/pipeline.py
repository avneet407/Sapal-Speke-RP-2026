# Python file in which functions called

import handling_data as hd

# SETTINGS
# Set the path to the expression matrix file
MATRIX_PATH = "rigorous_code/data/expression_matrix.tsv"

# Load the expression matrix
expression_matrix = hd.load_matrix(MATRIX_PATH)
# Tom coding everything from scratch using only in text suggestions instead of chat prompts.

# Import relevant libraries
import pandas as pd

# Function which loads expression matrix data from a file
def load_matrix(path):
    """Rows = isoforms (transcript IDs), columns = samples, values = CPM."""

    print(f"Loading expression matrix from {path}...")
    
    # sep="\t"     -> the file is tab-separated (.tsv)
    # index_col=0  -> use the first column (transcript_id) as row names, not as data
    df = pd.read_csv(path, sep="\t", index_col=0)
    
    print(df.head())
    
    return df
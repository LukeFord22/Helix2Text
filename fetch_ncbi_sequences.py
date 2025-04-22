import pandas as pd
from Bio import Entrez

# Set your email for NCBI Entrez API
Entrez.email = "your_email@example.com"

def fetch_dna_sequences_from_ncbi(query, quantity):
    """
    Fetch a specific quantity of DNA sequences from NCBI along with their textual information.

    Args:
        query (str): The search query for NCBI (e.g., "Homo sapiens[ORGN] AND BRCA1").
        quantity (int): The number of sequences to fetch.

    Returns:
        pd.DataFrame: A DataFrame with columns 'sequence' and 'text'.
    """
    try:
        # Search NCBI for the query
        print(f"Searching NCBI for query: {query}")
        handle = Entrez.esearch(db="nucleotide", term=query, retmax=quantity)
        record = Entrez.read(handle)
        handle.close()

        # Get the list of IDs
        ids = record["IdList"]
        print(f"Found {len(ids)} sequences. Fetching details...")

        sequences = []
        texts = []

        # Fetch details for each ID
        for seq_id in ids:
            handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="gb", retmode="text")
            data = handle.read()
            handle.close()

            # Parse the GenBank file
            from Bio import SeqIO
            from io import StringIO

            record = SeqIO.read(StringIO(data), "genbank")
            sequences.append(str(record.seq))
            texts.append(data)

        # Create a DataFrame
        df = pd.DataFrame({"sequence": sequences, "text": texts})
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    query = "Homo sapiens[ORGN] AND BRCA1"
    quantity = 20
    df = fetch_dna_sequences_from_ncbi(query, quantity)

    # Save to CSV
    output_file = "ncbi_search.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} sequences to {output_file}")
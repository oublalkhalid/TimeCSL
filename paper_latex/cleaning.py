import bibtexparser
import hashlib

# Function to read the .bib file
def read_bibtex_file(file_path):
    with open(file_path, 'r') as file:
        bibtex_str = file.read()
    return bibtex_str

# Parse bibtex entries
def parse_bibtex_entries(bibtex_str):
    bib_database = bibtexparser.loads(bibtex_str)
    return bib_database.entries

# Hash the entry content to identify duplicates based on content
def hash_entry(entry):
    entry_str = "".join(f"{field}={entry[field]}" for field in sorted(entry.keys()) if field not in ['ENTRYTYPE', 'ID'])
    return hashlib.md5(entry_str.encode('utf-8')).hexdigest()

# Remove duplicates based on content hash
def remove_duplicates(entries):
    seen_hashes = set()  # Set to store already seen content hashes
    unique_entries = []
    
    for entry in entries:
        entry_hash = hash_entry(entry)
        if entry_hash not in seen_hashes:
            unique_entries.append(entry)
            seen_hashes.add(entry_hash)
    
    return unique_entries

# Clean the bibtex entries by removing duplicates based on citation key or content
def clean_bibtex_file(input_path, output_path):
    # Read and parse the .bib file
    bibtex_str = read_bibtex_file(input_path)
    entries = parse_bibtex_entries(bibtex_str)
    
    # Remove duplicates
    unique_entries = remove_duplicates(entries)
    
    # Convert cleaned entries back to BibTeX format
    cleaned_bibtex = "".join([format_bibtex_entry(entry) for entry in unique_entries])
    
    # Write the cleaned BibTeX data to the output file
    with open(output_path, 'w') as output_file:
        output_file.write(cleaned_bibtex)
    
    print(f"Cleaned BibTeX entries saved to {output_path}")

# Convert a bibtex entry dictionary back into string format
def format_bibtex_entry(entry):
    """ Convert a bibtex entry dictionary back into string format """
    entry_str = f"@{entry['ENTRYTYPE']}{{{entry['ID']},\n"
    for field, value in entry.items():
        if field not in ['ENTRYTYPE', 'ID']:
            entry_str += f"  {field} = {{{value}}},\n"
    entry_str = entry_str.rstrip(',\n') + "\n}\n"
    return entry_str
    
# Example usage
if __name__ == "__main__":
    # Define the input and output .bib file paths
    input_bib_path = 'references.bib'  # Path to your input .bib file
    output_bib_path = '/tsi/data_education/Ladjal/koublal/TimeCSL/TimeCSL/paper_latex/cleaned_references.bib'  # Path where cleaned .bib will be saved

    # Clean the .bib file by removing duplicates based on citation keys
    clean_bibtex_file(input_bib_path, output_bib_path)

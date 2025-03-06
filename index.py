import os
import pefile
import pandas as pd

def extract_pe_features(file_path):
    """Extract static features from a PE (Portable Executable) file."""
    try:
        pe = pefile.PE(file_path)
        features = {
            "file_size": os.path.getsize(file_path),
            "num_sections": len(pe.sections),
            "entry_point": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            "image_base": pe.OPTIONAL_HEADER.ImageBase,
            "num_imports": sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,
            "num_exports": len(pe.DIRECTORY_ENTRY_EXPORT.symbols) if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT') else 0,
            "has_debug": hasattr(pe, 'DIRECTORY_ENTRY_DEBUG'),
        }
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_directory(directory):
    """Process all PE files in a directory and extract features."""
    data = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".exe"):  # Process only executable files
            features = extract_pe_features(file_path)
            if features:
                features["file_name"] = file_name
                data.append(features)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    benign_dir = "../data/benign_samples"  # Adjust path as needed
    malware_dir = "../data/malware_samples"

    benign_df = process_directory(benign_dir)
    malware_df = process_directory(malware_dir)

    benign_df["label"] = 0  # Benign label
    malware_df["label"] = 1  # Malware label

    dataset = pd.concat([benign_df, malware_df], ignore_index=True)
    dataset.to_csv("../data/malware_dataset.csv", index=False)

    print("Feature extraction complete! Dataset saved as malware_dataset.csv")
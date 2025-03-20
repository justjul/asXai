import config
import os
import pandas as pd

def load_dataset(dataset_source='semanticsscholar', years=None, data_types=None):
    """Fetch dataset metadata and text data for given years and data types."""

    # Normalize inputs
    if years is None:
        years = [year for year in os.listdir(config.METADATA_PATH) if year.isdigit()]
    elif isinstance(years, int) or isinstance(years, str):
        years = [str(years)]
    else:
        years = [str(year) for year in years]

    data_types = data_types or ['metadata', 'text']
    data_types = [data_types] if isinstance(data_types, str) else data_types

    # Initialize loaders
    data_map = {
        "metadata": (config.METADATA_PATH, []),
        "text": (config.TEXTDATA_PATH, []),
        "pdf": (config.PDFDATA_PATH, [])
    }

    for year in years:
        for dtype in data_types:
            path = data_map.get(dtype, (None, None))[0]
            if not path:
                continue

            file_name = f'{dataset_source}_{"textpdf" if dtype == "pdf" else dtype}_{year}.parquet'
            file_path = path / year / file_name

            # fallback for missing pdf parquet
            if dtype == "pdf" and not file_path.exists():
                file_path = config.TEXTDATA_PATH / year / f'{dataset_source}_text_{year}.parquet'
                print(f"Fallback: no PDF data for {year}, loading text data instead.")

            if file_path.exists():
                df = pd.read_parquet(file_path, engine="pyarrow")
                data_map[dtype][1].append(df)
            else:
                print(f"No {dtype} data for year {year}")

    # Concatenate and clean outputs
    output = []
    if 'metadata' in data_types:
        metadata = pd.concat(data_map['metadata'][1], ignore_index=True) if data_map['metadata'][1] else pd.DataFrame()
        output.append(metadata)
    if 'text' in data_types or 'pdf' in data_types:
        textdata = pd.concat(data_map.get('text')[1], ignore_index=True) if data_map.get('text')[1] else pd.DataFrame()
        pdfdata = pd.concat(data_map.get('pdf')[1], ignore_index=True) if data_map.get('pdf')[1] else pd.DataFrame()
        if not pdfdata.empty:
            textdata = pdfdata.set_index("paperId").combine_first(textdata.set_index("paperId")).reset_index(drop=False)
        output.append(textdata)

    output = tuple(output) if output and len(output) > 1 else output[0]

    return output

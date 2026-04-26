import pandas as pd

TEXT_LENGTH_THRESHOLD = 50 #identifying a columns as text if the average string len is more thna 50 chars, leaving short categories (Basically filtering)


def detect_columns(df: pd.DataFrame, target: str = 'fraudulent') -> dict:
  
    #I have grouped columns into different types such as binary, text and categorical based on their values
    #dictionary is returned with keys of respective data types
    text, binary, categorical, id_cols = [], [], [], []

    for col in df.columns:
        if col == target:
            continue

        if col.lower().endswith('_id') or col.lower() == 'id': #skipping id columns
            id_cols.append(col)
            continue

        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            #if found values {0,1}, it's a binary
            uniques = set(series.dropna().unique())
            if uniques.issubset({0, 1, 0.0, 1.0}):
                binary.append(col)
            else:
                
                categorical.append(col) #other numeric columns will be treated as categorical
        else:
            #string column, checks long text or short category
            non_null = series.dropna()
            avg_len = non_null.astype(str).str.len().mean() if len(non_null) else 0
            if avg_len > TEXT_LENGTH_THRESHOLD:
                text.append(col)
            else:
                categorical.append(col)

    return {
        'text': text,
        'binary': binary,
        'categorical': categorical,
        'target': target,
        'id': id_cols,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_dataset

    df = load_dataset()
    cols = detect_columns(df)
    for kind, names in cols.items():
        print(f"{kind}: {names}")

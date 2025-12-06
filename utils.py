# utils.py
import io
import pandas as pd

def df_to_csv(df):
    """
    Return CSV as UTF-8 encoded bytes for Streamlit download_button.

    - Handles None / empty DataFrame
    - Converts datetime index to a 'timestamp' column if present
    - Normalizes datetime columns to ISO strings to avoid serialization issues
    """
    if df is None:
        return b""

    # if it's a pandas DataFrame and empty, return empty bytes
    if isinstance(df, pd.DataFrame) and df.empty:
        return b""

    # work on a copy to avoid mutating original
    try:
        df_copy = df.copy()
    except Exception:
        # if df isn't a DataFrame (unlikely), convert it
        df_copy = pd.DataFrame(df)

    # If index looks like datetime and has no name, turn it into a timestamp column
    try:
        idx = df_copy.index
        if getattr(idx, "dtype", None) is not None and "datetime" in str(idx.dtype).lower():
            # reset index only if index has meaningful datetime values
            if idx.name is None or idx.name == "":
                df_copy = df_copy.reset_index().rename(columns={"index": "timestamp"})
            else:
                df_copy = df_copy.reset_index()
    except Exception:
        # if anything goes wrong, ignore and continue
        pass

    # Convert datetime columns to ISO strings to ensure clean CSV text
    for col in df_copy.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                # use microseconds then strip trailing zeros in display if needed
                df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception:
            # ignore columns that cause issues
            pass

    # Generate CSV text and encode as UTF-8 bytes
    buf = io.StringIO()
    df_copy.to_csv(buf, index=False)
    csv_text = buf.getvalue()
    return csv_text.encode("utf-8")

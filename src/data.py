from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml


RAW_PATH = Path("data/raw/credit-g.csv")


def load_credit_g() -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
        X: DataFrame of features
        y: Series of labels (binary: 1 = "bad", 0 = "good")
    """
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        if "class" not in df.columns:
            raise ValueError("CSV must contain 'class' column.")
    else:
        # Try OpenML
        # data = fetch_openml(name="credit-g", as_frame=True)
        data = fetch_openml(data_id=31, as_frame=True)  # credit-g v1
        df = data.frame
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False)

    y_raw = df["class"].astype(str)
    # OpenML uses "good"/"bad"
    y = (y_raw == "bad").astype(int)
    X = df.drop(columns=["class"])
    return X, y

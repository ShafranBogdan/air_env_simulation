import os
import pandas as pd

def singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


@singleton
class Logger:
    def __init__(self) -> None:
        self.path = os.path.join(os.getcwd(), "logs")
        try:
            os.mkdir(self.path)
        except OSError:
            print(f"[INFO] dir {self.path} already exist")

    def log_dataFrame(self, data: pd.DataFrame, file_name: str) -> None:
        data.to_csv(f"{os.path.join(self.path, file_name)}.csv", index=False)

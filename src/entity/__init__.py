from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    root_dir: str
    Train_data_path: str
    test_data: str
    data_source: str
    # test_size:int
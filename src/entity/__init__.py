from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    root_dir: str
    Train_data_path: str
    test_data: str
    data_source: str
    # test_size:int

@dataclass
class DataProcessingConfig:
    root_dir: str
    Train_data_path: str
    test_data: str
    clean_Train_data_path: str
    clean_test_data: str

@dataclass
class FeatureEngineeringConfig:
    root_dir: str
    process_Train_data_path: str
    process_test_data: str
    clean_Train_data_path: str
    clean_test_data: str
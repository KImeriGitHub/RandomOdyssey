import pandas as pd
import msgpack
from dataclasses import asdict, is_dataclass, dataclass

from src.common import AssetData

class FileInOut:
    def __init__(self, filepath: str):
        """Initialize the class with a file path."""
        self.filepath = filepath

    def saveToFile(self, ad: AssetData):
        """Save a dataclass instance to a file using msgpack."""

        # Convert dataclass to a dictionary for msgpack compatibility
        ad_dict = asdict(ad)
        
        # Write the serialized object to a file
        with open(self.filepath, 'wb') as f:
            packed_data = msgpack.packb(ad_dict)
            f.write(packed_data)

    def loadFromFile(self, ad: AssetData):
        """Load and deserialize an instance of a dataclass from a file using msgpack."""
        
        # Read from the file and unpack
        with open(self.filepath, 'rb') as f:
            packed_data = f.read()
            ad_dict = msgpack.unpackb(packed_data)
        
        # Create an instance of the dataclass using the unpacked dictionary
        return ad(**ad_dict)

# Example usage:
@dataclass
class ExampleData:
    name: str
    age: int
    active: bool

# Usage Example
file_io = FileInOut('example_data.msgpack')

# Create a dataclass instance
example_instance = ExampleData(name="Alice", age=30, active=True)

# Save the dataclass instance to a file
file_io.save_to_file(example_instance)

# Load the dataclass instance from the file
loaded_instance = file_io.load_from_file(ExampleData)

print(loaded_instance)

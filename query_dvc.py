from sounddevice import query_devices
from typing import Optional

def get_device_index_structured(target_name: str, input_ch: int, output_ch: int, sample_rate: float) -> Optional[int]:
    # Enumerate gives us the index (i) and the device data (d) directly
    for i, device in enumerate(query_devices()):
        if device['name'] == target_name and device['max_input_channels'] == input_ch and device['max_output_channels'] == output_ch and device['default_samplerate'] == sample_rate:
            return device['index']
    return None

# print(get_device_index_structured(target_name='CABLE Input (VB-Audio Virtual Cable)', input_ch=0, output_ch=2, sample_rate=22050.0))

print(query_devices('CABLE Input (VB-Audio Virtual Cable), Windows WASAPI')['index'])
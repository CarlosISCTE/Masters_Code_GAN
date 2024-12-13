import wfdb
from tqdm import tqdm
import os


class ECGUtils:
    @staticmethod
    def extract_heartbeat_segments(record_names, heartbeat_type, max_segments=5000, segment_length=256,
                                   start_offset=-88, end_offset=168,
                                   data_path='/content/drive/My Drive/Colab Notebooks/Datasets/mitbih_data/'):
        """
        Extract heartbeat segments from ECG records.

        Parameters:
        - record_names (list): List of record names
        - heartbeat_type (str or list): Heartbeat type symbol(s) to extract
        - max_segments (int): Maximum number of segments to collect
        - segment_length (int): Length of each segment (in number of data points)
        - start_offset (int): Number of samples before the R-peak to include in the segment (should be a negative value)
        - end_offset (int): Number of samples after the R-peak to include in the segment (should be a positive value)
        - data_path (str): Path to the directory containing the record files

        Returns:
        - segments (list): List of extracted heartbeat segments
        - labels (list): List of corresponding labels for the extracted segments
        """
        segments = []
        labels = []

        if isinstance(heartbeat_type, str):
            heartbeat_classes = [heartbeat_type]
        else:
            heartbeat_classes = heartbeat_type

        for record_name in tqdm(record_names):
            try:
                # Read the record and annotation
                record = wfdb.rdrecord(os.path.join(data_path, record_name))
                annotation = wfdb.rdann(os.path.join(data_path, record_name), 'atr')

                # Find the index of the MLII signal
                mlii_index = record.sig_name.index('MLII')

                # Get the MLII signal
                signal = record.p_signal[:, mlii_index]

                # Get the indices of the beats we're interested in
                indices = [i for i, sym in enumerate(annotation.symbol) if sym in heartbeat_classes]
                r_peaks = annotation.sample[indices]
                symbols = [annotation.symbol[i] for i in indices]

                # Process each R peak
                for r_peak, sym in zip(r_peaks, symbols):
                    start = r_peak + start_offset
                    end = r_peak + end_offset

                    # Check if the segment is within the signal boundaries
                    if start >= 0 and end <= len(signal):
                        segment = signal[start:end]

                        # Ensure the segment has the expected length
                        if len(segment) == segment_length:
                            # Check for additional R peaks within the segment
                            peaks_in_segment = [peak for peak in r_peaks if start <= peak < end]
                            if len(peaks_in_segment) == 1:
                                segments.append(segment)
                                labels.append(sym)

                    if max_segments != -1:
                        if len(segments) >= max_segments:
                            print("Stopped Collecting")
                            return segments, labels

            except Exception as e:
                print(f"Error processing record {record_name}: {e}")
                continue

        return segments, labels

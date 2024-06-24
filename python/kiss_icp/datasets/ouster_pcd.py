# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import os
from pathlib import Path

import natsort
import numpy as np


# DZ: Ouster PCD files exported using rosbag2_tools
class OusterPCDDataset:
    def __init__(self, data_dir: Path, *_, **__):
        self.sequence_id = os.path.basename(data_dir)
        self.scans_dir = os.path.join(os.path.realpath(data_dir), "")
        self.scan_files = natsort.natsorted(glob.glob(f"{data_dir}/*.pcd"))
        if len(self.scan_files) == 0:
            raise ValueError(f"Tried to read point cloud files in {self.scans_dir} but none found")
        self.file_extension = self.scan_files[0].split(".")[-1]
        self.timestamps = [float(f.split("/")[-1].split(".")[0]) / 1e9 for f in self.scan_files]

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        data = np.loadtxt(self.scan_files[idx], dtype=np.float64, skiprows=10)
        xyz = data[:, :3]
        # point-wise timestamps from t field, *10 to scale to 1 as deskewing assuming the mid-timestamp is 0.5
        # if frequency is not 10, must update
        # cannot normalise existing values because there is a chunk missing in the point cloud
        timestamps = data[:, 4] / 1e9 * 10
        return xyz, timestamps

    def get_frames_timestamps(self) -> list:
        return self.timestamps

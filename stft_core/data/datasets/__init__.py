# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.


from .concat_dataset import ConcatDataset
from .vid import VIDDataset
from .vid_rdn import VIDRDNDataset
from .vid_mega import VIDMEGADataset
from .vid_fgfa import VIDFGFADataset
from .vid_stft import VIDSTFTDataset
from .cvcvid_mega import CVCVIDMEGADataset
from .cvcvid_fgfa import CVCVIDFGFADataset
from .cvcvid_image import CVCVIDImageDataset
from .cvcvid_rdn import CVCVIDRDNDataset
from .cvcvid_stft import CVCVIDSTFTDataset

from .JF_cvcvid_image import JF_CVCVIDImageDataset
from .JF_cvcvid_stft import JF_CVCVIDSTFTDataset
from .JF_cvcvid_mega import JF_CVCVIDMEGADataset

__all__ = [
    "ConcatDataset",
    "VIDDataset",
    "VIDRDNDataset",
    "VIDMEGADataset",
    "VIDFGFADataset",
    "VIDSTFTDataset",
    "CVCVIDImageDataset",
    "CVCVIDMEGADataset",
    "CVCVIDFGFADataset",
    "CVCVIDRDNDataset",
    "CVCVIDSTFTDataset",

    "JF_CVCVIDImageDataset",
    "JF_CVCVIDSTFTDataset",
    "JF_CVCVIDMEGADataset"
]

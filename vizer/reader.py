r"""
A plugin for reading raw binary files using threads
"""

from vtkmodules.vtkCommonDataModel import vtkImageData, vtkStructuredData
from vtkmodules.vtkCommonExecutionModel import vtkExtentTranslator
from vtkmodules.numpy_interface import dataset_adapter as dsa

import asyncio
import os
import numpy as np
import concurrent.futures

from . import utils
log = utils.get_logger(__name__)

class Config:
    @classmethod
    def create(cls, filename):
        import re
        reg = re.compile(r"_(?P<bits>\d+)b(?P<unsigned>u?)_?.*_(?P<xdim>\d+)x(?P<ydim>\d+)x(?P<zdim>\d+)")
        m = reg.search(filename)
        if not m:
            return None
        groups = m.groupdict()
        config = Config()
        config.dims[0] = int(groups.get('xdim', 0))
        config.dims[1] = int(groups.get('ydim', 0))
        config.dims[2] = int(groups.get('zdim', 0))
        config.bits = int(groups.get('bits', 0))
        config.unsigned = True if 'unsigned' in groups else False
        if config.dims[0] * config.dims[1] * config.dims[2] == 0:
            return None
        if not config.bits:
            return None
        return config

    def __init__(self) -> None:
        self.dims = [0, 0, 0]
        self.bits = 0
        self.unsigned = False

    def get_dtype(self):
        if self.unsigned:
            return f'uint{self.bits}'
        else:
            return f'int{self.bits}'
        
    def get_extents(self):
        return (0, self.dims[0]-1, 0, self.dims[1]-1, 0, self.dims[2]-1)

    def get_piece_offsets(self, num_pieces):
        et = vtkExtentTranslator()
        et.SetSplitModeToZSlab()
        et.SetWholeExtent(*self.get_extents())
        et.SetNumberOfPieces(num_pieces)
        et.SetGhostLevel(0)

        result = []
        for i in range(num_pieces):
            et.SetPiece(i)
            et.PieceToExtent()
            lext = et.GetExtent()
            result.append({
                'offset': vtkStructuredData.ComputePointIdForExtent(et.GetWholeExtent(), (lext[0], lext[2], lext[4])),
                'count': vtkStructuredData.GetNumberOfPoints(lext)
            })
        return result

def _read_subset(filename, chunk, dtype, wordsize, result):
    log.info('read_chunk %r', chunk)
    import time
    offset = chunk['offset']
    count = chunk['count']
    array = np.fromfile(filename, dtype=dtype, sep='', count=count, offset=wordsize*offset)
    result[offset:offset+count] = array


def _read(filename, num_chunks):
    log.info(f'start read_all: ({filename}, {num_chunks})')
    config = Config.create(filename)
    data = np.empty(vtkStructuredData.GetNumberOfPoints(config.get_extents()), config.get_dtype())
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for chunk in config.get_piece_offsets(num_chunks):
            executor.submit(_read_subset, filename, chunk, config.get_dtype(), config.bits//8, data)
    log.info('end read_all')
    return data

def load_dummy_dataset(filename):
    """loads a dummy dataset with dims based on the file"""
    config = Config.create(filename)
    assert config
    scalars = np.zeros(vtkStructuredData.GetNumberOfPoints(config.get_extents()), config.get_dtype())
    return get_vtk_image(scalars, config)

def get_vtk_image(scalars, config):
    image = vtkImageData()
    output = dsa.WrapDataObject(image)
    output.SetExtent(*config.get_extents())
    output.PointData.append(scalars, "ImageFile")
    output.PointData.SetActiveScalars("ImageFile")
    return image

async def load_dataset(filename):
    """loads dataset"""
    num_chunks = max(os.cpu_count()*2, 4)
    log.info(f'start read_dataset: num_chunks={num_chunks}')
    scalars = await asyncio.to_thread(_read, filename, num_chunks)
    log.info('done read_dataset')
    return get_vtk_image(scalars, Config.create(filename))

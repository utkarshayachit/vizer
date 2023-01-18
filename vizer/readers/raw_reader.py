from vizer import utils
import numpy
import concurrent.futures
import time

from vtkmodules.vtkCommonDataModel import vtkStructuredData
from vtkmodules.vtkCommonExecutionModel import vtkExtentTranslator

log = utils.get_logger(__name__)

def read(filename, raw_config, num_chunks):
    """read data from a raw file"""
    t = time.time()
    log.info('start read %s in %d chunks', filename, num_chunks)
    word_size = numpy.dtype(raw_config.dtype).itemsize

    # allocate memory buffer
    bytes_buffer = bytearray(vtkStructuredData.GetNumberOfPoints(raw_config.vtk_extent) * word_size)
    view = memoryview(bytes_buffer)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for count, offset in _compute_byte_chunks(raw_config, num_chunks):
            executor.submit(_read_bytes, filename, offset, view[offset:offset+count])
    log.info('read %s in %d chunks in %f seconds', filename, num_chunks, time.time() - t)
    return numpy.frombuffer(bytes_buffer, dtype=raw_config.dtype)

def _compute_byte_chunks(raw_config, num_chunks):
    """compute the number of items and byte offset for each chunk"""
    et = vtkExtentTranslator()
    et.SetSplitModeToZSlab()
    et.SetWholeExtent(*raw_config.vtk_extent)
    et.SetNumberOfPieces(num_chunks)
    et.SetGhostLevel(0)

    word_size = numpy.dtype(raw_config.dtype).itemsize
    for i in range(num_chunks):
        et.SetPiece(i)
        et.PieceToExtent()
        lext = et.GetExtent()
        item_count = vtkStructuredData.GetNumberOfPoints(lext)
        item_offset = vtkStructuredData.ComputePointIdForExtent(et.GetWholeExtent(), (lext[0], lext[2], lext[4]))
        yield item_count * word_size, item_offset * word_size

def _read_bytes(filename, offset, view):
    """read a chunk of data from a file"""
    # log.info('read %d bytes from %s at offset %d', len(view), filename, offset)
    with open(filename, "rb", buffering=0) as f:
        f.seek(offset)
        count = f.readinto(view)
        assert count == len(view)
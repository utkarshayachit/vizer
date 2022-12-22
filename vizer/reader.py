r"""
A plugin for reading raw binary files using threads
"""

from paraview.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkStructuredData
from vtkmodules.vtkCommonExecutionModel import vtkExtentTranslator

import os
import numpy as np
import concurrent.futures

class Config:
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
    offset = chunk['offset']
    count = chunk['count']
    array = np.fromfile(filename, dtype=dtype, sep='', count=count, offset=wordsize*offset)
    np.copyto(result[offset:offset+count], array)

@smproxy.reader(name="RAWReader", label="RAW Reader",
                extensions="raw", file_description="raw files")
class RAWReader(VTKPythonAlgorithmBase):
    """Reads raw binary files"""
    def __init__(self):
        super().__init__(nInputPorts=0, nOutputPorts=1, outputType='vtkImageData')
        self._filename = None
        self._num_threads = os.cpu_count()
        self._num_chunks = os.cpu_count()

    def _get_config(self) -> Config:
        """extracts metadata from filename"""
        import re
        reg = re.compile(r"_(?P<bits>\d+)b(?P<unsigned>u?)_?.*_(?P<xdim>\d+)x(?P<ydim>\d+)x(?P<zdim>\d+)")
        m = reg.search(self._filename)
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
 
    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions='raw', file_description="raw files")
    def SetFileName(self, name):
        if self._filename != name:
            self._filename = name
            self.Modified()

    def RequestInformation(self, request, inInfoVec, outInfoVec):
        config = self._get_config()
        if not config:
            return 0

        executive = self.GetExecutive()
        outInfo = executive.GetOutputInformation(0)
        outInfo.Set(executive.WHOLE_EXTENT(), *config.get_extents())
        outInfo.Set(vtkImageData.SPACING(), 1, 1, 1)
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        from vtkmodules.numpy_interface import dataset_adapter as dsa

        config = self._get_config()
        if not config:
            return 0

        data = np.empty(vtkStructuredData.GetNumberOfPoints(config.get_extents()), config.get_dtype())

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            for chunk in config.get_piece_offsets(self._num_chunks):
                executor.submit(_read_subset, self._filename, chunk, config.get_dtype(), config.bits//8, data)

        # # read raw binary data.
        # data = np.fromfile(self._filename, dtype=config.get_dtype(), sep='')
        # dims = config.dims
        # assert data.shape[0] == dims[0]*dims[1]*dims[2], "dimension mismatch"

        output = dsa.WrapDataObject(vtkImageData.GetData(outInfoVec, 0))
        output.SetExtent(*config.get_extents())
        output.PointData.append(data, "ImageFile")
        output.PointData.SetActiveScalars("ImageFile")
        return 1

def load_dataset_paraview(filename: str, subsamplingFactor: int):
    from paraview import simple as pvs
    import logging as log

    reader = RAWReader()
    # reader._num_threads = 2
    # reader._num_chunks = 2
    reader.SetFileName(filename)
    log.info('start update reader')
    reader.Update()
    log.info('end update reader')
    proxy = pvs.PVTrivialProducer()

    if subsamplingFactor > 1:
        from vtkmodules.vtkImagingCore import vtkExtractVOI
        voi = vtkExtractVOI()
        voi.SetVOI(*reader._get_config().get_extents())
        voi.SetSampleRate(subsamplingFactor, subsamplingFactor, subsamplingFactor)
        voi.IncludeBoundaryOff()
        voi.SetInputData(reader.GetOutputDataObject(0))
        log.info('start subsampling')
        voi.Update()
        log.info('done subsampling')
        proxy.GetClientSideObject().SetOutput(voi.GetOutputDataObject(0))
        proxy.WholeExtent = voi.GetOutputDataObject(0).GetExtent()
    else:
        proxy.WholeExtent = reader._get_config().get_extents()
        proxy.GetClientSideObject().SetOutput(reader.GetOutputDataObject(0))
    return proxy

if __name__ == '__main__':
    import logging, argparse, timeit
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-pieces', '-p', help="number of chunks to split the reads in",
        default=os.cpu_count(), type=int)
    parser.add_argument('--num-threads', '-t', help="number of threads to read with",
        default=os.cpu_count(), type=int)
    parser.add_argument('file', help="name of the dataset file to load")

    args = parser.parse_args()

    reader = RAWReader()
    reader.SetFileName(args.file)
    reader._num_chunks = args.num_pieces
    reader._num_threads = args.num_threads
    t0 = timeit.timeit('reader.Update()', globals=globals())
    logging.info(f'read: {t0}')

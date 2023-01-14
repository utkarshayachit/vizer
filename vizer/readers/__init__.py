from os import path, cpu_count
from vizer import utils
import re
import asyncio
import numpy

from paraview import simple, vtk
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkStructuredData
from vtkmodules.numpy_interface import dataset_adapter as dsa

from . import raw_reader

log = utils.get_logger(__name__)

class RawConfig:
    def __init__(self) -> None:
        self.dims = [0, 0, 0]
        self.bits = 0
        self.unsigned = False

    def __str__(self) -> str:
        return f'bits: {self.bits}, unsigned: {self.unsigned}, dims: {self.dims}'

    def is_valid(self) -> bool:
        return self.dims[0] * self.dims[1] * self.dims[2] > 0 and self.bits > 0

    @property
    def vtk_scalar_type(self):
        if self.unsigned:
            typeMap = {8 : vtk.VTK_UNSIGNED_CHAR,
             16: vtk.VTK_UNSIGNED_SHORT,
             32: vtk.VTK_UNSIGNED_INT,
             64: vtk.VTK_UNSIGNED_LONG}
        else:
            typeMap = {8 : vtk.VTK_CHAR,
             16: vtk.VTK_SHORT,
             32: vtk.VTK_INT,
             64: vtk.VTK_LONG}
        return typeMap.get(self.bits, None)

    @property
    def vtk_extent(self):
        return [0, self.dims[0]-1, 0, self.dims[1]-1, 0, self.dims[2]-1]

    @property
    def dtype(self):
        """returns numpy dtype"""
        if self.unsigned:
            return f'uint{self.bits}'
        else:
            return f'int{self.bits}'

    @staticmethod
    def extract_config(filename):
        """extracts metadata from the file's name"""
        reg = re.compile(r"_(?P<bits>\d+)b(?P<unsigned>u?)_?.*_(?P<xdim>\d+)x(?P<ydim>\d+)x(?P<zdim>\d+)")
        m = reg.search(filename)
        if not m:
            return RawConfig()
        groups = m.groupdict()
        config = RawConfig()
        config.dims[0] = int(groups.get('xdim', 0))
        config.dims[1] = int(groups.get('ydim', 0))
        config.dims[2] = int(groups.get('zdim', 0))
        config.bits = int(groups.get('bits', 0))
        config.unsigned = True if 'unsigned' in groups else False
        return config if config.is_valid() else None


class Metadata:
    """Metadata for a file."""
    def __init__(self, filename) -> None:
        self.filename = filename
        self.vtk_type = None
        self.is_structured = False
        self.raw_config = None

        if not path.isfile(filename):
            log.error(f'file "{filename}" does not exist')
        else:
            root, ext = path.splitext(filename)
            
            if ext.lower() == ".raw":
                # read metadata for raw file
                self.raw_config = RawConfig.extract_config(filename)
                self.vtk_type = 6 # VTK_IMAGE_DATA
                self.is_structured = True
            else:
                # use VTK to read metadata
                reader = simple.OpenDataFile(filename)
                if not reader:
                    log.error(f'could not read file "{filename}"')
                    return None
                reader.UpdatePipelineInformation()
                self.vtk_type = reader.GetDataInformation().GetDataSetType()
                self.is_structured = reader.GetDataInformation().IsDataStructured()
                simple.Delete(reader)
            
    def __str__(self) -> str:
        if self.vtk_type is None:
            pretty_type = 'None'
        else:
            pretty_type = simple.servermanager.vtkPVDataInformation.GetDataSetTypeAsString(self.vtk_type)
        return f'filename: {self.filename}, vtk_type: {self.vtk_type} ({pretty_type}), is_structured: {self.is_structured}'

    def is_empty(self):
        return self.vtk_type is None or self.vtk_type < 0

    def is_raw(self):
        return self.raw_config is not None

    def sync_read_dataset(self, args):
        """read data synchronously"""
        return sync_read(self, args)

    async def async_read_dataset(self, args):
        """read data asynchronously"""
        return await async_read(self, args)

def _create_vtk_image_data(raw_config: RawConfig, buffer:numpy.ndarray=None):
    """creates a vtkImageData object with the given dimensions and scalar type"""
    dataset = vtkImageData()
    dataset.SetExtent(raw_config.vtk_extent)
    if buffer is None:
        dataset.AllocateScalars(raw_config.vtk_scalar_type, 1)
        dataset.GetPointData().GetScalars().SetName('ImageFile')
        dataset.GetPointData().GetScalars().Fill(0)
    else:
        nds = dsa.WrapDataObject(dataset)
        nds.PointData.append(buffer, 'ImageFile')
        nds.PointData.SetActiveScalars('ImageFile')
        log.info(f'buffer range: {nds.PointData["ImageFile"].GetRange()}')
    return dataset

def sync_read(meta, args):
    """read data synchronously; may only read dummy data"""
    if meta.is_raw() and not args.use_vtk_reader:
        # can read data asynchronously
        dataset = _create_vtk_image_data(meta.raw_config)
        return (dataset, True)
    else:
        # read data synchronously
        if meta.is_raw():
            producer = simple.OpenDataFile(meta.filename,
                DataScalarType=meta.raw_config.vtk_scalar_type,
                DataExtent=meta.raw_config.vtk_extent,
                DataByteOrder='LittleEndian')
        else:
            producer = simple.OpenDataFile(meta.filename)
        producer.UpdatePipeline()
        dataset = producer.GetClientSideObject().GetOutputDataObject(0)
        return (dataset, False)

async def async_read(meta, args):
    """reads data asynchronously"""
    if not meta.is_raw() or args.use_vtk_reader:
        log.info('asynchronous reading is only supported for raw files')
        return None

    num_chunks = max(cpu_count() + 4, 4)
    log.info(f'reading {meta.filename} in {num_chunks} chunks')

    # we don't want to block the main thread, so we use a separate thread for
    # the actual reading
    buffer = await asyncio.to_thread(raw_reader.read, meta.filename, meta.raw_config, num_chunks)
    return _create_vtk_image_data(meta.raw_config, buffer)
from os import path
import re
from . import utils

from paraview import simple

log = utils.get_logger(__name__)

class Config:
    def __init__(self) -> None:
        self.dims = [0, 0, 0]
        self.bits = 0
        self.unsigned = False
        self.raw_filename = None

def extract_config(filename: str) -> Config:
    """extracts metadata from filename"""
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

def process_config(metadata: dict) -> Config:
    """generates config from metadata"""
    return None

def load_dataset(filename):
    """loads dataset and reads meta-data"""
    if not path.isfile(filename):
        log.error(f'file "{filename}" does not exit')
        return False
    root, ext = path.splitext(filename)
    if ext.lower() != ".raw":
        log.error(f'only .raw files are supported (got {ext})')
        return False

    config = extract_config(filename)
    if not config:
        log.error('could not determine metadata')
        return False
    config.raw_filename = filename
    reader = create_reader(config)
    return reader

def create_reader(config: Config):
    typeMap = {8 : 'char', 16: 'short', 32: 'int', 64: 'long' }
    scalarType = typeMap.get(config.bits, None)
    if not scalarType:
        log.error('failed to determine scalar type (%d)', config.bits)
        return None
    if config.unsigned:
        scalarType = 'unsigned %s' % scalarType

    log.info('scalar type "%s"', scalarType)
    log.info('dims: "%r"', config.dims)
    reader = simple.OpenDataFile(filename=config.raw_filename,
        DataScalarType=scalarType,
        DataByteOrder='LittleEndian',
        DataExtent=[ 0, config.dims[0]-1,
                     0, config.dims[1]-1,
                     0, config.dims[2]-1])
    return reader


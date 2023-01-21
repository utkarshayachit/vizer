
from paraview.web import venv # processes --venv arg
from vizer.readers import Metadata
import argparse

parser = argparse.ArgumentParser(description='vizer.reader')
parser.add_argument('--dataset', help='dataset(s) to load (REQUIRED)', required=True)
parser.add_argument('--use-vtk-reader',
        help='use standard VTK reader (default: False)', default=False, action='store_true')

# parse args
args = parser.parse_known_args()[0]
meta = Metadata(args.dataset)
dataset, needs_async = meta.sync_read_dataset(args)

if needs_async:
    import asyncio
    asyncio.run(meta.async_read_dataset(args))

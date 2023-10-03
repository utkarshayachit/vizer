from trame.app import get_server, asynchronous
from trame.widgets import client, vuetify, html, paraview
from trame.ui.vuetify import VAppLayout

import os
import sys
import asyncio

from paraview.simple import LoadPlugins
LoadPlugins(os.path.join(os.path.dirname(__file__), 'plugins', 'outline_filter.py'))
LoadPlugins(os.path.join(os.path.dirname(__file__), 'plugins', 'extract_voi.py'))
LoadPlugins(os.path.join(os.path.dirname(__file__), 'plugins', 'color_mappyer.py'))

from . import utils, views, readers
log = utils.get_logger(__name__)


def create_view(filename, args):
    """Creates a view from a dataset metadata."""
    # read dataset metadata
    meta = readers.Metadata(filename)
    log.info(f'metadata: {meta}')
    view = views.create(meta, args)
    log.info("created view '%s'", view.__class__.__name__)
    return view


def exec():
    server = get_server()
    paraview.initialize(server)


    state, ctrl = server.state, server.controller
    server.cli.add_argument('--dataset', help='dataset(s) to load (REQUIRED) (REPEATABLE)', required=True, action='append')
    server.cli.add_argument('--create-on-server-ready', help='file to create when server is ready')
    server.cli.add_argument('--use-vtk-reader',
            help='use standard VTK reader (default: False)', default=False, action='store_true')
    server.cli.add_argument('--subsampling-factor',
            help='specify image sub-sampling factor', default=1, type=int)
    server.cli.add_argument('--force-view', help="force view type (primarily for debugging)", default=None,
        choices=views.get_view_types())
    server.cli.add_argument('--link-views', help='link interaction between views of same time (default: True)', default=False, action='store_true')
    server.cli.add_argument('--legend-visibility', help='color legend visibility (default: "never")', default='never', type=str, choices=['never', 'auto'])
    # server.cli.add_argument('--segmentation', help='if supported, enable segmentation support (default: False)', default=False, action='store_true')
    # parse args
    args = server.cli.parse_known_args()[0]
    
    all_views = [create_view(dataset.format_map(os.environ), args) for dataset in args.dataset]

    layout = VAppLayout(server)
    with layout:
        # disable vertical scrollbars (they are not needed)
        client.Style('html { overflow-y: hidden; } .container { max-width: 999999px; }')
        with layout.root as root:
            with html.Div(classes='d-flex flex-row flex-nowrap fill-height'):
                for v in all_views:
                    v.widget

    def startup(*_, **__):
        """callback which loads dataset when the server starts up"""

        # load dataset, synchronously; if possible only dummy data is loaded
        # and actual reading is done asynchronously
        for view in all_views:
            view.set_status('preparing data ...')

            # setup async task to read data; if needed.
            asynchronous.create_task(view.load_dataset())
        
        if args.create_on_server_ready is not None:
            fname = args.create_on_server_ready.format_map(os.environ)
            with open(fname, 'w') as f:
                f.write('server ready')
            log.info('created \'%s\'', fname)

    log.info('starting server')
    ctrl.on_server_ready.add(startup)
    server.start()

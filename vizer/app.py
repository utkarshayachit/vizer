from trame.app import get_server, asynchronous
from trame.widgets import vuetify
from trame.ui.vuetify import VAppLayout

from . import utils, simple_view, quad_view
import os
import sys
import asyncio

log = utils.get_logger(__name__)

def exec():
    server = get_server()
    state, ctrl = server.state, server.controller
    server.cli.add_argument('--dataset',help='dataset to load (REQUIRED)', required=True)
    server.cli.add_argument('--create-on-server-ready',
            help='file to create when server is ready')
    server.cli.add_argument('--use-vtk-reader',
            help='use standard VTK reader', default=False, action='store_true')
    server.cli.add_argument('--subsampling-factor',
            help='specify image sub-sampling factor', default=4, type=int)

    # parse args
    args = server.cli.parse_known_args()[0]
    args.dataset = args.dataset.format_map(os.environ)

    # dataset to load
    if quad_view.can_show(args.dataset):
        viewer = quad_view
    elif simple_view.can_show(args.dataset):
        viewer = simple_view
    else:
        log.error('dataset not supported')
        return

    log.info('prepare to load dataset "%s"', args.dataset)
    viewer.prepare(args)

    def startup(*_, **__):
        """callback which loads dataset when the server starts up"""
        # setup async task to load dataset
        asynchronous.create_task(viewer.async_load(args))

        if args.create_on_server_ready is not None:
            fname = args.create_on_server_ready.format_map(os.environ)
            with open(fname, 'w') as f:
                f.write('server ready')
            log.info('created \'%s\'', fname)

    with VAppLayout(server) as layout:
        with layout.root:
            viewer.get_widget()

    log.info('setting up visualizations')
    viewer.setup_visualizations(state)

    log.info('starting server')
    ctrl.on_server_ready.add(startup)
    server.start()
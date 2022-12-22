from trame.app import get_server
from trame.widgets import vuetify
from trame.ui.vuetify import VAppLayout

from . import utils, simple_view, quad_view
import os
import sys

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

    # dataset to load
    dataset = args.dataset.format_map(os.environ)
    if quad_view.can_show(dataset):
        viewer = quad_view
    elif simple_view.can_show(dataset):
        viewer = simple_view
    else:
        log.error('dataset not supported')
        return

    log.info('loading dataset "%s"', dataset)
    viewer.load_dataset(dataset, args)

    def startup(*_, **__):
        """callback which loads dataset when the server starts up"""
        if args.create_on_server_ready is not None:
            fname = args.create_on_server_ready.format_map(os.environ)
            with open(fname, 'w') as f:
                f.write('server ready')

    with VAppLayout(server) as layout:
        with layout.root:
            viewer.get_widget()

    log.info('setting up visualizations')
    viewer.setup_visualizations(state)

    log.info('starting server')
    ctrl.on_server_ready.add(startup)
    server.start()
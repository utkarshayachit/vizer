from paraview import simple
from trame.widgets import vuetify, paraview
import asyncio

# setup logging
from . import utils
log = utils.get_logger(__name__)

class GLOBALS:
    View = None
    HTMLView = None
    Reader = None
    Display = None

def can_show(filename):
    """returns true if this view can show the dataset from the file"""
    return True

def get_widget():
    view = create_view()
    card = vuetify.VCard(app=True,
        dark=True,
        fluid=True,
        classes='fill-height')
    with card:
        htmlView = paraview.VtkRemoteView(view, ref='simpleview', interactive_ratio=0.5)
        GLOBALS.View = view
        GLOBALS.HTMLView = htmlView
    return card
 
def prepare(args):
    GLOBALS.Reader = simple.OpenDataFile(args.dataset)
    GLOBALS.Reader.UpdatePipeline()

def setup_visualizations(state):
    assert GLOBALS.View and GLOBALS.HTMLView and GLOBALS.Reader
    GLOBALS.Display = simple.Show(GLOBALS.Reader, GLOBALS.View)
    simple.ResetCamera()
    GLOBALS.View.CenterOfRotation = GLOBALS.View.CameraFocalPoint.GetData()


def create_view():
    view = simple.CreateRenderView()
    # change background color
    view.Background = [0.12, 0.12, 0.12]
    view.UseColorPaletteForBackground = False
    return view


async def async_load(args):
    pass

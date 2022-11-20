r"""
  Quad-view: suitable for showing image datasets
"""
from paraview import simple
from trame.widgets import vuetify, paraview

# setup logging
from . import utils, loader, simple_view

log = utils.get_logger(__name__)

class GLOBALS:
    Reader = None
    LUT = None
    SliceViews = [None, None, None]
    VolumeView = None
    HTMLSliceViews = [None, None, None]
    HTMLVolumeView = None


def can_show(filename):
    """returns true if the quad_view can show the dataset from the file"""
    config = loader.extract_config(filename)
    return True if config is not None else False

def get_widget():
    card = vuetify.VCard(app=True,
        dark=True,
        fluid=True,
        classes='fill-height')
    with card:
        with vuetify.VRow(no_gutters=True, style="height:50%;"):
            with vuetify.VCol():
                create_slice_view(axis=0)
            with vuetify.VCol():
                create_slice_view(axis=1)
        with vuetify.VRow(no_gutters=True, style="height:50%;"):
            with vuetify.VCol():
                create_slice_view(axis=2)
            with vuetify.VCol():
                create_volume_view()
    return card
 
def load_dataset(filename):
    GLOBALS.Reader = loader.load_dataset(filename)

def setup_visualizations(state):
    GLOBALS.LUT = simple.GetColorTransferFunction('ImageFile')
    GLOBALS.LUT.ApplyPreset('Blue Orange (divergent)', True)
    setup_volume()
    setup_slice(0, state)
    setup_slice(1, state)
    setup_slice(2, state)

def create_card():
    return vuetify.VCard(tile=True, fluid=True,
        classes="fill-height grow d-flex flex-column flex-nowrap")

def create_slice_view(axis:int):
    card = create_card()
    with card:
        with vuetify.VRow(classes="grow"):
            with vuetify.VContainer(classes="fill-height"):
                view = simple_view.create_view()
                htmlView = paraview.VtkRemoteView(view, ref=f'view_slice_{axis}', interactive_ratio=0.5)
                GLOBALS.SliceViews[axis] = view
                GLOBALS.HTMLSliceViews[axis] = htmlView

        with vuetify.VRow(classes="shrink ma-1"):
            vuetify.VSlider(hide_details=True, min=(f'min{axis}', 0), max=(f'max{axis}', 0), step=1,
                v_model=(f'slice{axis}', 0))
    return card

def create_volume_view():
    card = create_card()
    with card:
        with vuetify.VRow(classes="grow"):
            with vuetify.VContainer(classes="fill-height"):
                view = simple_view.create_view()
                htmlView = paraview.VtkRemoteView(view, ref=f'view_volume', interactive_ratio=0.5)
                GLOBALS.VolumeView = view
                GLOBALS.HTMLVolumeView = htmlView
    return card
    
def setup_volume():
    simple.SetActiveView(GLOBALS.VolumeView)
    display = simple.Show(GLOBALS.Reader, GLOBALS.VolumeView)
    simple.ColorBy(display, ('POINTS', 'ImageFile'))
    display.SetRepresentationType('Volume')
    simple.ResetCamera()
    GLOBALS.VolumeView.CenterOfRotation = GLOBALS.VolumeView.CameraFocalPoint.GetData()

def setup_slice(axis:int, state):
    ext = GLOBALS.Reader.GetDataInformation().GetExtent()
    view = GLOBALS.SliceViews[axis]
    htmlView = GLOBALS.HTMLSliceViews[axis]

    simple.SetActiveView(view)
    display = simple.Show(GLOBALS.Reader, view)
    simple.ColorBy(display, ('POINTS', 'ImageFile'))
    display.SetRepresentationType('Slice')
    view.InteractionMode = '2D'

    modes = ['YZ Plane', 'XZ Plane', 'XY Plane']
    display.SliceMode = modes[axis]
    display.Slice = (ext[2*axis] + ext[2*axis+1]) // 2
    state[f'min{axis}'] = ext[2*axis]
    state[f'max{axis}'] = ext[2*axis + 1]
    state[f'slice{axis}'] = (ext[2*axis] + ext[2*axis+1]) // 2

    pos = [ [10, 0, 0], [0, -10, 0], [0, 0, 10] ]
    view.CameraPosition = pos[axis]

    up = [ [0, 0, 1], [0, 0, 1], [0, 1, 0] ]
    view.CameraViewUp = up[axis]

    simple.ResetCamera()
    name = ['X', 'Y', 'Z']
    text = simple.Text()
    text.Text = f'{name[axis]} Slice: {display.Slice}'
    simple.Show(text, view)

    @state.change(f'slice{axis}')
    def slice(**kwargs):
        offset = kwargs.get(f'slice{axis}')
        display.Slice = offset
        text.Text = f'{name[axis]} Slice: {display.Slice}'
        htmlView.update()
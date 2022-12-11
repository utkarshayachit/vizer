r"""
  Quad-view: suitable for showing image datasets
"""
from paraview import simple
from trame.widgets import vuetify, paraview
from sympy.ntheory import primefactors

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
    ExtractSubsets = [None, None, None]
    SliceOutlines = [None, None, None]

    Callbacks = [[], [], []]

    @classmethod
    def get_views(cls):
        return cls.SliceViews + [ cls.VolumeView ]

    @classmethod
    def get_html_views(cls):
        return cls.HTMLSliceViews + [cls.HTMLVolumeView]


class CONSTANTS:
    Modes = ['YZ Plane', 'XZ Plane', 'XY Plane']
    Colors = [ [1., 0., 0.], [1., 1., 0.],  [0., 1., 0.] ]


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
    GLOBALS.Reader.UpdatePipeline()
    GLOBALS.LUT = simple.GetColorTransferFunction('ImageFile')
    GLOBALS.LUT.ApplyPreset('Blue Orange (divergent)', True)
    # setup_volume()
    for axis in range(3):
        setup_slice(axis, state)
    setup_3dview(state)
    setup_outlines()

def create_card():
    return vuetify.VCard(tile=True, fluid=True,
        classes="fill-height grow d-flex flex-column flex-nowrap")

def get_interaction_callback(axis:int):
    def callback(*args, **kwargs):
        sview = GLOBALS.SliceViews[axis]
        sfp = sview.CameraFocalPoint

        for i in range(3):
            if i == axis: continue
            tview = GLOBALS.SliceViews[i]
            tview.CameraParallelScale = sview.CameraParallelScale

            pos = [0, 0, 0]
            for cc in range(3):
                pos[cc] = sfp[cc] + tview.CameraPosition[cc] - tview.CameraFocalPoint[cc]

            tview.CameraFocalPoint = sfp
            tview.CameraPosition = pos
            GLOBALS.HTMLSliceViews[i].update()

    return callback

def create_slice_view(axis:int):
    card = create_card()
    with card:
        with vuetify.VRow(classes="grow"):
            with vuetify.VContainer(classes="fill-height"):
                view = simple_view.create_view()
                view.GetInteractor().AddObserver('InteractionEvent', get_interaction_callback(axis))
                htmlView = paraview.VtkRemoteView(view, ref=f'view_slice_{axis}', interactive_ratio=1.0)
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
                htmlView = paraview.VtkRemoteView(view, ref=f'view_volume', interactive_ratio=1.0)
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

def setup_3dview(state):
    simple.SetActiveView(GLOBALS.VolumeView)
    ext = GLOBALS.Reader.GetDataInformation().GetExtent()

    sampleIJK = [1, 1, 1]
    # we use prime-factors to ensure we always include the boundaries
    for dim in range(3):
        sampleIJK[dim] = primefactors(ext[2*dim + 1] - ext[2*dim], limit=200)[0]

    subset = simple.ExtractSubset(Input=GLOBALS.Reader)
    subset.SampleRateI = sampleIJK[0]
    subset.SampleRateJ = sampleIJK[1]
    subset.SampleRateK = sampleIJK[2]
    display = simple.Show(subset, GLOBALS.VolumeView)
    display.SetRepresentationType('Outline')
    # display.Opacity = 0.3
    simple.ColorBy(display, ('POINTS', 'ImageFile'))

    for axis in range(3):
        slice = simple.ExtractSubset(Input=GLOBALS.Reader, VOI=ext)
        slice.VOI[2*axis] = slice.VOI[2*axis+1] = state[f'slice{axis}']

        sliceDisplay = simple.Show(GLOBALS.ExtractSubsets[axis], GLOBALS.VolumeView)
        simple.ColorBy(sliceDisplay, ('POINTS', 'ImageFile'))
        sliceDisplay.SetRepresentationType('Slice')
        # outline = simple.Outline(Input=GLOBALS.ExtractSubsets[axis])
        # outlineDisplay = simple.Show(outline, GLOBALS.VolumeView)
        # outlineDisplay.SetRepresentationType('Outline')
        # outlineDisplay.AmbientColor = CONSTANTS.Colors[axis]
        # outlineDisplay.DiffuseColor = CONSTANTS.Colors[axis]
        # outlineDisplay.LineWidth=4

    simple.ResetCamera()
    GLOBALS.VolumeView.CenterOfRotation = GLOBALS.VolumeView.CameraFocalPoint.GetData()

def setup_outlines():
    for axis in range(3):
        outline = simple.Outline(Input=GLOBALS.ExtractSubsets[axis])

        for view in GLOBALS.get_views():
            outlineDisplay = simple.Show(outline, view)
            outlineDisplay.SetRepresentationType('Outline')
            outlineDisplay.AmbientColor = CONSTANTS.Colors[axis]
            outlineDisplay.DiffuseColor = CONSTANTS.Colors[axis]
            outlineDisplay.LineWidth=4


def setup_slice(axis:int, state):
    ext = GLOBALS.Reader.GetDataInformation().GetExtent()
    view = GLOBALS.SliceViews[axis]
    htmlView = GLOBALS.HTMLSliceViews[axis]

    state[f'min{axis}'] = ext[2*axis]
    state[f'max{axis}'] = ext[2*axis + 1]
    state[f'slice{axis}'] = (ext[2*axis] + ext[2*axis+1]) // 2

    simple.SetActiveView(view)

    # create extract slice filter
    slice = simple.ExtractSubset(Input=GLOBALS.Reader, VOI=ext)
    slice.VOI[2*axis] = slice.VOI[2*axis+1] = state[f'slice{axis}']

    # setup display for slice
    display = simple.Show(slice, view)
    simple.ColorBy(display, ('POINTS', 'ImageFile'))
    display.SetRepresentationType('Slice')

    # setup camera position
    pos = [ [10, 0, 0], [0, -10, 0], [0, 0, 10] ]
    up = [ [0, 0, 1], [0, 0, 1], [0, 1, 0] ]
    view.CameraPosition = pos[axis]
    view.CameraViewUp = up[axis]
    view.InteractionMode = '2D'
    simple.ResetCamera()

    # add text annotation
    name = ['X', 'Y', 'Z']
    text = simple.Text()
    text.Text = f'{name[axis]} Slice: {display.Slice}'
    textDisplay = simple.Show(text, view)
    textDisplay.Color = CONSTANTS.Colors[axis]

    GLOBALS.ExtractSubsets[axis] = slice

    @state.change(f'slice{axis}')
    def slice(**kwargs):
        offset = kwargs.get(f'slice{axis}')
        GLOBALS.ExtractSubsets[axis].VOI[2*axis] = \
            GLOBALS.ExtractSubsets[axis].VOI[2*axis+1] = offset
        text.Text = f'{name[axis]} Slice: {offset}'
        for v in GLOBALS.get_html_views():
            v.update()

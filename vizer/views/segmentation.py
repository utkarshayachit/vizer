r"""segmentation view"""

import math

from .base import Base, Layout
from vizer import utils

from paraview import simple, vtk
from trame.widgets import vuetify, paraview, html
from trame.app import get_server, asynchronous

log = utils.get_logger(__name__)

import vtkmodules.vtkRenderingCore as rc
from vtkmodules.vtkFiltersSources import vtkOutlineSource

class Swatch:
    """Renders a single swatch"""
    def __init__(self, category, color, voi=[0, -1, 0, -1, 0, -1]) -> None:
        super().__init__()

        self._average = 0
        self.category = category
        self.voi = voi
        self.box_source = vtkOutlineSource()
        self.box_source.SetBounds(0, 100, 0, 100, 0, 100)
        self.box_source.GenerateFacesOn()

        mapper = rc.vtkPolyDataMapper2D()
        mapper.SetInputConnection(self.box_source.GetOutputPort())

        # specify transform coordinate system since data is in world coordinates
        # and not viewport coordinates
        coord = rc.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        mapper.SetTransformCoordinate(coord)

        self.border = rc.vtkActor2D()
        self.border.SetMapper(mapper)
        self.border.GetProperty().SetColor(color)
        self.border.GetProperty().SetOpacity(0.2)
        self.border.GetProperty().SetLineWidth(2)
        self.hide()

    def get_average(self):
        """Returns the average value of the swatch"""
        return self._average

    def compute_average(self, dataset):
        # from vtkmodules.vtkAcceleratorsVTKmFilters import vtkmExtractVOI as vtkExtractVOI
        from vtkmodules.vtkImagingCore import vtkExtractVOI
        from vtkmodules.numpy_interface import dataset_adapter as dsa
        import numpy

        if not hasattr(self, 'extractor'):
            self.extractor = vtkExtractVOI()
        self.extractor = vtkExtractVOI()
        self.extractor.SetInputData(dataset)
        self.extractor.SetVOI(self.voi)
        self.extractor.Update()
        self.extractor.SetInputData(None)
        
        vtk_data = self.extractor.GetOutputDataObject(0)
        data = dsa.WrapDataObject(vtk_data)
        scalars = data.PointData[vtk_data.GetPointData().GetScalars().GetName()]
        if scalars is None or scalars.shape[0] == 0:
            return 0
        return numpy.mean(scalars)

    def set_voi(self, voi, dataset):
        self.voi = voi
        self._average = self.compute_average(dataset)

        # using dataset dimensions, convert voi to world coordinates
        spacing = dataset.GetSpacing()
        ext = dataset.GetExtent()
        origin = dataset.GetOrigin()

        bbox = vtk.vtkBoundingBox()
        minPt = [0] * 3
        maxPt = [0] * 3
        for i in range(3):
            minPt[i] = voi[2*i] * spacing[i] + origin[i]
            maxPt[i] = voi[2*i+1] * spacing[i] + origin[i]
        bbox.AddPoint(minPt)
        bbox.AddPoint(maxPt)
 
        bds=[0]*6
        bbox.GetBounds(bds)
        self.box_source.SetBounds(bds)
        # log.info(f"swatch voi: {voi}    bounds: {bds}")

    def add_to_view(self, view):
        pvview = view.GetClientSideObject()
        renderer = pvview.GetRenderer(pvview.NON_COMPOSITED_RENDERER)
        renderer.AddActor(self.border)
    
    def remove_from_view(self, view):
        pvview = view.GetClientSideObject()
        renderer = pvview.GetRenderer(pvview.NON_COMPOSITED_RENDERER)
        renderer.RemoveActor(self.border)

    def show(self):
        self.border.VisibilityOn()

    def hide(self):
        self.border.VisibilityOff()
    
    def on_slice(self, axis, slice):
        return self.voi[2*axis] == self.voi[2*axis+1] and self.voi[2*axis] == slice

def clamp(x, min, max):
    return min if x < min else max if x > max else x

class Segmentation(Base):
    CATEGORIES = ['Pore', 'Mineral', 'Porous Material']
    CATEGORY_COLORS = ['red', 'green', 'blue']
    CATEGORY_RGB_COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    
    def __init__(self, meta, opts, **kwargs) -> None:
        super().__init__(meta, opts)
        self.parent = kwargs.get('parent', None)

        self._state = {}
        self._state['categories'] = Segmentation.CATEGORIES
        
        self._state['tab_index'] = 0
        self._state['disabled'] = False
        self._state['reset_disabled'] = False
        self._state['active_average'] = 0
        self._selection_active = False
        self._swatches = []
        self._active_swatches = []

        self._axis = 0
        self._slice = 0

    def update_active_swatches(self):
        """updates the active swatches"""
        category = Segmentation.CATEGORIES[self._state['tab_index']]
        self._active_swatches = []
        for swatch in self._swatches:
            if swatch.category == category and swatch.on_slice(self._axis, self._slice):
                self._active_swatches.append(swatch)
                swatch.show()
            else:
                swatch.hide()

    def update_client_state(self):
        """updates the client with the current state."""
        self._state['disabled'] = len(self._active_swatches) == 0
        self._state['reset_disabled'] = len(self._swatches) == 0

        # update the average values
        active_category = Segmentation.CATEGORIES[self._state['tab_index']]
        self._state['category_average'] = self.get_average(active_category)
        self._state['active_average'] = self._get_average(self._active_swatches)
        
        state = get_server().state
        new_state = {}
        for x, y in self._state.items():
            new_state[f'{self.id}_{x}'] = y
        state.update(new_state)
        state.flush()

    def _get_average(self, swatches):
        if len(swatches) == 0:
            return 0
        count = len(swatches)
        avg = 0
        for swatch in swatches:
            avg += swatch.get_average()
        avg /= count
        return avg

    def create_render_view(self):
        view = super().create_render_view()

        # setup callbacks to handle selection
        iren = view.GetInteractor()
        iren.AddObserver('LeftButtonPressEvent', lambda obj, eventId: self._selection_callback(eventId))
        iren.AddObserver('LeftButtonReleaseEvent', lambda obj, eventId: self._selection_callback(eventId))
        iren.AddObserver('MouseMoveEvent', lambda obj, eventId: self._selection_callback(eventId))

        # update manipulators
        view.Camera2DManipulators = [ 1, 2, 2, 0, 1, 1, 0, 2, 2 ]
        return view

    def create_widget(self):
        self.update_client_state()
        # this is needed to ensure the rendering viewport has full height; otherwise
        # it doesn't size properly :/.
        self.layout.viewport.classes = "grow d-flex pa-0 ma-0"

        with self.layout.viewport:
            with vuetify.VRow(no_gutters=True, classes="grow"):
                with vuetify.VContainer(classes="fill-height pa-0 ma-0"):
                    self._view = self.create_render_view()
                    self._html_view = self.create_html_view(self._view)
            with vuetify.VRow(no_gutters=True, classes="shrink"):
                with vuetify.VCard(classes="pa-0 ma-0 fill-height"):
                    vuetify.VCardTitle("Assign Selections")
                    vuetify.VCardSubtitle("Ctrl + Left Button to add a new selection.")
                    with vuetify.VTabs(v_model=f"{self.id}_tab_index"):
                        for idx in range(len(Segmentation.CATEGORIES)):
                            vuetify.VTab(Segmentation.CATEGORIES[idx], style=f"color: {Segmentation.CATEGORY_COLORS[idx]}")
                    with vuetify.VCardText():
                        html.Div("Averages (slice): {{ %s_active_average}}" % self.id)
                        html.Div("Averages (category): {{ %s_category_average}}" % self.id)
                    with vuetify.VCardActions():
                        vuetify.VBtn("Delete Last", color="error", disabled=(f'{self.id}_disabled',),
                                        click=self._delete_last)
                        vuetify.VBtn("Clear", color="error", disabled=(f'{self.id}_disabled',),
                                        click=self._delete_all)
                        vuetify.VBtn("Reset", color="error", disabled=(f'{self.id}_reset_disabled',),
                                        click=self._reset)

        state = get_server().state
        @state.change(f'{self.id}_tab_index')
        def _tab_index_changed(**kwargs):
            val = int(kwargs.get(f'{self.id}_tab_index', 0))
            self._state['tab_index'] = val
            self.update_active_swatches() 
            self.update_client_state()
            self._html_view.update()

    @staticmethod
    def can_show(meta):
        """returns true if this view can show the dataset"""
        return meta.is_structured and  meta.is_raw()

    @property
    def layout(self):
        if not hasattr(self, '_layout'):
            self._layout = Layout("Segmentation Tool",
                close=(lambda **_: self.parent.layout.hide_dialog()) if self.parent is not None else None)
            self.set_status('ready')
        return self._layout
    
    @property
    def subsampling_factor(self):
        return 1 # we don't support subsampling for this view

    def setup(self, axis, slice, subsampling_factor, dataset):
        """reinitializes the view to show the given slice"""
        self._axis = axis
        self._slice = slice
        self._state['tab_index'] = 0

        # reorient the camera
        pos = [ [10, 0, 0], [0, -10, 0], [0, 0, 10] ]
        up = [ [0, 0, 1], [0, 0, 1], [0, 1, 0] ]
        self._view.CameraPosition = pos[axis]
        self._view.CameraViewUp = up[axis]
        self._view.CameraFocalPoint = [0, 0, 0]
        self._view.InteractionMode = '2D'
        self._view.OrientationAxesVisibility = False

        # set dataset (and update pipeline)
        self.set_dataset(dataset)

        self.text.Text = f"%s Slice: {slice * subsampling_factor}" % ['X', 'Y', 'Z'][axis]

        self.update_active_swatches()
        self.update_client_state()
        log.info(f'{self.id}: setup complete {self._view.ViewSize}')
        # self._html_view.update()

    def update_pipeline(self):
        newly_created = self.create_pipeline()
        self.reset_camera(self._view)

    def create_pipeline(self):
        if hasattr(self, '_created_pipeline'):
            return False
        self._created_pipeline = True

        log.info(f'{self.id}: creating pipeline')
        slice_display = simple.Show(self.producer, self._view)
        simple.ColorBy(slice_display, ('POINTS', self.get_scalar_name()))
        slice_display.SetRepresentationType('Slice')
        slice_display.MapScalars = self.parent.get_map_scalars()
        if self.parent:
            slice_display.LookupTable = self.parent._lut
        self._slice_display = slice_display

        # label display
        self.text = simple.Text()
        self.text.Text = ''
        self.textDisplay = simple.Show(self.text, self._view)
        self.textDisplay.FontSize = 20
        self.textDisplay.FontFamily = 'Arial'

    def get_segments(self, category):
        swatches = [swatch for swatch in self._swatches if swatch.category == category]
        return [swatch.voi for swatch in swatches]

    def get_all_segments(self):
        result = {}
        for category in Segmentation.CATEGORIES:
            result[category] = self.get_segments(category)
        log.info(f'{self.id}: get_all_segments: {result}')
        return result

    def get_all_averages(self):
        result = {}
        for category in Segmentation.CATEGORIES:
            result[category] = self.get_average(category)
        log.info(f'{self.id}: get_all_average: {result}')
        return result

    def get_average(self, category):
        return self._get_average(\
            list(filter(lambda x: x.category == category, self._swatches)))

    def _get_world_event_position(self):
        interactor = self._view.GetInteractor()
        pos = interactor.GetEventPosition()
        picker = rc.vtkWorldPointPicker()
        picker.Pick(pos[0], pos[1], 0, self._view.GetRenderer())
        return picker.GetPickPosition()

    def _selection_callback(self, eventId: str):
        assert self.meta.is_raw()
        
        if eventId == 'LeftButtonPressEvent':
            iren = self._view.GetInteractor()
            if not iren.GetControlKey() and not iren.GetShiftKey():
                return
            self._selection_start = self._get_world_event_position()
            self._selection_end = self._selection_start
            self._selection_active = True
            swatch = Swatch(category=Segmentation.CATEGORIES[self._state['tab_index']], color=Segmentation.CATEGORY_RGB_COLORS[self._state['tab_index']])
            swatch.add_to_view(self._view)
            self._swatches.append(swatch)
            self._active_swatches.append(swatch)

        elif eventId == 'LeftButtonReleaseEvent':
            if self._selection_active:
                self._selection_active = False

        elif eventId == 'MouseMoveEvent':
            if self._selection_active:
                self._selection_end = self._get_world_event_position()

        if self._selection_active:
            bbox = vtk.vtkBoundingBox()
            bbox.AddPoint(self._selection_start)
            bbox.AddPoint(self._selection_end)

            # convert to VOI.
            spacing = self.dataset.GetSpacing()
            ext = self.dataset.GetExtent()
            origin = self.dataset.GetOrigin()

            voi = [0] * 6
            for i in range(3):
                voi[2*i] = clamp(math.floor((bbox.GetMinPoint()[i] - origin[i]) / spacing[i]), ext[2*i], ext[2*i+1])
                voi[2*i+1] = clamp(math.ceil((bbox.GetMaxPoint()[i] - origin[i]) / spacing[i]), ext[2*i], ext[2*i+1])
           
            self._active_swatches[-1].set_voi(voi, self.dataset)
            self._active_swatches[-1].show()
            self.update_client_state()
            self._html_view.update()

    def _delete_last(self, **kwargs):
        if len(self._active_swatches) > 0:
            self._active_swatches[-1].remove_from_view(self._view)
            self._swatches.remove(self._active_swatches.pop())
            self.update_active_swatches()
            self.update_client_state()
            self._html_view.update()

    def _delete_all(self, **kwargs):
        for swatch in self._active_swatches:
            swatch.remove_from_view(self._view)
            self._swatches.remove(swatch)
        self.update_active_swatches()
        self.update_client_state()
        self._html_view.update()

    def _reset(self, **kwargs):
        for swatch in self._swatches:
            swatch.remove_from_view(self._view)
        self._swatches = []
        self.update_active_swatches()
        self.update_client_state()
        self._html_view.update()


#----------------------------------------------------------------------------------------------------------------------
# register route to query segmentation data
from aiohttp import web

async def get_segments(request):
    result = {}
    for ref in Base._all_views_refs:
        if not ref: continue
        view =ref()
        if isinstance(view, Segmentation):
            result[view.id] = view.get_all_segments()
    return web.json_response(result)

async def get_averages(request):
    result = {}
    for ref in Base._all_views_refs:
        if not ref: continue
        view =ref()
        if isinstance(view, Segmentation):
            result[view.id] = view.get_all_averages()
    return web.json_response(result)

server = get_server()
@server.controller.add('on_server_bind')
def on_server_bind(server):
    server.app.add_routes([\
        web.get('/segments', get_segments),
        web.get('/averages', get_averages) ])

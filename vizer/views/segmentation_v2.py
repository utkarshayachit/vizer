from .base import Base
from vizer import utils

from trame.app import get_server
from trame.widgets import vuetify, html

from paraview import simple, vtk
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import vtkStructuredExtent
from vtkmodules.vtkFiltersSources import vtkOutlineSource
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
# from vtkmodules.vtkAcceleratorsVTKmFilters import vtkmExtractVOI as vtkExtractVOI
from vtkmodules.vtkImagingCore import vtkExtractVOI
import vtkmodules.vtkRenderingCore as rc

import math
import numpy
import pandas as pd

log = utils.get_logger(__name__)

class UIBuilder:
    def select_buttons(self, view, labels, colors):
        with vuetify.VBtnToggle(v_model=f'{view.id}_selection_mode'):
            for label, color in zip(labels, colors):
                with vuetify.VBtn(label, color=color, small=True, outlined=True):
                    vuetify.VIcon("mdi-select-drag")


HEADER_OPTIONS = {
    "type": { "text": "Type" },
    "text": { "text": "Label" },
}

CATEGORIES_INFO = [
    { "text": "Pore", "color": [10, 135, 255] },
    { "text": "Porous Material", "color": [100, 255, 100] },
    { "text": "Mineral", "color": [255, 220, 235] },
]
CATEGORIES = dict((i, info) for i, info in enumerate(CATEGORIES_INFO))
CATEGORIES_COUNT = len(CATEGORIES_INFO)

class Categories:
    @staticmethod
    def get_text(index):
        return CATEGORIES[index]['text']
    
    @staticmethod
    def get_color(index):
        return CATEGORIES[index]['color']
    
    @staticmethod
    def get_color_numpy(index, opacity=255):
        return numpy.array(CATEGORIES[index]['color'] + [opacity], dtype=numpy.uint8)
    
    @staticmethod
    def get_color_js(index):
        return f'rgb({CATEGORIES[index]["color"][0]}, {CATEGORIES[index]["color"][1]}, {CATEGORIES[index]["color"][2]})'
    

class Segment:
    """class for volume of interest"""
    EXTRACTOR = None
    
    def __init__(self, segment_type, anchor, axis, slice) -> None:
        # expecting 2D VOIs
        self._type = segment_type
        self._axis = axis
        self._slice = slice
        self._anchor = anchor
        self._dims = [1] * 3
        self._dims[self._axis] = 0
        self._average = None

    @property
    def text(self):
        return f'slice: {self._slice}, axis: {self._axis}, type: {self._type}, anchor: {self._anchor}, dims: {self._dims}'

    @property 
    def info(self):
        return f'{self._average:.1f}'

    @property
    def color(self):
        return Categories.get_color_js(self._type)
        
    def get_color_numpy(self, opacity=255):
        return Categories.get_color_numpy(self._type, opacity)
    
    @property
    def voi(self):
        voi = [0] * 6
        for i in range(3):
            a = self._anchor[i]
            b = a + self._dims[i] - 1
            voi[2*i] = min(a,b)
            voi[2*i+1] = max(a,b)
        return voi
    
    def compute_bounds(self, origin, spacing):
        """computes the bounds of the VOI in world coordinates"""
        bounds = [0, 0, 0, 0, 0, 0]
        voi = self.voi
        for i in range(3):
            bounds[2*i] = origin[i] + voi[2*i] * spacing[i]
            bounds[2*i+1] = origin[i] + voi[2*i+1] * spacing[i]
        return bounds

    def expand_to(self, ijk):
        """expands the VOI to include the given ijk point"""
        for i in range(3):
            if i == self._axis:
                continue
            self._dims[i] = ijk[i] - self._anchor[i] + 1
        # log.info(f'VOI: {self._anchor}, {self._dims}')

    def set_average(self, value):
        self._average = value


class SegmentsSource(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=0,
            nOutputPorts=1,
            outputType="vtkPolyData",
        )
        self._segments = []
        self._highlighted_segment = None
        self._axis = 2
        self._slice = 0
        self._origin = [0, 0, 0]
        self._spacing = [1, 1, 1]

    def SetSegments(self, segments):
        if self._segments != segments:
            self._segments = segments
            self.Modified()

    def HighlightSegment(self, index):
        if self._highlighted_segment != index:
            self._highlighted_segment = index
            self.Modified()

    def SetAxis(self, axis):
        if self._axis != axis:
            self._axis = axis
            self.Modified()

    def SetSlice(self, slice):
        if self._slice != slice:
            self._slice = slice
            self.Modified()

    def SetMetadata(self, origin, spacing):
        if self._origin != origin or self._spacing != spacing:
            self._origin = origin
            self._spacing = spacing
            self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        pd = vtk.vtkPolyData.GetData(outInfo, 0)
        pd.Initialize()

        box_source = vtkOutlineSource()
        box_source.GenerateFacesOn()

        appender = vtkAppendPolyData()

        idx = -1
        for seg in self._segments:
            idx += 1
            if seg is None or seg._slice != self._slice or seg._axis != self._axis:
                continue

            bds = seg.compute_bounds(self._origin, self._spacing)
            box_source.SetBounds(bds)
            box_source.Update()
            box = box_source.GetOutput().NewInstance()
            box.ShallowCopy(box_source.GetOutput())

            boxDS = dsa.WrapDataObject(box)
            opacity = 100 if idx == self._highlighted_segment else 40
            colors = numpy.linspace(seg.get_color_numpy(opacity), seg.get_color_numpy(opacity), boxDS.GetNumberOfCells(), dtype=numpy.uint8)
            boxDS.CellData.append(colors, 'colors')
            boxDS.CellData.SetActiveScalars('colors')
            appender.AddInputData(box)
        
        if appender.GetNumberOfInputConnections(0) == 0:
            return 1
        
        appender.Update()
        pd.ShallowCopy(appender.GetOutput())
        # log.info(f'active cell scalars {pd.GetCellData().GetScalars()}')
        return 1



class Segmentation(Base):
    def __init__(self, meta, opts, **kwargs) -> None:
        super().__init__(meta, opts)
        self._view = None
        self._html_view = None
        self._ui_builder = kwargs.get('ui_builder', UIBuilder())

        # values stored in _state is used by the UI on the client side
        # to render UI elements
        self._state = {}
        self._state['min'] = 0
        self._state['max'] = 0
        self._state['slice'] = -1
        self._state['axis'] = 2
        self._state['selected_segment'] = None
        self._state['selection_mode'] = None
        self._state['segments'] = []
        self._state['segment_totals'] = []

        self._segments = []
        self._active_segment = None

    @property
    def state(self):
        return self._state
    
    @property
    def subsampling_factor(self):
        # this view does not support subsampling
        return 1
    
    @staticmethod
    def can_show(meta):
        """returns true if the view can be show the dataset"""
        return meta.is_structured and not meta.is_empty()
    
    def update_client_state(self):
        """updates the client with the current state"""

        state = get_server().state
        new_state = {}
        for key, value in self.state.items():
            new_state[f'{self.id}_{key}'] = value

        new_segments = []
        if self._active_segment is not None:
            new_segments.append( { 'info': self._active_segment.info,
                                   'color': self._active_segment.color })
        new_segments += [{ 'info': s.info, 'color': s.color } for s in self._segments]
        new_segments = [ { 'id': i, **s } for i, s in enumerate(new_segments) ]

        frame = pd.DataFrame.from_dict(new_segments)
        headers, rows = vuetify.dataframe_to_grid(frame, HEADER_OPTIONS)
        new_state[f'{self.id}_segments'] = rows

        totals = [{ 'info': Categories.get_text(i), 'color': Categories.get_color_js(i), 'avg': self.compute_category_average(i)} for i in range(CATEGORIES_COUNT)]

        new_state[f'{self.id}_segment_totals'] = totals

        # log.info(f'{self.id}: updating client state: {new_state}')
        state.update(new_state)
        state.flush()

    def create_render_view(self):
        view = super().create_render_view()
        view.InteractionMode = '2D'
        view.OrientationAxesVisibility = False
        view.Camera2DManipulators = ['Pan', 'None', 'Zoom'] * 2 + ['Zoom', 'None', 'Pan']

        iren = view.GetInteractor()
        iren.AddObserver('LeftButtonPressEvent', lambda *_: self.on_left_button_press())
        iren.AddObserver('LeftButtonReleaseEvent',  lambda *_: self.on_left_button_release())
        iren.AddObserver('MouseMoveEvent', lambda *_: self.on_mouse_move())
        return view
    
    def create_widget(self):
        self.update_client_state()
        with self.layout.viewport:
            with vuetify.VRow(no_gutters=True, classes="grow"):
                with vuetify.VCol(cols=9):
                    self._view = self.create_render_view()
                    self._html_view = self.create_html_view(self._view)

                with vuetify.VCol(cols=3):
                    with vuetify.VCard(classes="ma-2 py-4", tile=True, outlined=True): # classes="pa-0 ma-0 fill-height"):
                        vuetify.VCardTitle("Segmentation")
                        vuetify.VCardText("Click on a segment button to select it and then click and drag to define a segment."
                                          "Segments are defined on the shown slice which can be changed using the Slice widgets.")
                        # vuetify.VCardSubtitle("Select Segment")
                        with vuetify.VCardText():
                            with vuetify.VBtnToggle(v_model=f'{self.id}_selection_mode'):
                                with vuetify.VDataTable(
                                    headers=(f'{self.id}_headers', [ 
                                        {'text': 'Total', 'value': 'avg'},
                                        {'text': 'Info', 'value': 'info'}, ]),
                                    items=(f'{self.id}_segment_totals', []),
                                    no_data_text='No segments selected',
                                    hide_default_footer=True,
                                    hide_default_header=True,
                                    items_per_page=None,
                                    class_='elevation-1',
                                    dense=True,
                                ):
                                    with vuetify.Template(v_slot_item_info="{ item }",
                                                          __properties=[("v_slot_item_info", "v-slot:item.info")]):
                                        with vuetify.VBtn(color=("item.color",), small=True, outlined=True):
                                            vuetify.VIcon("mdi-select-drag", left=True)
                                            html.Pre("{{ item.info }}")
                        # vuetify.VCardSubtitle("Select slice and axis")
                        with vuetify.VCardText():
                            vuetify.VSelect(label="Slice Direction",
                                            v_model=(f'{self.id}_axis',0),
                                            items=(f'{self.id}_items', [
                                                {'text': 'D1', 'value': 0},
                                                {'text': 'D2', 'value': 1},
                                                {'text': 'D3', 'value': 2}
                                            ]),
                                            item_text='text',
                                            item_value='value')
                            
                            with vuetify.VSlider(dense=True, hide_details=False,
                                   min=(f'{self.id}_min',0), max=(f'{self.id}_max',0),
                                   v_model=(f'{self.id}_slice',0)):
                                with vuetify.Template(v_slot_append=True):
                                    vuetify.VTextField(v_model=(f'{self.id}_slice',0),
                                                       dense=True,
                                                       type="number",
                                                       style="width: 70px;")


                    with vuetify.VCard(classes="ma-2", tile=True, outlined=True): # classes="pa-0 ma-0 fill-height"):
                        vuetify.VCardTitle("Selected Segments")
                        with vuetify.VCardText():
                            with vuetify.VBtnToggle(v_model=f'{self.id}_selected_segment'):
                                with vuetify.VDataTable(
                                    headers=(f'{self.id}_headers', [
                                        {'text': 'Segment', 'value': 'info'},
                                    ]),
                                    items=(f'{self.id}_segments', []),
                                    no_data_text='No segments selected',
                                    hide_default_header=True,
                                    hide_default_footer=True,
                                    items_per_page=None,
                                    class_='elevation-1',
                                    dense=True,
                                ):
                                    with vuetify.Template(v_slot_item_info="{ item }",
                                                          __properties=[("v_slot_item_info", "v-slot:item.info")]):
                                        with vuetify.VBtn(color=("item.color",), small=True, outlined=True):
                                            vuetify.VIcon("mdi-checkbox-blank-outline", left=True)
                                            html.Pre("{{ item.info }}")
                                        vuetify.VIcon("mdi-delete", right=True, color="red",
                                                      click=(self.delete_segment, "[item.id]"))
                                                    #   click=(delete_segment_callback, 1, "item"))
 
    def update_pipeline(self, reset_camera=False):
        log.info(f'{self.id}: updating pipeline')
        
        ext = self.producer.GetDataInformation().GetExtent()
        newly_created = self.create_pipeline()

        if newly_created:
            # setup defaults based on data information
            self._state['min'] = ext[2*self.state['axis']]
            self._state['max'] = ext[2*self.state['axis']+1]
            self._state['slice'] = math.floor((self._state['min'] + self._state['max']) / 2)

            self.update_color_map()

        # update pipeline state.
        voi = list(ext)
        voi[2*self.state['axis']] = self.state['slice']
        voi[2*self.state['axis']+1] = self.state['slice']
        self._slice.VOI = voi

        # update segments pipeline
        segs = self._segments if self._active_segment is None else [self._active_segment] + self._segments
        self._segments_source.SetSegments(segs)
        self._segments_source.SetAxis(self._state['axis'])
        self._segments_source.SetSlice(self._state['slice'])
        self._segments_source.HighlightSegment(self._state['selected_segment'])
        if self.dataset:
            self._segments_source.SetMetadata(self.dataset.GetOrigin(),
                                              self.dataset.GetSpacing())

        if newly_created or reset_camera:
            axis = self._state['axis']
            pos = [ [10, 0, 0], [0, -10, 0], [0, 0, 10] ]
            up = [ [0, 0, 1], [0, 0, 1], [0, 1, 0] ]
            self._view.CameraPosition = pos[axis]
            self._view.CameraViewUp = up[axis]
            self._view.CameraFocalPoint = [0, 0, 0]
            self.reset_camera(self._view)
            self._view.StillRender()
        self.update_client_state()
        self._html_view.update()

    def create_pipeline(self):
        if hasattr(self, '_created_pipeline'):
            return False
        self._created_pipeline = True

        log.info(f'{self.id}: creating pipeline')

        # setup color map we'll use for this view
        self._lut = simple.GetColorTransferFunction(f'{self.id}_lut')
        self._lut.ApplyPreset('Grayscale', True)
        self._lut.RGBPoints = [0, 0.2, 0.2, 0.2, 1, 1, 1, 1]

        ext = self.producer.GetDataInformation().GetExtent()
        voi = list(ext)
        voi[2*self.state['axis']] = ext[2*self.state['axis']]
        voi[2*self.state['axis']+1] = ext[2*self.state['axis']]

        # log.info(f'{self.id}: Initial VOI: {voi}')
        self._slice = simple.ExtractVOI(Input=self.producer, VOI=voi)
        self._color_mapyer = simple.ColorMappyer(Input=self._slice)
        self._display = simple.Show(self._color_mapyer, self._view)
        simple.ColorBy(self._display, ('POINTS', self.get_scalar_name()))
        self._display.SetRepresentationType('Slice')
        self._display.LookupTable = self._lut
        self._display.MapScalars = 1 # FIXME

        self._segments_source = SegmentsSource()
        mapper = rc.vtkPolyDataMapper2D()
        mapper.SetInputConnection(self._segments_source.GetOutputPort())
        mapper.SetScalarVisibility(True)
        mapper.SetColorModeToDirectScalars()
        mapper.SetScalarModeToUseCellData()
        # mapper.ColorByArrayComponent('colors', 0)

        # specify transform coordinate system since data is in world coordinates
        # and not viewport coordinates
        coord = rc.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        mapper.SetTransformCoordinate(coord)

        actor = rc.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(4)
        # actor.GetProperty().SetColor(1, 0, 0)
        # actor.GetProperty().SetOpacity(0.2)

        pvview =  self._view.GetClientSideObject()
        pvview.GetRenderer(pvview.NON_COMPOSITED_RENDERER).AddActor(actor)

        state = get_server().state
        @state.change(f'{self.id}_slice')
        def slice_changed(**kwargs):
            if kwargs[f'{self.id}_slice'] != self._state['slice']:
                self.update_slice(axis=self._state['axis'], slice=kwargs[f'{self.id}_slice'])
                self._html_view.update()

        @state.change(f'{self.id}_axis')
        def axis_changed(**kwargs):
            if kwargs[f'{self.id}_axis'] != self._state['axis']:
                self.update_slice(axis=kwargs[f'{self.id}_axis'], slice=None)
                self._html_view.update()

        @state.change(f'{self.id}_selection_mode')
        def selection_mode_changed(**kwargs):
            self.set_selection_mode(kwargs[f'{self.id}_selection_mode'])

        @state.change(f'{self.id}_selected_segment')
        def selected_segment_changed(**kwargs):
            if kwargs[f'{self.id}_selected_segment'] != self._state['selected_segment']:
                self.select_segment(kwargs[f'{self.id}_selected_segment'])
                self._html_view.update()

        return True

    def update_slice(self, axis, slice):
        assert axis is not None
        ext = self.producer.GetDataInformation().GetExtent()

        # need to reset camera if axis changes
        reset_camera = self._state['axis'] != axis
        self._state['axis'] = axis
        self._state['min'] = ext[2*axis]
        self._state['max'] = ext[2*axis+1]
        if slice is not None:
            self._state['slice'] = slice
        else:
            # reset slice to center of volume
            self._state['slice'] = math.floor((self._state['min'] + self._state['max']) / 2)
        self.update_pipeline(reset_camera=reset_camera)

    def set_selection_mode(self, mode):
        self._state['selection_mode'] = mode
        if mode is not None:
            # disable all interaction
            self._view.Camera2DManipulators = ['None'] * 3 + ['Pan', 'None', 'Zoom', 'Zoom', 'None', 'Pan']
        else:
            # restore interaction
            self._view.Camera2DManipulators = ['Pan', 'None', 'Zoom'] * 2 + ['Zoom', 'None', 'Pan']
        self.update_client_state()

    def select_segment(self, index):
        self._state['selected_segment'] = index
        segs = self._segments if self._active_segment is None else [self._active_segment] + self._segments
        if index is not None and index >= 0 and index < len(segs):
            seg = segs[index]
            self.update_slice(seg._axis, seg._slice)
        else:
            self.update_pipeline()

    def delete_segment(self, index):
        log.info(f'{self.id}: deleting segment {index}')
        assert self._active_segment is None # can't be interactively modifying a segment when a delete happens!
        self._segments.pop(index)
        if self._state['selected_segment'] == index:
            self._state['selected_segment'] = None
        # HACK: force modify the segments_source since it currently does not detect
        # changes to the segments
        self._segments_source.Modified()
        self.update_pipeline()

    def _get_event_position_ijk(self):
        interactor = self._view.GetInteractor()
        pos = interactor.GetEventPosition()
        picker = rc.vtkWorldPointPicker()
        picker.Pick(pos[0], pos[1], 0, self._view.GetRenderer())
        xyz = picker.GetPickPosition()

        # convert to ijk
        ijk = [0, 0, 0]

        ext = self.producer.GetDataInformation().GetExtent()
        bounds = self.producer.GetDataInformation().GetBounds()

        origin = [ext[0], ext[2], ext[4]]
        spacing = [ (bounds[1] - bounds[0]) / (ext[1] - ext[0] + 1),
                    (bounds[3] - bounds[2]) / (ext[3] - ext[2] + 1),
                    (bounds[5] - bounds[4]) / (ext[5] - ext[4] + 1) ]
        for i in range(3):
            ijk[i] = math.floor((xyz[i] - origin[i]) / spacing[i])

        ijk[self._state['axis']] = self._state['slice']
        # log.info(f'Picked position: {xyz} -> {ijk}')
        return ijk

    def on_left_button_press(self):
        if self._state['selection_mode'] is None:
            return
        
        # start a new segment
        # NOTE: ijk is not clamped to volume bounds...should we?
        ijk = self._get_event_position_ijk()

        segment = Segment(segment_type=self._state['selection_mode'],
                    anchor=ijk,
                    axis=self._state['axis'],
                    slice=self._state['slice'])
        segment.set_average(self.compute_segment_average(segment))
        self._active_segment = segment
        self._state['selected_segment'] = 0 # change the selected segment to the active one
        self.update_pipeline()

    def on_left_button_release(self):
        if self._state['selection_mode'] is None:
            return

        self._segments.insert(0, self._active_segment)
        self._active_segment = None

    def on_mouse_move(self):
        if self._state['selection_mode'] is None:
            return
        
        if self._active_segment is None:
            return
        
        ijk = self._get_event_position_ijk()
        self._active_segment.expand_to(ijk)
        self._active_segment.set_average(self.compute_segment_average(self._active_segment))
        # log.info(f'Active segment: {self._active_segment.text}')

        # HACK: force modify the segments_source since it currently does not detct
        # changes to the segments
        self._segments_source.Modified()
        self.update_pipeline()

    def compute_segment_average(self, segment):
        voi = segment.voi
        vtkStructuredExtent.Clamp(voi, self.producer.GetDataInformation().GetExtent())
        if voi[0] > voi[1] or voi[2] > voi[3] or voi[4] > voi[5]:
            return 0

        if not hasattr(self, 'extractor'):
            self.extractor = vtkExtractVOI()
        self.extractor.SetInputData(self.dataset)
        self.extractor.SetVOI(voi)
        self.extractor.Update()
        self.extractor.SetInputData(None)
        data = dsa.WrapDataObject(self.extractor.GetOutputDataObject(0))
        scalars = data.PointData[data.GetPointData().GetScalars().GetName()]
        if scalars is None or scalars.shape[0] == 0:
            return 0
        return numpy.mean(scalars)
    
    def compute_category_average(self, category):
        segs = self._segments if self._active_segment is None else [self._active_segment] + self._segments
        avg = [s._average for s in  filter(lambda s: s._type == category, segs)]
        return int(numpy.mean(avg) if len(avg) > 0 else 0)


    def update_color_map(self):
        log.info(f'{self.id}: updating color map')

        if self.meta.raw_config and self.meta.raw_config.colormap is not None:
            log.info(f'{self.id}: using categorical color map (with color_mappyer)')
            # update color mapyer
            # using this direct API call since the XML wrapping for this is broken
            self._color_mapyer.GetClientSideObject().SetColors(self.meta.raw_config.colormap['color'].flatten())
            self._color_mapyer.GetClientSideObject().SetScalars(self.meta.raw_config.colormap['scalar'].flatten())
            self._display.MapScalars = 0
        else:
            drange = self.producer.GetDataInformation().GetArrayInformation(self.get_scalar_name(), vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS).GetComponentRange(0)
            if drange[0] != drange[1]:
                ds = dsa.WrapDataObject(self.dataset)
                array = ds.PointData[self.get_scalar_name()]
                percentiles =numpy.percentile(array, [5, 95])
                drange = [percentiles[0], percentiles[1]]
            log.info('5/95 percentile: %f/%f', drange[0], drange[1])
            self._lut.InterpretValuesAsCategories = False
            self._lut.ApplyPreset('Grayscale', True)
            self._lut.RGBPoints = [0, 0.2, 0.2, 0.2, 1, 1, 1, 1]
            self._lut.RescaleTransferFunction(drange[0], drange[1])
            self._color_mapyer.Colors = []
            self._color_mapyer.Scalars = []

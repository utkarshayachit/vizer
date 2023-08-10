
def get_view_types():
    """returns the list of supported view types."""
    return ['quad', 'simple', 'blank', 'crop', 'segmentation']

def create(meta, opts):
    """Creates a view from a dataset metadata.
    Returns None if the file is not supported. Otherwise, an appropriate
    subclass of base.Base is returned.
    """
    from .simple import Simple
    from .quad import Quad
    from .blank import Blank
    from .crop import Crop
    from .segmentation_v2 import Segmentation as SegmentationV2

    if opts.force_view == 'quad':
        return Quad(meta, opts)
    elif opts.force_view == 'simple':
        return Simple(meta, opts)
    elif opts.force_view == 'blank':
        return Blank(meta, opts)
    elif opts.force_view == 'crop':
        return Crop(meta, opts)
    elif opts.force_view == 'segmentation':
        return SegmentationV2(meta, opts)

    if Quad.can_show(meta):
        return Quad(meta, opts)
    elif Simple.can_show(meta):
        return Simple(meta, opts)

    return Blank(meta, opts)

from .graph import disparity as disparity_graph
from .ssd import disparity as disparity_ssd

METHOD_SSD='ssd'
METHOD_GRAPHCUT='graphcut'

DISPARITY_METHODS = {
    METHOD_SSD: disparity_ssd,
    METHOD_GRAPHCUT: disparity_graph,
}

def disparity(image_left, image_right, **kwargs):
    method = kwargs.pop('method')
    disparity_method = DISPARITY_METHODS.get(method, lambda: None)
    if disparity_method is None:
        raise Error('invalid method {}'.format(method))

    return disparity_method(image_left, image_right, **kwargs)

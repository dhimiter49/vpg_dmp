from .base_projection_layer import BaseProjectionLayer
from .frob_projection_layer import FrobeniusProjectionLayer
from .kl_projection_layer import KLProjectionLayer
from .papi_projection import PAPIProjection
from .w2_projection_layer import WassersteinProjectionLayer
from .w2_projection_layer_non_com import WassersteinProjectionLayerNonCommuting

MAP_TR_LAYER = {
    "base": BaseProjectionLayer,
    "wasserstein": WassersteinProjectionLayer,
    "kl": KLProjectionLayer,
    "frobenius": FrobeniusProjectionLayer,
    "papi": PAPIProjection,
    "wasserstein_noncomm": WassersteinProjectionLayerNonCommuting,
}

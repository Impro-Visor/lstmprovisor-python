import theano
import os
import theano.tensor as T
import numpy as np
from theano_lstm import MultiDropout

def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    return T.repeat(T.shape_padleft(vector), n, axis=0)

def initial_state(layer, dimensions = None):
    """
    Initalizes the recurrence relation with an initial hidden state
    if needed, else replaces with a "None" to tell Theano that
    the network **will** return something, but it does not need
    to send it to the next step of the recurrence
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None

def initial_state_with_taps(layer, dimensions = None):
    """Optionally wrap tensor variable into a dict with taps=[-1]"""
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None


def get_last_layer(result):
    if isinstance(result, list):
        return result[-1]
    else:
        return result

def ensure_list(result):
    if isinstance(result, list):
        return result
    else:
        return [result]

def UpscaleMultiDropout(shapes, dropout = 0.):
    """
    Return all the masks needed for dropout outside of a scan loop.
    """
    orig_masks = MultiDropout(shapes, dropout)
    fixed_masks = [m / (1-dropout) for m in orig_masks]
    return fixed_masks

class _SliceHelperObj(object):
    """
    Helper object that exposes the slice from __getitem__ directly
    """
    def __getitem__(self, key):
        return key

sliceMaker = _SliceHelperObj()

def _better_print_fn(op, xin):
    for item in op.attrs:
        if callable(item):
            pmsg = item(xin)
        else:
            temp = getattr(xin, item)
            if callable(temp):
                pmsg = temp()
            else:
                pmsg = temp
        print(op.message, attr, '=', pmsg)

def FnPrint(name, items=['__str__']):
    return theano.printing.Print(name, items, _better_print_fn)

def Save(path="", preprocess=lambda x:x, text=False):
    def _save_fn(op, xin):
        val = preprocess(xin)
        if text:
            np.savetxt(path + ".csv", val, delimiter=",")
        else:
            np.save(path + ".npy", val)
    return theano.printing.Print(path, [], _save_fn)

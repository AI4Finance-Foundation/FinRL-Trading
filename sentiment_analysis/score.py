import numpy as np


def calc_alpha(window, weight_proportion):
    '''
    Calculate alpha parameter for exponentially weighted moving average
    
    :param window: number of values that weights will add up to the weight_proportion
    :param weight_proportion: float [0,1], gives the amount of cumulative weight given to last window values
    :return: alpha parameter for ewma
    '''
    return 1 - np.exp(np.log(1-weight_proportion)/window)


def get_weights(alpha,num_steps):
    '''
    Shows weights of last num_steps values
    '''
    return [alpha] + [alpha*(1-alpha)**i for i in range(1,num_steps)]

def update_ewma(prev_stat, data_point, alpha):
    '''
    Updates the exponentially weighted moving average given a new data_point and parameter alpha
    '''
    return data_point*alpha + (1-alpha) * prev_stat

# This can be used for calculating the ewma given a vector as a start. Safe for large sizes of input
def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    Params:
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    :return out
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out


    initial_avg = vals[0]
    get_weights(alpha,len(vals))



def test_calc_alpha():
    alpha = (calc_alpha(7,.99))
    weights = get_weights(alpha, 10)
    print(alpha)
    print(weights)
    print(sum(weights))

def test_update_ewma():
    alpha = (calc_alpha(10,0.9))
    scores = np.array([-1,-1,-1,0,0,0,0,1,1,1,1,1,1,1,1])
    avg = scores[0]
    ewmas = [avg]
    for i in range(1,len(scores)):
        avg = (update_ewma(avg,scores[i],alpha))
        ewmas.append(avg)
    print(ewmas)



if __name__ == "__main__":
    test_update_ewma()
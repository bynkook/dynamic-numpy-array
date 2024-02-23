import numpy as np

class DynArrNp1d:

    # note:
    # only for 1-D numpy array
    # always return 0 for out-of-bound index
    #
    # naming convention:
    # _(name of function) : local function
    # (name of variable)_ : local variable

    def __init__(self, size, dtype):

        self.dtype = dtype  # int32 or float32
        self.n = size       # avoid frequent call len(), .shape[]
        self.A = self._make(self.n)

    def __len__(self):

        return self.n

    def __str__(self):

        data_str = np.array2string(self.A, separator=', ', formatter={'float_kind':lambda x: "%g" % x})
        return f"{data_str}"

    def __getitem__(self, key):

        try:
            self._dynamic_array_resize(key)
        except Exception as err:
            print(err)
        return self.A[key]

    def __setitem__(self, key, value):

        try:
            self._dynamic_array_resize(key)
        except Exception as err:
            print(err)
        self.A[key] = value

    def _get_indices(self, key):

        # handling None index
        if key.start == None: istart_ = 0
        else: istart_ = key.start
        if key.stop == None: istop_ = 0
        else: istop_ = key.stop
        if key.step == None: istep_ = 1
        else: istep_ = key.step
        key = slice(istart_, istop_, istep_)

        # this is to handle negative index
        istart_, istop_, istep_ = key.indices(self.n)
        key_ = slice(istart_, istop_, istep_)

        return key, key_

    def _dynamic_array_resize(self, key):
        
        # this is to handle None, negative and out-of-bound index
        if isinstance(key, slice):
            key0, key_ = self._get_indices(key)
            maxindex_ = max(abs(key0.start), abs(key0.stop), \
                            abs(key_.start), abs(key_.stop))
            while maxindex_ + 1 > self.n:
                self._resize(int(2 * self.n))

        # this is to handle list, numpy array as index
        elif isinstance(key, (list, tuple, np.ndarray)):
            key_ = np.asarray(key)
            key_[key_ < 0] += self.n
            while (key_ > self.n - 1).any():
                self._resize(int(2 * self.n))
        
        # handling negative single index
        else:            
            if key < 0:
                key += self.n
            while key + 1 > self.n:
                self._resize(int(2 * self.n))

    def _resize(self, size):

        self.A = np.pad(self.A, (0, size))
        self.n = len(self.A)

    def _make(self, size):

        return np.zeros(size, self.dtype)

    def append(self, value):

        self.A = np.append(self.A, value)
        self.n = len(self.A)

    def clear(self):

        self.A = np.zeros_like(self.A)

    def trim(self):

        self.A = np.trim_zeros(self.A, 'b')
        self.n = len(self.A)

class DynArrNp2d:

    # note:
    # only for 2-D numpy array
    # always return 0 for out-of-bound index
    #
    # naming convention:
    # _(name of function) : local function
    # (name of variable)_ : local variable

    def __init__(self, shape, dtype):

        self.dtype  = dtype     # int32 or float32
        self.size   = shape     # tuple, e.g., (2, 3)
        self.A      = self._make()

    def __str__(self):

        data_str = np.array2string(self.A, separator=', ', formatter={'float_kind':lambda x: "%g" % x})
        return f"{data_str}"

    def __getitem__(self, key):

        try:
            self._dynamic_array_resize(key)
        except Exception as err:
            print(err)
        return self.A[key]

    def __setitem__(self, key, value):

        try:
            self._dynamic_array_resize(key)
        except Exception as err:
            print(err)
        self.A[key] = value

    def _get_indices(self, axis, key):

        # handling None index
        if key.start == None: istart_ = 0
        else: istart_ = key.start
        if key.stop == None: istop_ = 0
        else: istop_ = key.stop
        if key.step == None: istep_ = 1
        else: istep_ = key.step
        key = slice(istart_, istop_, istep_)

        # this is to handle negative index
        istart_, istop_, istep_ = key.indices(self.size[axis])
        key_ = slice(istart_, istop_, istep_)

        return key, key_
    
    def _check_idx_2d(self, axis, idx):

        if isinstance(idx, slice):
            key0, key1 = self._get_indices(axis, idx)
            maxindex_ = max(abs(key0.start), abs(key0.stop), \
                            abs(key1.start), abs(key1.stop))
            while maxindex_ + 1 > self.size[axis]:
                self._resize(axis, int(2*self.size[axis]))
        else:
            # handling negative single index
            if idx < 0:
                idx += self.size[axis]
            while idx + 1 > self.size[axis]:
                self._resize(axis, int(2*self.size[axis]))

    def _dynamic_array_resize(self, key):
        
        key_ = np.asarray(key)        
        if isinstance(key, list):
            # when key is list, all elements are row index
            for elem in key_.flatten():
                if elem > self.size[0]:
                    self._resize(0, int(2*self.size[0]))
        elif isinstance(key, tuple):            
            for axis, idx in enumerate(key):
                if isinstance(idx, (list, tuple, np.ndarray)):
                    for val in idx:
                        self._check_idx_2d(axis, val)
                else:
                    self._check_idx_2d(axis, idx)
        else:
            raise IndexError
        
    def _resize(self, axis, size):

        if axis == 0: self.A = np.pad(self.A, ((0, size),(0, 0)))    # pad row
        if axis == 1: self.A = np.pad(self.A, ((0, 0),(0, size)))    # pad col
        self.size = tuple(map(int, self.A.shape))

    def _make(self):

        return np.zeros(self.size, self.dtype)

    def clear(self):

        self.A = np.zeros_like(self.A)

if __name__ == "__main__":

    # test numpy 1d array
    arr = DynArrNp1d(10, "int32")
    arr[11:15] += [10,20,30,40]
    print(arr)

    # test numpy 2d array
    arr = DynArrNp2d((4,4), "float32")    
    arr[6,0:6] = 10.123
    arr[(4,7,10),(10,12,14)] = 20.123
    print(arr)
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
        self.n = size       # just for speed, avoid frequent call len(), .shape[]
        self.A = self._make(self.n)

    def __len__(self):

        return self.n

    def __str__(self):

        data_str = np.array2string(self.A, separator=', ', formatter={'float_kind':lambda x: "%g" % x})
        return f"{data_str}"

    def __getitem__(self, key):

        print("#### __getitem__ ####")
        try:
            self._dynamic_array_resize(key)
        except Exception as err:
            print("Error : index out of bound for the key =  ", key)
            print(err)
        # return array value with original key(not clipped)
        return self.A[key]

    def __setitem__(self, key, value):

        print("#### __setitem__ ####")
        try:
            self._dynamic_array_resize(key)
        except Exception as err:
            print("Error : index out of bound for the key =  ", key)
            print(err)
        # set value to array with original key(not clipped)
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
        print("Info : slice input =  ", key)

        # this is to handle negative index
        # key_ is clipped slice to self.n length
        istart_, istop_, istep_ = key.indices(self.n)
        key_ = slice(istart_, istop_, istep_)
        print("Info : slice input normalized =  ", key_)

        return key, key_

    def _dynamic_array_resize(self, key):
        
        # this is to handle None, negative and out-of-bound index
        if isinstance(key, slice):            
            # find maximum of slice size
            key0, key_ = self._get_indices(key)
            maxindex_ = max(abs(key0.start), abs(key0.stop), \
                            abs(key_.start), abs(key_.stop))
            while maxindex_ + 1 > self.n:
                self._resize(int(2 * self.n))
                ### this is for debugging only
                _, key_ = self._get_indices(key)
                print("Info : slice input normalized after array resized =  ", key_)
        # this is to handle list, numpy array as index
        elif isinstance(key, (list, np.ndarray)):
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
        print("Info : array resized")

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

        print("#### __getitem__ ####")
        for axis, idx in enumerate(key):
            try:
                # this is to handle None, negative and out-of-bound index
                if isinstance(idx, slice):
                    self._dynamic_array_resize(axis, idx)
                # handling negative single index
                else:
                    if idx < 0:
                        idx += self.size[axis]
                    while idx + 1 > self.size[axis]:
                        self._resize(axis, int(2*self.size[axis]))
            except Exception as err:
                print(f"Error : index out of bound for key = {key}")
                print(err)
        # return value with original key(not clipped)
        return self.A[key]

    def __setitem__(self, key, value):

        print("#### __setitem__ ####")
        for axis, idx in enumerate(key):
            try:
                # this is to handle None, negative and out-of-bound index
                if isinstance(idx, slice):                    
                    self._dynamic_array_resize(axis, idx)
                # handling negative single index
                else:                    
                    if idx < 0:
                        idx += self.size[axis]
                    while idx + 1 > self.size[axis]:
                        self._resize(axis, int(2*self.size[axis]))
            except Exception as err:
                print(f"Error : index out of bound for key = {key}")
                print(err)
        # set value to array with original key(not clipped)
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
        print("Info : slice input =  ", key)

        # this is to handle negative index
        # key_ is clipped slice to self.n length
        istart_, istop_, istep_ = key.indices(self.size[axis])
        key_ = slice(istart_, istop_, istep_)
        print("Info : slice input normalized =  ", key_)

        return key, key_

    def _dynamic_array_resize(self, axis, key):
        key0, key_ = self._get_indices(axis, key)
        # find maximum of slice size
        maxindex_ = max(abs(key0.start), abs(key0.stop), \
                        abs(key_.start), abs(key_.stop))
        while maxindex_ + 1 > self.size[axis]:
            self._resize(axis, int(2*self.size[axis]))
            ### this is for debugging only
            _, key_ = self._get_indices(axis, key)
            print("Info : slice input normalized after array resized =  ", key_)

    def _resize(self, axis, size):

        if axis == 0: self.A = np.pad(self.A, ((0, size),(0, 0)))    # pad row
        if axis == 1: self.A = np.pad(self.A, ((0, 0),(0, size)))    # pad col
        self.size = tuple(map(int, self.A.shape))
        print("Info : array resized")

    def _make(self):

        return np.zeros(self.size, self.dtype)

    def clear(self):

        self.A = np.zeros_like(self.A)

if __name__ == "__main__":

    #
    # test numpy 1d array
    #

    arr = DynArrNp1d(10, "int32")
    arr[50:54] += [10,20,30,40]
    arr[-150:-154:-1] += [-10,-20,-30,-40]
    arr.trim()

    arr = DynArrNp1d(10, "float32")
    key = np.arange(30, 50)
    key = [20, 21, 22]    
    arr[key]
    arr[key] = 99.1

    #
    # test numpy 2d array
    #

    arr = DynArrNp2d((4,4), "float32")
    arr[0,7] = 100
    arr = DynArrNp2d((4,4), "float32")
    arr[:9,5:7] = 11
    arr = DynArrNp2d((4,4), "float32")
    arr[3:6,3:6] = 1
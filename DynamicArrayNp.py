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
        self.n = size       # just for speed, avoid frequent call len()
        self.A = self._make(self.n)

    def __len__(self):

        return self.n

    def __str__(self):

        data_str = np.array2string(self.A, separator=', ', formatter={'float_kind':lambda x: "%g" % x})
        return f"{data_str}"

    def __getitem__(self, key):

        print("#### __getitem__ ####")
        try:
            if isinstance(key, slice):
                # this is to handle None, negative and out-of-bound index
                self._dynamic_array_resize(key)
            else:
                # handling negative single index
                if key < 0:
                    key += self.n
                # resize array if required
                while key + 1 > self.n:
                    self._resize(int(2 * self.n))
        except Exception as err:
            print("Error : index out of bound for the key =  ", key)
            print(err)
        # return array value with original key(not clipped)
        return self.A[key]

    def __setitem__(self, key, value):

        print("#### __setitem__ ####")
        try:
            if isinstance(key, slice):
                # this is to handle None, negative and out-of-bound index
                self._dynamic_array_resize(key)
            else:
                # handling negative single index
                if key < 0:
                    key += self.n
                # resize array if required
                while key + 1 > self.n:
                    self._resize(int(2 * self.n))
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
        key0, key_ = self._get_indices(key)
        # find maximum of slice size
        maxindex_ = max(abs(key0.start), abs(key0.stop), \
                        abs(key_.start), abs(key_.stop))
        while maxindex_ + 1 > self.n:
            self._resize(int(2 * self.n))
            ### this is for debugging only
            _, key_ = self._get_indices(key)
            print("Info : slice input normalized after array resized =  ", key_)

    def _resize(self, size):

        # np.resize 는 array 값을 반복하기 때문에 사용하지 않는다.
        # np.pad 는 확장된 공간을 0으로 채워 사이즈를 증가시킨다.
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

    def trimtrailzero(self):

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
                if isinstance(idx, slice):
                    # this is to handle None, negative and out-of-bound index
                    self._dynamic_array_resize(axis, idx)
                else:
                    # handling negative single index
                    if idx < 0:
                        idx += self.size[axis]
                    # resize array if required
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
                if isinstance(idx, slice):
                    # this is to handle None, negative and out-of-bound index
                    self._dynamic_array_resize(axis, idx)
                else:
                    # handling negative single index
                    if idx < 0:
                        idx += self.size[axis]
                    # resize array if required
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

        # np.resize 는 array 값을 반복하기 때문에 사용하지 않는다.
        # np.pad 는 확장된 공간을 0으로 채워 사이즈를 증가시킨다.
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
    # this is to see numpy 1d array working
    #

    arr = DynArrNp1d(10,"float32")
    print(f"array length = {len(arr)}\n", arr)
    arr[50:54] += [10,20,30,40]
    print(f"array length = {len(arr)}\n", arr)
    arr[-150:-154:-1] += [-10,-20,-30,-40]
    print(f"array length = {len(arr)}\n", arr)
    arr.trimtrailzero()
    print(f"array length = {len(arr)}\n", arr)
    
    #
    # this is to see numpy 2d array working
    #

    arr = np.arange(16).reshape(4,4)
    print(arr)
    print(arr.size)
    print(arr.shape)

    arr = DynArrNp2d((4,4), "float32")
    print(arr)
    print(arr.size)
    arr._resize(0, 5)
    print(arr)
    print(arr.size)
    arr._resize(1, 2)
    print(arr)
    print(arr.size)
    arr[0,7] = 100
    print(arr)
    print(arr.size)
    arr[0,19] = 200
    print(arr)
    print(arr.size)
    arr.clear()
    print(arr)
    print(arr.size)
    arr = DynArrNp2d((4,4), "float32")
    print(arr)
    print(arr.size)
    arr[:9,5:7] = 11
    print(arr)
    print(arr.size)

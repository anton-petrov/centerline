import pickle
from shapely.geometry import Polygon
from centerline.geometry import Centerline
import time


def st_time(func):
    """
        st decorator to calculate the total time of a func
    """

    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        print("Function=%s, Time=%s" % (func.__name__, t2 - t1))
        return r

    return st_func


# 'poligon_filered_all_lines_simple'
file_name = 'poligon_filered_all_lines_simple'

with open(file_name + '.pkl', 'rb') as f:
    input_data = pickle.load(f)

print(type(input_data))

attributes = {
    "id": 1,
    "name": "polygon",
    "valid": True,
    'interpolation_distance': 0.0001,
    "multiprocess": True,
    "run": False,
    "save_to_file": file_name + "_Voronoi"
}


@st_time
def get_centerline():
    centerline_simple = Centerline(input_data, **attributes)
    return centerline_simple


output_data = get_centerline()

print(type(output_data))

with open(file_name + '_centerline.pkl', 'wb') as f:
    pickle.dump(output_data, f)

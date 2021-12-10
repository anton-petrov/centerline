from __future__ import print_function

from multiprocessing import Process
import sys
import os
import re
import pickle
import more_itertools as mit
from tqdm.auto import tqdm
from p_tqdm import p_map, p_umap, p_imap, p_uimap
from shapely.geometry import Polygon
from centerline.geometry import Centerline
import time
import typer

app = typer.Typer()


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


@st_time
def extract_voronoi_diagram(file):
    with open(file_name + '.pkl', 'rb') as f:
        input_data = pickle.load(f)

    attributes = {
        "id": 1,
        "name": "polygon",
        "valid": True,
        'interpolation_distance': 0.0001,
        "multiprocess": False,
        "run": False,
        "save_to_file": file
    }
    centerline = Centerline(input_data, **attributes)
    return centerline._dump_voronoi_vertices_and_ridges(file)


@st_time
def slice_voronoi(file, n=10):
    with open(file + '_vr.pkl', 'rb') as f:
        vertices_and_ridges = pickle.load(f)
    vertices, ridges = vertices_and_ridges

    ridges = [list(r) for r in mit.divide(n, ridges)]

    for i, ridges_chunk in enumerate(ridges):
        with open(file + f'_vr{i}.pkl', 'wb') as f:
            pickle.dump((vertices, ridges_chunk), f)
    pass


def get_centerline(*args):
    file, n = args[0]
    with open(file + f'_vr{n}.pkl', 'rb') as f:
        vertices_and_ridges = pickle.load(f)
    with open(file + f'.pkl', 'rb') as f:
        input_data = pickle.load(f)
    attributes = {
        "id": 1,
        "name": "polygon",
        "valid": True,
        'interpolation_distance': 0.0001,
        "multiprocess": False,
        "run": True,
        "save_to_file": file + "{n}"
    }
    centerline = Centerline(input_data, vertices_and_ridges, **attributes)
    return centerline


@app.command()
def dump(file_name):
    typer.echo(f"Extracting Voronoi diagram from `{file_name}`.")
    v, r = extract_voronoi_diagram(file_name)
    typer.echo(f"Extracted {len(v)} vertices and {len(r)} ridges!")


@app.command()
def slice(file_name: str, n: int):
    typer.echo(f"Slicing Voronoi diagram from {file_name} into {n} files.")
    slice_voronoi(file_name, n)
    typer.echo(f"Finished ok.")


@app.command()
def start(file_name: str, n: int):
    processes = []

    results = p_map(get_centerline, [(file_name, i) for i in range(n)])

    typer.echo(f"{len(results)} centerline's processed.")

    # for i in range(0, n):
    #     p = Process(target=get_centerline, args=(file_name, i))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()


if __name__ == "__main__":
    app()

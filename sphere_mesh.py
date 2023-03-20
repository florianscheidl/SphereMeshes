import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import scipy

from icosphere import icosphere
from itertools import combinations
from tqdm import tqdm


# copied from https://code.activestate.com/recipes/578276-3d-polygon-area/
def poly_area(poly):
    if len(poly) < 3:  # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i + 1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result / 2)


# unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1, a[1], a[2]],
                       [1, b[1], b[2]],
                       [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]],
                       [b[0], 1, b[2]],
                       [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1],
                       [b[0], b[1], 1],
                       [c[0], c[1], 1]])
    magnitude = (x ** 2 + y ** 2 + z ** 2) ** .5
    return (x / magnitude, y / magnitude, z / magnitude)


def reorder_points(points):

    # Calculate the center of the points
    c = points.mean(axis=0)
    c = c / np.linalg.norm(c, ord=2)
    v1 = np.array([-c[1], c[0], 0])
    v1 = v1 / np.linalg.norm(v1, ord=2)

    # print(c.shape, v1.shape)

    v2 = np.cross(c, v1)
    pmc = points - c.reshape((1, 3))
    pmc0 = (pmc * v1).sum(axis=-1)
    pmc1 = (pmc * v2).sum(axis=-1)

    # Calculate the angles between the horizontal and the line joining center to each point
    angles = np.arctan2(pmc1, pmc0)

    return np.argsort(angles).tolist()


def build_dual(vertices: np.array, faces: list) -> (np.array, list):
    """
    Given a geodesic polyhedron (as vertices and faces), computes the dual (i.e. a Goldberg polyhedron). The dual operation build a new set
    of vertices and faces as follows. A vertex in the dual is formed by the centers of each face in the original
    polyhedron. The faces of the dual are formed by connecting each set of new vertices for which the corresponding
    faces touched in the original polyhedron
    :return: vertices (coordinates), faces (indices of vertices they connect)

    :param vertices: numpy array of three-dimensional coordinates, shape: (n_vertices, 3)
    :param faces: list of vertex indices
    :return: new array of vertices and list of faces
    """

    vertices_dual = np.concatenate([vertices[face].mean(axis=0).reshape(1, 3) for face in faces], axis=0)
    vertices_dual = vertices_dual / np.linalg.norm(vertices_dual, axis=-1, ord=2).reshape((-1, 1))

    ncell = len(faces)
    nvert = vertices.shape[0]

    indices = np.array([vert for face in faces for vert in face])
    data = np.ones(len(indices))
    indptr = np.cumsum(np.array([0]+[len(face) for face in faces]))
    adj_matrix_dual = scipy.sparse.csr_matrix((data, indices, indptr), shape=(ncell, nvert)).transpose().tocsr()

    faces_dual = [adj_matrix_dual.indices[adj_matrix_dual.indptr[j]: adj_matrix_dual.indptr[j + 1]] for j in
                  range(nvert)]
    faces_dual = [pl[reorder_points(vertices_dual[pl])] for pl in faces_dual]

    return vertices_dual, faces_dual


class SphereMesh:

    def __init__(self, type: str = "icosphere", level: int = 1):

        self.level = level
        self.type = type

        # vertices and faces
        self.vertices, self.faces = icosphere(nu=level)
        self.faces = list(self.faces)

        if type == "goldberg_polyhedron":
            self.vertices, self.faces = build_dual(self.vertices, self.faces)

        # to assert that duality is an involution
        if type == "icosphere":
            tmp_vertices, tmp_faces = build_dual(self.vertices, self.faces)
            self.vertices, self.faces = build_dual(tmp_vertices, tmp_faces)

        # face areas
        self.face_areas = []
        for face in self.faces:
            face_vertices = [self.vertices[j] for j in face]
            self.face_areas.append(poly_area(face_vertices))
        self.face_areas = np.array(self.face_areas)


        # edges:
        self.edges_by_face_indices = []
        self.edges_by_vertex_indices = []
        for i, j in combinations(range(len(self.faces)), 2):
            face_intersection = set([i for i in set.intersection(set(self.faces[i]), set(self.faces[j])) if i!=0])
            if len(face_intersection) == 2:
                self.edges_by_face_indices.append([i, j])  # could potentially use set here instead ...
                # edges_by_face_indices.append({i,j}) # in set format ...
                self.edges_by_vertex_indices.append(list(face_intersection))
        self.edges_by_face_indices = np.array(self.edges_by_face_indices)
        self.edges_by_vertex_indices = np.array(self.edges_by_vertex_indices)

    def visualise(self, show_midpoints: bool = False):

        fig = plt.figure()

        # creating mesh
        poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection([np.array([self.vertices[v_idx] for v_idx in face]) for face in self.faces])

        poly.set_edgecolor('black')
        poly.set_linewidth(0.25)

        # would be nice to visualise edges (to make sure what we defined as edges is the right thing)

        # and now -- visualization!
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.add_collection3d(poly)
        ax.scatter(np.array(self.vertices)[:,0],np.array(self.vertices)[:,1],np.array(self.vertices)[:,2], marker = "o", color="green", s=40)

        ax.set_title(f'{self.type} with subdivision frequency {self.level}.')

        plt.show()


if __name__ == "__main__":
    # mesh = SphereMesh(level=2)
    mesh = SphereMesh(type="goldberg_polyhedron", level=3)
    mesh.visualise(show_midpoints=True)

    print("Done.")
    # mesh.visualise(show_midpoints=True)
